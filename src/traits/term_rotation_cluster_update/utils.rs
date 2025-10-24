use crate::traits::term_rotation_cluster_update::ClusterWeights;
use crate::traits::term_rotation_cluster_update::sampling::SampleFraction;
use num_traits::One;
use rand::Rng;
use std::cmp::min;
use std::fmt::Debug;
use std::ops::{Add, Mul};

pub(crate) fn allocate_terms_to_timeslices<T, P, R>(
    weights: &[ClusterWeights<T, P>],
    flip_label_index: usize,
    mut rng: R,
) -> Vec<&T>
where
    R: Rng,
    P: Mul<P, Output = P> + One + Clone + Debug + SampleFraction<P>,
    for<'a> P: Add<&'a P, Output = P>,
    for<'a, 'b> &'a P: Mul<&'b P, Output = P>,
{
    let total_num_terms = weights.iter().map(|w| w.n_occupied_slices).sum::<usize>();
    let mut weight_arrays = vec![];

    let mut last_weights = vec![P::one()];
    weight_arrays.push(last_weights.clone());
    for w in weights {
        for _ in &w.timeslices {
            get_next_weight_array_from_previous(
                total_num_terms,
                &mut last_weights,
                &w.weights_per_flip_label[flip_label_index],
            );
            weight_arrays.push(last_weights.clone());
        }
    }

    let mut timeslices = vec![];
    let timeslices_iter = weights.iter().rev().flat_map(|w| w.timeslices.iter().rev());
    for ((t, warr), wnextarr) in timeslices_iter
        .zip(weight_arrays.iter().rev())
        .zip(weight_arrays[..weight_arrays.len() - 1].iter().rev())
    {
        let terms_to_allocate = total_num_terms - timeslices.len();

        // The weight comes from up to two sources, randomly choose one based on the related
        // contribution. If the choice is an insertion, mark the timeslices in the timeslices array.
        // There are some unnecessary clones on the weights due to limitations in the trait solver.
        let total_weight = warr[terms_to_allocate].clone();
        match terms_to_allocate {
            0 => break,
            // If wnextarr only supports (terms_to_allocate-1) possible allocations, then we must.
            terms_to_allocate if terms_to_allocate == wnextarr.len() => {
                // Must allocate term
                timeslices.push(t);
            }
            terms_to_allocate => {
                let weight_from_no_allocation = wnextarr[terms_to_allocate].clone();
                let no_allocate = weight_from_no_allocation.sample(total_weight, &mut rng);
                if !no_allocate {
                    timeslices.push(t);
                }
            }
        }
    }

    timeslices.reverse();
    timeslices
}

pub(crate) fn get_weights_for_flips<T, P>(
    n_flip_labels: usize,
    weights: &[ClusterWeights<T, P>],
) -> impl IntoIterator<Item = P>
where
    P: Mul<P, Output = P> + One + Clone + Debug,
    for<'a> P: Add<&'a P, Output = P>,
    for<'a, 'b> &'a P: Mul<&'b P, Output = P>,
{
    let total_num_terms = weights.iter().map(|w| w.n_occupied_slices).sum::<usize>();
    (0..n_flip_labels)
        .map(move |flip_label| get_weight_for_flip_label(weights, flip_label, total_num_terms))
}

fn get_weight_for_flip_label<T, P>(
    weights: &[ClusterWeights<T, P>],
    flip_label_index: usize,
    total_num_terms: usize,
) -> P
where
    P: Mul<P, Output = P> + One + Clone + Debug,
    for<'a> P: Add<&'a P, Output = P>,
    for<'a, 'b> &'a P: Mul<&'b P, Output = P>,
{
    debug_assert_eq!(
        total_num_terms,
        weights.iter().map(|w| w.n_occupied_slices).sum::<usize>()
    );

    let mut last_weights = vec![P::one()];
    for w in weights {
        for _ in &w.timeslices {
            get_next_weight_array_from_previous(
                total_num_terms,
                &mut last_weights,
                &w.weights_per_flip_label[flip_label_index],
            );
        }
    }
    debug_assert_eq!(last_weights.len(), total_num_terms + 1);
    last_weights
        .pop()
        .expect("Vector must be of length at least 1")
}

/// Reads previous set of weights from the buffer then writes the new set.
fn get_next_weight_array_from_previous<P>(
    total_num_terms: usize,
    last_weights_buffer: &mut Vec<P>,
    insertion_weight: &P,
) where
    P: Mul<P, Output = P> + One + Clone + Debug,
    for<'a> P: Add<&'a P, Output = P>,
    for<'a, 'b> &'a P: Mul<&'b P, Output = P>,
{
    let old_vec_size = last_weights_buffer.len();
    let new_vec_size = min(last_weights_buffer.len(), total_num_terms) + 1;
    last_weights_buffer.resize(new_vec_size, P::one());
    for n_terms in (0..new_vec_size).rev() {
        // Written to avoid P::zero() so we can work in logspace if desired.
        let new_weight = match n_terms {
            n if n > 0 && n < old_vec_size => {
                (&last_weights_buffer[n - 1] * insertion_weight) + &last_weights_buffer[n]
            }
            n if n > 0 => &last_weights_buffer[n - 1] * insertion_weight,
            n if n < old_vec_size => last_weights_buffer[n].clone(),
            _ => unreachable!(),
        };
        last_weights_buffer[n_terms] = new_weight;
    }
}

#[cfg(test)]
mod cluster_flip_tests {
    use super::*;
    use crate::utils::logwrapper::LogWrapper;

    #[test]
    fn test_single_sector_weights_simple() {
        let n_flip_labels = 2;
        let initial_weights = vec![1, 2];
        let weights = vec![ClusterWeights::<(), _> {
            timeslices: vec![()],
            n_occupied_slices: 1,
            weights_per_flip_label: initial_weights.clone(),
        }];

        let final_weights = get_weights_for_flips(n_flip_labels, &weights)
            .into_iter()
            .collect::<Vec<_>>();

        assert_eq!(final_weights, initial_weights);
    }

    #[test]
    fn test_two_sector_weights() {
        let n_flip_labels = 2;
        let initial_weights = vec![1, 2];
        let weights = vec![
            ClusterWeights::<(), _> {
                timeslices: vec![()],
                n_occupied_slices: 1,
                weights_per_flip_label: initial_weights.clone(),
            },
            ClusterWeights::<(), _> {
                timeslices: vec![()],
                n_occupied_slices: 1,
                weights_per_flip_label: initial_weights.clone(),
            },
        ];
        let final_weights = get_weights_for_flips(n_flip_labels, &weights)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(final_weights, vec![1 * 1, 2 * 2]);
    }

    #[test]
    fn test_two_sector_weights_single_term() {
        let n_flip_labels = 2;
        let initial_weights = vec![1, 3];
        let weights = vec![
            ClusterWeights::<(), _> {
                timeslices: vec![()],
                n_occupied_slices: 1,
                weights_per_flip_label: initial_weights.clone(),
            },
            ClusterWeights::<(), _> {
                timeslices: vec![()],
                n_occupied_slices: 0,
                weights_per_flip_label: initial_weights.clone(),
            },
        ];
        let final_weights = get_weights_for_flips(n_flip_labels, &weights)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(final_weights, vec![1 + 1, 3 + 3]);

        let weights = vec![
            ClusterWeights::<(), _> {
                timeslices: vec![()],
                n_occupied_slices: 0,
                weights_per_flip_label: initial_weights.clone(),
            },
            ClusterWeights::<(), _> {
                timeslices: vec![()],
                n_occupied_slices: 1,
                weights_per_flip_label: initial_weights.clone(),
            },
        ];
        let final_weights = get_weights_for_flips(n_flip_labels, &weights)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(final_weights, vec![1 + 1, 3 + 3]);
    }

    #[test]
    fn test_multi_timeslice_sector() {
        let n_flip_labels = 2;
        let n_timeslices = 3;
        let initial_weights = vec![1, 2];
        let weights = vec![ClusterWeights::<(), _> {
            timeslices: vec![(); n_timeslices],
            n_occupied_slices: 1,
            weights_per_flip_label: initial_weights.clone(),
        }];
        let final_weights = get_weights_for_flips(n_flip_labels, &weights)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(final_weights, vec![n_timeslices, n_timeslices * 2]);
        let weights = vec![ClusterWeights::<(), _> {
            timeslices: vec![(); n_timeslices],
            n_occupied_slices: n_timeslices - 1,
            weights_per_flip_label: initial_weights.clone(),
        }];
        let final_weights = get_weights_for_flips(n_flip_labels, &weights)
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(
            final_weights,
            vec![
                n_timeslices * 1_usize.pow((n_timeslices - 1) as u32),
                n_timeslices * 2_usize.pow((n_timeslices - 1) as u32)
            ]
        );
    }

    #[test]
    fn test_single_sector_weights_log() {
        let n_flip_labels = 2;
        let initial_weights = vec![LogWrapper::new(1.), LogWrapper::new(2.)];
        let weights = vec![ClusterWeights::<(), _> {
            timeslices: vec![()],
            n_occupied_slices: 1,
            weights_per_flip_label: initial_weights.clone(),
        }];

        let final_weights = get_weights_for_flips(n_flip_labels, &weights)
            .into_iter()
            .collect::<Vec<_>>();

        assert_eq!(final_weights, initial_weights);
    }

    #[test]
    fn test_allocate_timeslices() {
        let weights = vec![
            ClusterWeights::<_, _> {
                timeslices: vec![1, 2, 3, 4],
                n_occupied_slices: 2,
                weights_per_flip_label: vec![1_usize],
            },
            ClusterWeights::<_, _> {
                timeslices: vec![11, 12, 13, 14],
                n_occupied_slices: 0,
                weights_per_flip_label: vec![100_usize],
            },
        ];
        let mut rng = rand::rng();
        let allocations = allocate_terms_to_timeslices(&weights, 0, &mut rng);
        assert_eq!(allocations.len(), 2);
        assert!(allocations.into_iter().all(|x| *x > 10));
    }

    #[test]
    fn test_allocate_timeslices_logprob() {
        let weights = vec![
            ClusterWeights::<_, _> {
                timeslices: vec![1, 2, 3, 4],
                n_occupied_slices: 2,
                weights_per_flip_label: vec![LogWrapper::new(1.0f64)],
            },
            ClusterWeights::<_, _> {
                timeslices: vec![11, 12, 13, 14],
                n_occupied_slices: 0,
                weights_per_flip_label: vec![LogWrapper::new(100.0f64)],
            },
        ];
        let mut rng = rand::rng();
        let allocations = allocate_terms_to_timeslices(&weights, 0, &mut rng);
        assert_eq!(allocations.len(), 2);
        assert!(allocations.into_iter().all(|x| *x > 10));
    }
}

use crate::traits::graph_traits::{GraphNode, LinkedGraphNode, TimeSlicedGraph};
use crate::traits::graph_weights::GraphWeight;
use num_traits::{One, Zero};
use rand::Rng;
use std::cmp::min;
use std::fmt::Debug;
use std::ops::{AddAssign, Mul};

pub trait MatrixTermRotationUpdate: TimeSlicedGraph + GraphWeight
where
    Self::Node: LinkedGraphNode,
{
    type SliceContext;
    type Cluster;
    type ClusterSlice: ClusterSliceData<Self::TimesliceIndex>;
    type ClusterFlipLabel;

    fn get_classical_slice_context(&self, timeslice: &Self::TimesliceIndex) -> Self::SliceContext;

    /// Returns none only if there are no operators in this worldline.
    /// Otherwise returns node and relative index.
    fn get_starting_operator_for_flippable_region_on_index(
        &self,
        context: &Self::SliceContext,
        index: &Self::DOFIndex,
    ) -> Option<(&Self::Node, usize)>;

    /// Returns none only if there are no operators in this worldline.
    /// Otherwise returns node and relative index.
    fn get_ending_operator_for_flippable_region_on_index(
        &self,
        context: &Self::SliceContext,
        index: &Self::DOFIndex,
    ) -> Option<(&Self::Node, usize)>;

    fn get_cluster_from_starting_index<R: Rng>(
        &self,
        context: &Self::SliceContext,
        index: &Self::DOFIndex,
        rng: &mut R,
    ) -> Self::Cluster;

    fn get_indices_in_cluster_at_timeslice(
        &self,
        context: &Self::SliceContext,
        cluster: &Self::Cluster,
        timeslice: &Self::TimesliceIndex,
    ) -> impl IntoIterator<Item = Self::DOFIndex>;

    fn get_flip_possibilities_for_cluster(
        &self,
        context: &Self::SliceContext,
        cluster: &Self::Cluster,
    ) -> impl IntoIterator<Item = Self::ClusterFlipLabel>;

    fn get_dof_value_for_index_and_flip(
        &self,
        context: &Self::SliceContext,
        index: &Self::DOFIndex,
        flip: &Self::ClusterFlipLabel,
    ) -> Self::DOFType;

    fn get_slice_for_start_of_cluster(&self, cluster: &Self::Cluster) -> Self::ClusterSlice;
    /// Returns the next slice, or None if the end of the cluster is reached.
    /// Should never return the start of cluster slice.
    fn get_next_slice(
        &self,
        cluster: &Self::Cluster,
        existing_cluster_slice: Self::ClusterSlice,
    ) -> Option<Self::ClusterSlice>;

    /// Using the output weights of any nodes at this timeslices, calculate the total weight of all diagonal terms which could be added to the state.
    fn get_weight_associated_with_term_at_slice_with_flip(
        &self,
        cluster: &Self::Cluster,
        cluster_slice: &Self::ClusterSlice,
        flip: &Self::ClusterFlipLabel,
        term: &Self::MatrixTerm,
    ) -> f64;

    fn get_total_weight_for_cluster_flip(
        &self,
        cluster: &Self::Cluster,
        cluster_slice: &Self::ClusterSlice,
        flip: &Self::ClusterFlipLabel,
    ) -> f64 {
        self.get_possible_terms()
            .iter()
            .map(|term| {
                self.get_weight_associated_with_term_at_slice_with_flip(
                    cluster,
                    cluster_slice,
                    flip,
                    term,
                )
            })
            .sum::<f64>()
    }

    fn get_slices_and_cluster_weights<M>(
        &self,
        context: &Self::SliceContext,
        cluster: &Self::Cluster,
    ) -> Vec<ClusterWeights<Self::TimesliceIndex>> {
        let mut cluster_slice = self.get_slice_for_start_of_cluster(cluster);
        let first_timeslice = cluster_slice.get_timeslice().clone();

        let mut sections = vec![];
        let mut section = ClusterWeights::default();
        loop {
            let nodetype = NodeType::from(self.get_node(cluster_slice.get_timeslice()));
            if matches!(nodetype, NodeType::Diagonal | NodeType::Empty) {
                if section.weights_per_flip_label.is_empty() {
                    section.weights_per_flip_label = self
                        .get_flip_possibilities_for_cluster(context, cluster)
                        .into_iter()
                        .map(|flip| {
                            self.get_total_weight_for_cluster_flip(cluster, &cluster_slice, &flip)
                        })
                        .collect();
                }
                if matches!(nodetype, NodeType::Diagonal) {
                    section.n_occupied_slices += 1;
                }
                section
                    .timeslices
                    .push(cluster_slice.get_timeslice().clone());
            } else {
                sections.push(section);
                section = ClusterWeights::default();
            }

            cluster_slice = if let Some(cluster_slice) = self.get_next_slice(cluster, cluster_slice)
            {
                cluster_slice
            } else {
                if !section.timeslices.is_empty() {
                    sections.push(section);
                }
                break sections;
            };
            debug_assert_ne!(
                cluster_slice.get_timeslice(),
                &first_timeslice,
                "Cannot cycle back to start of cluster."
            );
        }
    }

    fn get_context_and_cluster_from_index_at_slice<R>(
        &self,
        timeslice: &Self::TimesliceIndex,
        starting_index: &Self::DOFIndex,
        rng: &mut R,
    ) -> (Self::SliceContext, Self::Cluster)
    where
        R: Rng,
    {
        let context = self.get_classical_slice_context(timeslice);
        let cluster = self.get_cluster_from_starting_index(&context, starting_index, rng);
        (context, cluster)
    }

    fn perform_cluster_update_on_index_at_slice<R>(
        &mut self,
        timeslice: &Self::TimesliceIndex,
        starting_index: &Self::DOFIndex,
        rng: &mut R,
    ) -> bool
    where
        R: Rng,
    {
        let (context, cluster) =
            self.get_context_and_cluster_from_index_at_slice(timeslice, starting_index, rng);

        todo!()
    }

    fn perform_cluster_update<R>(
        &mut self,
        cluster: Self::Cluster,
        cluster_flip_label: Self::ClusterFlipLabel,
    ) -> bool;
}

pub struct ClusterWeights<T, P = f64> {
    timeslices: Vec<T>,
    n_occupied_slices: usize,
    weights_per_flip_label: Vec<P>,
}

impl<T, P> Default for ClusterWeights<T, P> {
    fn default() -> Self {
        Self {
            timeslices: vec![],
            n_occupied_slices: 0,
            weights_per_flip_label: vec![],
        }
    }
}

pub trait ClusterSliceData<T> {
    fn get_timeslice(&self) -> &T;

    fn get_number_of_cluster_dof(&self) -> usize;
}

enum NodeType {
    Diagonal,
    Offdiagonal,
    Empty,
}

impl<GN> From<Option<&GN>> for NodeType
where
    GN: GraphNode,
{
    fn from(value: Option<&GN>) -> Self {
        match value {
            None => Self::Empty,
            Some(value) if value.is_diagonal() => Self::Diagonal,
            _ => Self::Offdiagonal,
        }
    }
}

impl<GN> From<&GN> for NodeType
where
    GN: GraphNode,
{
    fn from(value: &GN) -> Self {
        match value {
            value if value.is_diagonal() => Self::Diagonal,
            _ => Self::Offdiagonal,
        }
    }
}

fn get_term_allocation_for_weights<'a, F, T, R>(
    flip_labels: &'a [F],
    weights: &[ClusterWeights<T>],
    rng: &mut R,
) -> (&'a F, Vec<usize>)
where
    R: Rng,
{
    let total_num_terms = weights.iter().map(|w| w.n_occupied_slices).sum::<usize>();
    let total_weights = get_weights_for_flips(flip_labels.len(), weights);
    let normalization = total_weights.iter().sum::<f64>();
    let mut choice = rng.random::<f64>() * normalization;
    let (flip_choice, _) = flip_labels
        .iter()
        .zip(total_weights)
        .find(|(_label, weight)| {
            choice -= weight;
            choice <= 0.0
        })
        .expect("Must pick one of the options");

    todo!();
    (flip_choice, vec![])
}

fn get_weights_for_flips<T, P>(n_flip_labels: usize, weights: &[ClusterWeights<T, P>]) -> Vec<P>
where
    P: Mul<P, Output = P> + AddAssign<P> + One + Zero + Debug,
    for<'a> P: AddAssign<&'a P>,
    for<'a, 'b> &'a P: Mul<&'b P, Output = P>,
{
    let total_num_terms = weights.iter().map(|w| w.n_occupied_slices).sum::<usize>();

    (0..n_flip_labels)
        .map(|flip_label| {
            let mut last_weights = vec![P::one()];
            for w in weights {
                for _ in &w.timeslices {
                    // Can add at most a single term per t
                    last_weights = (0..=min(last_weights.len(), total_num_terms))
                        .map(|n_terms| {
                            let mut weight = P::zero();
                            if n_terms > 0 {
                                weight += &last_weights[n_terms - 1]
                                    * &w.weights_per_flip_label[flip_label];
                            }
                            if n_terms < last_weights.len() {
                                weight += &last_weights[n_terms];
                            }
                            weight
                        })
                        .collect();
                }
            }
            debug_assert_eq!(last_weights.len(), total_num_terms + 1);
            last_weights
                .pop()
                .expect("Vector must be of length at least 1")
        })
        .collect::<Vec<P>>()
}

#[cfg(test)]
mod cluster_flip_tests {
    use super::*;

    #[test]
    fn test_single_sector_weights() {
        let n_flip_labels = 2;
        let initial_weights = vec![1., 2.];
        let weights = vec![ClusterWeights::<(), _> {
            timeslices: vec![()],
            n_occupied_slices: 1,
            weights_per_flip_label: initial_weights.clone(),
        }];

        let final_weights = get_weights_for_flips(n_flip_labels, &weights);

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
        let final_weights = get_weights_for_flips(n_flip_labels, &weights);
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
        let final_weights = get_weights_for_flips(n_flip_labels, &weights);
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
        let final_weights = get_weights_for_flips(n_flip_labels, &weights);
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
        let final_weights = get_weights_for_flips(n_flip_labels, &weights);
        assert_eq!(final_weights, vec![n_timeslices, n_timeslices * 2]);
        let weights = vec![ClusterWeights::<(), _> {
            timeslices: vec![(); n_timeslices],
            n_occupied_slices: n_timeslices - 1,
            weights_per_flip_label: initial_weights.clone(),
        }];
        let final_weights = get_weights_for_flips(n_flip_labels, &weights);
        assert_eq!(
            final_weights,
            vec![
                n_timeslices * 1_usize.pow((n_timeslices - 1) as u32),
                n_timeslices * 2_usize.pow((n_timeslices - 1) as u32)
            ]
        );
    }
}

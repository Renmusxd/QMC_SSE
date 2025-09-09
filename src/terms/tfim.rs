use std::fmt::Debug;
use std::ops::{Add, Neg, Sub};
use num_traits::{One, Signed, Zero};
use rand::Rng;
use crate::qmc::cluster_impl::TermClusterExpander;
use crate::qmc::MatrixTermData;
use crate::qmc::naive_flip_impl::MatrixTermFlippable;
use crate::traits::cluster_update::{DirectionEnum, NodeClusterExpansion};
use crate::traits::graph_traits::DOFTypeTrait;

pub enum TFIMTerm<T> {
    Ising(T),
    Field(T)
}

impl<T> MatrixTermData<T> for TFIMTerm<T>
where
    T: Clone + Zero + Signed + Add + Debug,
{
    fn get_matrix_entry(&self, input: usize, output: usize) -> T {
        match self {
            TFIMTerm::Ising(jj) => {
                let weight = get_ising_weight(input, output, jj);
                weight.unwrap_or(T::zero())
            }
            TFIMTerm::Field(gamma) => gamma.clone(),
        }
    }

    fn dim(&self) -> usize {
        match self {
            TFIMTerm::Ising(_) => 4,
            TFIMTerm::Field(_) => 2,
        }
    }

    fn get_weight_change_for_diagonal(&self, old_state: usize, new_state: usize) -> Option<(T, T)> {
        match self {
            TFIMTerm::Field(_) => None,
            TFIMTerm::Ising(jj) => {
                let old_state_arr = bool::index_to_state::<2>(old_state);
                let old_ferro = old_state_arr[0] == old_state_arr[1];
                let new_state_arr = bool::index_to_state::<2>(new_state);
                let new_ferro = new_state_arr[0] == new_state_arr[1];
                if old_ferro == new_ferro {
                    None
                } else {
                    let result = Some((jj.clone() + jj.clone(), T::zero()));

                    debug_assert_eq!({
                        let old_weight = self.get_matrix_entry(old_state, old_state);
                        let new_weight = self.get_matrix_entry(new_state, new_state);
                        Some((old_weight, new_weight))
                    }, result.clone());

                    result
                }
            }
        }
    }

    fn get_number_of_equal_weight_outputs_for_input_distinct_from_output(&self, input: usize, output: usize) -> usize {
        match self {
            TFIMTerm::Ising(_) => 0,
            TFIMTerm::Field(_) => 1,
        }
    }
}

impl<T> MatrixTermFlippable<T> for TFIMTerm<T> where T: Zero + Clone + Signed
{
    fn is_maybe_flippable(&self) -> bool {
        matches!(self, TFIMTerm::Field(_))
    }

    fn get_weights_for_inputs_given_output(&self, input_a: usize, input_b: usize, output: usize) -> Option<(T, T)> {
        match self {
            TFIMTerm::Ising(jj) => {
                if input_a == input_b {
                    None
                } else {
                    let weight_a = if input_a == output { get_ising_weight(input_a, input_a, jj) } else { None };
                    let weight_b = if input_b == output { get_ising_weight(input_b, input_b, jj) } else { None };

                    match (weight_a, weight_b) {
                        (None, None) => None,
                        (Some(x), None) => Some((x, T::zero())),
                        (None, Some(x)) => Some((T::zero(), x)),
                        (Some(x), Some(y)) => Some((x, y)),
                    }
                }
            }
            TFIMTerm::Field(_) => {
                None
            }
        }
    }

    fn get_nth_equal_weight_output_for_input_distinct_from_output(&self, _input: usize, output: usize, n: usize) -> usize {
        match self {
            TFIMTerm::Ising(_) =>  unimplemented!("There are no equal weight outputs for a given input."),
            TFIMTerm::Field(_) => {
                debug_assert_eq!(n, 0);
                1 - output
            }
        }
    }
}

impl<T> TermClusterExpander<bool> for TFIMTerm<T> {
    fn output_changes_for_spin_flip<'a, R>(&self, _: &[bool], _: &[bool], direction: DirectionEnum, relative_index: usize, new_value: &bool, _: &mut R) -> impl NodeClusterExpansion<bool> + 'a
    where
        R: Rng
    {
        match self {
            TFIMTerm::Ising(_) => {
                let other_direction = direction.swap_direction();
                let other_index = 1 - relative_index;
                vec![
                    (direction, other_index, *new_value),
                    (other_direction, relative_index, *new_value),
                    (other_direction, other_index, *new_value),
                ]
            }
            TFIMTerm::Field(_) => {
                vec![]
            }
        }
    }
}

fn get_ising_weight<T>(input: usize, output: usize, scale: &T) -> Option<T> where T: Clone + Neg<Output=T> + Signed + Add {
    if input != output {
        None
    } else {
        let state = bool::index_to_state::<2>(input);
        let bond = if state[0] == state[1] {
            - scale.clone()
        } else {
            scale.clone()
        };
        let result = bond + scale.abs();
        if result == T::zero() {
            None
        } else {
            Some(result)
        }
    }
}
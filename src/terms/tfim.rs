use crate::qmc::MatrixTermData;
use crate::qmc::cluster_impl::TermClusterExpander;
use crate::qmc::naive_flip_impl::MatrixTermFlippable;
use crate::traits::cluster_update::{DirectionEnum, NodeClusterExpansion};
use crate::traits::graph_traits::DOFTypeTrait;
use num_traits::{Signed, Zero};
use rand::Rng;
use std::fmt::Debug;
use std::ops::{Add, Neg};

#[derive(Debug, Clone, Copy)]
pub enum TFIMTerm<T> {
    ZZ(T),
    X(T),
}

impl<T> MatrixTermData<T> for TFIMTerm<T>
where
    T: Clone + Zero + Signed + Add + Debug,
{
    fn get_matrix_entry(&self, input: usize, output: usize) -> T {
        match self {
            TFIMTerm::ZZ(jj) => {
                let weight = get_ising_weight(input, output, jj);
                weight.unwrap_or(T::zero())
            }
            TFIMTerm::X(gamma) => gamma.clone(),
        }
    }

    fn dim(&self) -> usize {
        match self {
            TFIMTerm::ZZ(_) => 4,
            TFIMTerm::X(_) => 2,
        }
    }

    fn get_number_of_equal_weight_outputs_for_input_distinct_from_output(
        &self,
        _input: usize,
        _output: usize,
    ) -> usize {
        match self {
            TFIMTerm::ZZ(_) => 0,
            TFIMTerm::X(_) => 1,
        }
    }

    fn get_weight_change_for_diagonal(&self, old_state: usize, new_state: usize) -> Option<(T, T)> {
        match self {
            TFIMTerm::X(_) => None,
            TFIMTerm::ZZ(jj) => {
                let old_state_arr = bool::index_to_state::<2>(old_state);
                let old_ferro = old_state_arr[0] == old_state_arr[1];
                let new_state_arr = bool::index_to_state::<2>(new_state);
                let new_ferro = new_state_arr[0] == new_state_arr[1];
                if old_ferro == new_ferro {
                    None
                } else {
                    let result = Some((jj.clone() + jj.clone(), T::zero()));

                    debug_assert_eq!(
                        {
                            let old_weight = self.get_matrix_entry(old_state, old_state);
                            let new_weight = self.get_matrix_entry(new_state, new_state);
                            Some((old_weight, new_weight))
                        },
                        result.clone()
                    );

                    result
                }
            }
        }
    }

    fn get_natural_offset(&self) -> T {
        match self {
            TFIMTerm::ZZ(jj) => jj.abs(),
            TFIMTerm::X(gamma) => gamma.clone()
        }
    }
}

impl<T> MatrixTermFlippable<T> for TFIMTerm<T>
where
    T: Zero + Clone + Signed,
{
    fn is_maybe_flippable(&self) -> bool {
        matches!(self, TFIMTerm::X(_))
    }

    fn get_weights_for_inputs_given_output(
        &self,
        input_a: usize,
        input_b: usize,
        output: usize,
    ) -> Option<(T, T)> {
        match self {
            TFIMTerm::ZZ(jj) => {
                if input_a == input_b {
                    None
                } else {
                    let weight_a = if input_a == output {
                        get_ising_weight(input_a, input_a, jj)
                    } else {
                        None
                    };
                    let weight_b = if input_b == output {
                        get_ising_weight(input_b, input_b, jj)
                    } else {
                        None
                    };

                    match (weight_a, weight_b) {
                        (None, None) => None,
                        (Some(x), None) => Some((x, T::zero())),
                        (None, Some(x)) => Some((T::zero(), x)),
                        (Some(x), Some(y)) => Some((x, y)),
                    }
                }
            }
            TFIMTerm::X(_) => None,
        }
    }

    fn get_nth_equal_weight_output_for_input_distinct_from_output(
        &self,
        _input: usize,
        output: usize,
        n: usize,
    ) -> usize {
        match self {
            TFIMTerm::ZZ(_) => {
                unimplemented!("There are no equal weight outputs for a given input.")
            }
            TFIMTerm::X(_) => {
                debug_assert_eq!(n, 0);
                1 - output
            }
        }
    }
}

impl<T> TermClusterExpander<bool> for TFIMTerm<T> {
    fn output_changes_for_spin_flip<'a, R>(
        &self,
        input: &[bool],
        output: &[bool],
        direction: DirectionEnum,
        relative_index: usize,
        _new_value: &bool,
        _: &mut R,
    ) -> impl NodeClusterExpansion<bool> + 'a
    where
        R: Rng,
    {
        // Rather than return one of two vectors, just construct a small array and return the
        // first n={0, 3} objects.
        // The compiler may be able to optimize this since there are no heap interactions.
        let (own_state, other_state) = match direction {
            DirectionEnum::Input => (input, output),
            DirectionEnum::Output => (output, input)
        };
        let other_direction = direction.swap_direction();


        let num_to_select = match self {
            TFIMTerm::ZZ(_) => 3,
            TFIMTerm::X(_) => 0,
        };

        let other_index = if num_to_select == 3 { 1 - relative_index } else { 0 };

        let ising_array = [
            (direction, other_index, !own_state[other_index]),
            (other_direction, relative_index, !other_state[relative_index]),
            (other_direction, other_index, !other_state[other_index]),
        ];

        ising_array.into_iter().take(num_to_select)
    }
}

fn get_ising_weight<T>(input: usize, output: usize, scale: &T) -> Option<T>
where
    T: Clone + Neg<Output = T> + Signed + Add,
{
    if input != output {
        None
    } else {
        let state = bool::index_to_state::<2>(input);
        let bond = if state[0] == state[1] {
            -scale.clone()
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

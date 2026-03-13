use crate::qmc::MatrixTermData;
use crate::qmc::naive_flip_impl::MatrixTermFlippable;
use num_traits::{Signed, Zero};
use std::fmt::Debug;
use std::ops::{Add, Neg};
use crate::qmc::resummed_flip_impl::{MatrixTermFlippableOrDiagonal, MatrixTermType};

/// The collection of allowed terms in the Hamiltonian, Z, ZZ, ZZZ, ..., or X.
#[derive(Debug, Clone, Copy)]
pub enum MultibodyTFIMTerm<T> {
    /// A Z^{otimes n} term.
    ZN(usize, T),
    /// A transverse field applied to a single site.
    X(T),
}

impl<T> MatrixTermData<T> for MultibodyTFIMTerm<T>
where
    T: Clone + Zero + Signed + Add + Debug,
{
    fn get_matrix_entry(&self, input: usize, output: usize) -> T {
        match self {
            MultibodyTFIMTerm::ZN(n, jj) => {
                let weight = get_ising_weight(input, output, *n, jj);
                weight.unwrap_or(T::zero())
            }
            MultibodyTFIMTerm::X(gamma) => gamma.clone(),
        }
    }

    fn dim(&self) -> usize {
        match self {
            MultibodyTFIMTerm::ZN(n, _) => 1 << n,
            MultibodyTFIMTerm::X(_) => 2,
        }
    }

    fn get_number_of_equal_weight_outputs_for_input_distinct_from_output(
        &self,
        _input: usize,
        _output: usize,
    ) -> usize {
        match self {
            MultibodyTFIMTerm::ZN(_, _) => 0,
            MultibodyTFIMTerm::X(_) => 1,
        }
    }

    fn get_weight_change_for_diagonal(&self, old_state: usize, new_state: usize) -> Option<(T, T)> {
        match self {
            MultibodyTFIMTerm::X(_) => None,
            MultibodyTFIMTerm::ZN(n, jj) => {
                debug_assert_eq!(old_state & ((1 << n) - 1), old_state);
                debug_assert_eq!(new_state & ((1 << n) - 1), new_state);

                let old_ferro = old_state.count_ones().is_multiple_of(2);
                let new_ferro = new_state.count_ones().is_multiple_of(2);
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
            MultibodyTFIMTerm::ZN(n, jj) => jj.abs(),
            MultibodyTFIMTerm::X(gamma) => gamma.clone(),
        }
    }
}

impl<T> MatrixTermFlippable<T> for MultibodyTFIMTerm<T>
where
    T: Zero + Clone + Signed,
{
    fn is_maybe_flippable(&self) -> bool {
        matches!(self, MultibodyTFIMTerm::X(_))
    }

    fn get_weights_for_inputs_given_output(
        &self,
        input_a: usize,
        input_b: usize,
        output: usize,
    ) -> Option<(T, T)> {
        match self {
            MultibodyTFIMTerm::ZN(n, jj) => {
                if input_a == input_b {
                    None
                } else {
                    let weight_a = if input_a == output {
                        get_ising_weight(input_a, input_a, *n, jj)
                    } else {
                        None
                    };
                    let weight_b = if input_b == output {
                        get_ising_weight(input_b, input_b, *n, jj)
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
            MultibodyTFIMTerm::X(_) => None,
        }
    }

    fn get_nth_equal_weight_output_for_input_distinct_from_output(
        &self,
        _input: usize,
        output: usize,
        n: usize,
    ) -> usize {
        match self {
            MultibodyTFIMTerm::ZN(_, _) => {
                unimplemented!("There are no equal weight outputs for a given input.")
            }
            MultibodyTFIMTerm::X(_) => {
                debug_assert_eq!(n, 0);
                1 - output
            }
        }
    }
}

fn get_ising_weight<T>(input: usize, output: usize, n: usize, scale: &T) -> Option<T>
where
    T: Clone + Neg<Output = T> + Signed + Add,
{
    if input != output {
        None
    } else {
        debug_assert_eq!(input & ((1 << n) - 1), input);
        let aligned = input.count_ones().is_multiple_of(2);
        let bond = if aligned {
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

impl<T> MatrixTermFlippableOrDiagonal for MultibodyTFIMTerm<T> {
    fn get_term_type(&self) -> MatrixTermType {
        match self {
            MultibodyTFIMTerm::ZN(_, _) => MatrixTermType::MultiSiteDiagonal,
            MultibodyTFIMTerm::X(_) => MatrixTermType::SingleSiteFlippable
        }
    }
}

#[cfg(test)]
mod multibody_tests {
    use crate::qmc::GenericQMC;
    use crate::traits::graph_traits::GraphStateNavigator;
    use crate::traits::naive_flip_update::NaiveFlipUpdater;
    use super::*;

    #[test]
    fn test_zzz_flip() -> Result<(), String> {
        let n = 3;
        let mut qmc = GenericQMC::<bool, _>::new(n);
        let zzz = qmc.add_term(MultibodyTFIMTerm::ZN(3, -1.0), [0,1,2]);
        let xa = qmc.add_term(MultibodyTFIMTerm::X(1.0), [0]);
        let xb = qmc.add_term(MultibodyTFIMTerm::X(1.0), [1]);
        let xc = qmc.add_term(MultibodyTFIMTerm::X(1.0), [2]);
        qmc.add_node(0, xa);
        qmc.add_node(1, xb);
        qmc.add_node(2, xc);
        qmc.add_node(3, zzz);
        qmc.add_node(4, xa);
        qmc.add_node(5, xb);
        qmc.add_node(6, xc);

        let mut rng = rand::rng();
        qmc.print_worldlines();
        println!("{:?}", qmc.get_initial_state());
        qmc.naive_flip_update_starting_from_timeslice(0, &mut rng);
        qmc.print_worldlines();
        println!("{:?}", qmc.get_initial_state());

        Ok(())
    }

    #[test]
    fn test_double_zzz_cluster() -> Result<(), String> {
        let n = 3;
        let mut qmc = GenericQMC::<bool, _>::new(n);
        let zzz = qmc.add_term(MultibodyTFIMTerm::ZN(3, -1.0), [0,1,2]);
        let xa = qmc.add_term(MultibodyTFIMTerm::X(1.0), [0]);
        let xb = qmc.add_term(MultibodyTFIMTerm::X(1.0), [1]);
        let xc = qmc.add_term(MultibodyTFIMTerm::X(1.0), [2]);
        qmc.add_node(0, xa);
        qmc.add_node(1, xb);
        qmc.add_node(2, xc);
        qmc.add_node(3, zzz);
        qmc.add_node(4, zzz);
        qmc.add_node(5, xa);
        qmc.add_node(6, xb);
        qmc.add_node(7, xc);

        let mut rng = rand::rng();
        qmc.print_worldlines();
        println!("{:?}", qmc.get_initial_state());
        qmc.naive_flip_update_starting_from_timeslice(0, &mut rng);
        qmc.print_worldlines();
        println!("{:?}", qmc.get_initial_state());

        Ok(())
    }
}
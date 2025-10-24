use crate::qmc::{DoublyLinkedNode, GenericQMC, MatrixTermData};
use crate::traits::diagonal_update::DiagonalUpdate;
use crate::traits::graph_traits::{DOFTypeTrait, GraphContext, Link};

impl<DOF: DOFTypeTrait, Data: MatrixTermData<f64>, GC> DiagonalUpdate
    for GenericQMC<DOF, Data, GC>
{
    fn get_number_of_time_slices(&self) -> usize {
        self.time_slices.len()
    }

    fn get_number_of_non_identity_operators(&self) -> usize {
        self.num_non_identity_terms
    }

    fn diagonal_update_resize_hook(&mut self) {
        self.maintain_maximum_filling_fraction(self.filling_fraction, self.min_space);
    }

    fn construct_node(
        timeslice: &Self::TimesliceIndex,
        context: GraphContext<Self::DOFType, Link<Self::TimesliceIndex>>,
        term: Self::MatrixTerm,
    ) -> Self::Node {
        DoublyLinkedNode {
            input_state: context.local_state.clone(),
            output_state: context.local_state.clone(),
            represents_term: term,
            previous_node_index_for_variable: context.prev_node_slice,
            next_node_index_for_variable: context.next_node_slice,
            timeslice: *timeslice,
            // Insert node will overwrite these values.
            index_of_entry_in_node_list_for_term: usize::MAX,
            index_of_entry_into_flippable_list: None,
        }
    }
}

#[cfg(test)]
mod test_diagonal {
    use super::*;
    use crate::terms::generic::GenericMatrixTermEnum;

    #[test]
    fn test_run_diagonal() {
        let mut qmc = GenericQMC::<bool, _>::new(3);
        let _term_handle = qmc.add_term(
            GenericMatrixTermEnum::make_diagonal(vec![1.0, 1.0]),
            vec![0],
        );
        qmc.set_minimum_timeslices(50);

        let mut rng = rand::rng();

        let beta = 10.0;
        let ntrials = 128;
        let mut count = 0;
        for _ in 0..ntrials {
            for _ in 0..16 {
                qmc.diagonal_update(beta, &mut rng);
            }
            count += qmc.num_non_identity_terms;
        }
        let avg_energy = (count as f64 / ntrials as f64) / beta;

        println!("Avg energy: {}", avg_energy);
        assert!((avg_energy - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_run_diagonal_scaled() {
        let mut qmc = GenericQMC::<bool, _>::new(3);
        let _term_handle = qmc.add_term(
            GenericMatrixTermEnum::make_diagonal(vec![2.0, 2.0]),
            vec![0],
        );
        qmc.set_minimum_timeslices(50);

        let mut rng = rand::rng();

        let beta = 10.0;
        let ntrials = 128;
        let mut count = 0;
        for _ in 0..ntrials {
            for _ in 0..16 {
                qmc.diagonal_update(beta, &mut rng);
            }
            count += qmc.num_non_identity_terms;
        }
        let avg_energy = (count as f64 / ntrials as f64) / beta;
        println!("Avg energy: {}", avg_energy);
        assert!((avg_energy - 2.0).abs() < 0.2);
    }

    #[test]
    fn test_run_diagonal_state_dependent() {
        let mut qmc = GenericQMC::new_with_state(vec![false, false, false]);
        let _term_handle = qmc.add_term(
            GenericMatrixTermEnum::make_diagonal(vec![1.0, 10.0]),
            vec![0],
        );
        qmc.set_minimum_timeslices(50);

        let mut rng = rand::rng();

        let beta = 10.0;
        let ntrials = 128;
        let mut count = 0;
        for _ in 0..ntrials {
            for _ in 0..16 {
                qmc.diagonal_update(beta, &mut rng);
            }
            count += qmc.num_non_identity_terms;
        }
        let avg_energy = (count as f64 / ntrials as f64) / beta;
        println!("Avg energy: {}", avg_energy);
        assert!((avg_energy - 1.0).abs() < 0.1);

        let mut qmc = GenericQMC::new_with_state(vec![true, true, true]);
        let _term_handle = qmc.add_term(
            GenericMatrixTermEnum::make_diagonal(vec![1.0, 2.0]),
            vec![0],
        );
        qmc.set_minimum_timeslices(50);

        let mut rng = rand::rng();

        let beta = 10.0;
        let ntrials = 128;
        let mut count = 0;
        for _ in 0..ntrials {
            for _ in 0..16 {
                qmc.diagonal_update(beta, &mut rng);
            }
            count += qmc.num_non_identity_terms;
        }
        let avg_energy = (count as f64 / ntrials as f64) / beta;
        println!("Avg energy: {}", avg_energy);
        assert!((avg_energy - 2.0).abs() < 0.2);
    }
}

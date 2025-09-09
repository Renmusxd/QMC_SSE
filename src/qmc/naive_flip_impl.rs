use crate::qmc::{GenericMatrixTermEnum, GenericQMC, MatrixTermData};
use crate::traits::graph_traits::{DOFTypeTrait, GraphNode};
use crate::traits::naive_flip_update::NaiveFlipUpdater;
use num_traits::{One, Zero};

impl<DOF: DOFTypeTrait, Data: MatrixTermData<f64>> NaiveFlipUpdater for GenericQMC<DOF, Data>
where
    Data: MatrixTermFlippable<f64>,
{
    fn num_potential_flip_boundaries(&self) -> usize {
        self.list_of_nodes_with_flippable_outputs.len()
    }

    fn get_potential_flip_boundary(&self, n: usize) -> Self::TimesliceIndex {
        self.list_of_nodes_with_flippable_outputs[n]
    }

    fn is_node_potentially_flippable(&self, node: &Self::Node) -> bool {
        self.all_term_data[node.represents_term.matrix_data_entry].is_maybe_flippable()
    }

    fn get_number_of_equal_weight_alternative_outputs(&self, node: &Self::Node) -> usize {
        let term_data = &self.all_term_data[node.represents_term.matrix_data_entry];
        let input = Self::DOFType::index_dimension_slice(&node.input_state);
        let output = Self::DOFType::index_dimension_slice(&node.output_state);
        term_data.get_number_of_equal_weight_outputs_for_input_distinct_from_output(input, output)
    }

    fn get_nth_equal_weight_alternative_output(
        &self,
        node: &Self::Node,
        n: usize,
    ) -> Vec<Self::DOFType> {
        let term_data = &self.all_term_data[node.represents_term.matrix_data_entry];
        let input = Self::DOFType::index_dimension_slice(&node.input_state);
        let output = Self::DOFType::index_dimension_slice(&node.output_state);
        let new_output =
            term_data.get_nth_equal_weight_output_for_input_distinct_from_output(input, output, n);
        Self::DOFType::index_to_state_vec(new_output, node.get_indices().len())
    }

    fn can_node_absorb_flip_from_top(
        &self,
        node: &Self::Node,
        new_input_state: &[Self::DOFType],
        _acts_on_dof: &[Self::DOFIndex],
        originating_matrix_term: &Self::MatrixTerm,
    ) -> bool {
        if node.represents_term.matrix_data_entry != originating_matrix_term.matrix_data_entry {
            // For efficiency, only consider like terms.
            false
        } else {
            let old_input_state = Self::DOFType::index_dimension_slice(&node.input_state);
            let new_input_state = Self::DOFType::index_dimension_slice(new_input_state);
            let output = Self::DOFType::index_dimension_slice(&node.output_state);
            let term_data = &self.all_term_data[node.represents_term.matrix_data_entry];
            term_data
                .get_weights_for_inputs_given_output(old_input_state, new_input_state, output)
                .is_none()
        }
    }

    fn get_relative_weight_change_for_new_state(
        &self,
        node: &Self::Node,
        new_state: &[Self::DOFType],
    ) -> Option<f64> {
        debug_assert!(node.is_diagonal());
        let term_data = &self.all_term_data[node.represents_term.matrix_data_entry];
        let old_state = Self::DOFType::index_dimension_slice(&node.input_state);
        let new_state = Self::DOFType::index_dimension_slice(new_state);
        term_data
            .get_weight_change_for_diagonal(old_state, new_state)
            .map(|(a, b)| b / a)
    }

    fn modify_node_at_timeslice_input_and_output<F>(
        &mut self,
        timeslice: &Self::TimesliceIndex,
        f: F,
    ) -> Option<&Self::Node>
    where
        F: Fn(&mut [Self::DOFType], &mut [Self::DOFType]),
    {
        if let Some(node) = self.time_slices[*timeslice].as_mut() {
            let inputs = &mut node.input_state;
            let outputs = &mut node.output_state;
            f(inputs, outputs);

            // Add to flippable list if now flippable.
            let matrix_data = &self.all_term_data[node.represents_term.matrix_data_entry];
            let input = DOF::index_dimension_slice(&node.input_state);
            let output = DOF::index_dimension_slice(&node.output_state);
            let n_flippable_outputs = matrix_data
                .get_number_of_equal_weight_outputs_for_input_distinct_from_output(input, output);
            if n_flippable_outputs > 0 {
                if node.index_of_entry_into_flippable_list.is_none() {
                    let index_to_insert = self.list_of_nodes_with_flippable_outputs.len();
                    self.list_of_nodes_with_flippable_outputs.push(*timeslice);
                    node.index_of_entry_into_flippable_list = Some(index_to_insert);
                }
            } else if let Some(index_to_remove) =
                node.index_of_entry_into_flippable_list.as_ref().copied()
            {
                node.index_of_entry_into_flippable_list = None;
                Self::handle_flippable_removal(
                    index_to_remove,
                    &mut self.list_of_nodes_with_flippable_outputs,
                    &mut self.time_slices,
                );
            }
        };

        self.time_slices[*timeslice].as_ref()
    }
}

pub trait MatrixTermFlippable<T> {
    /// Returns true if this term is allowed to be flipped offdiagonal by a change to either the
    /// input or the output. If both are required to change then the term is more suitable to
    /// cluster-like updates.
    fn is_maybe_flippable(&self) -> bool;

    /// For a fixed output, does changing the inputs result in a change to the matrix entry.
    /// Returns None if there's no change, or else outputs the first and second input weights.
    fn get_weights_for_inputs_given_output(
        &self,
        input_a: usize,
        input_b: usize,
        output: usize,
    ) -> Option<(T, T)>;
    fn get_nth_equal_weight_output_for_input_distinct_from_output(
        &self,
        input: usize,
        output: usize,
        n: usize,
    ) -> usize;
}

#[cfg(test)]
mod test_naive_flip_implementation {
    use super::*;
    use crate::qmc::GenericMatrixTermEnum;
    use crate::traits::graph_traits::{GraphStateNavigator, TimeSlicedGraph};

    #[test]
    fn test_simple_naive_flip() {
        let mut qmc = GenericQMC::<bool>::new(1);
        let handle = qmc.add_term(GenericMatrixTermEnum::make_uniform(1.0, 2), vec![0]);

        qmc.add_node(0, handle);
        qmc.add_node(1, handle);

        assert_eq!(qmc.num_potential_flip_boundaries(), 2);
        let node = qmc
            .get_node(&0)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());
        let node = qmc
            .get_node(&1)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());

        let mut rng = rand::rng();
        qmc.naive_flip_update_starting_from_timeslice(0, &mut rng);

        let node = qmc
            .get_node(&0)
            .expect("Node was added, should not be removed.");
        assert!(!node.is_diagonal());
        let node = qmc
            .get_node(&1)
            .expect("Node was added, should not be removed.");
        assert!(!node.is_diagonal());
    }

    #[test]
    fn test_simple_naive_flip_single_operator() {
        let mut qmc = GenericQMC::<bool>::new(1);
        qmc.set_minimum_timeslices(16);
        let handle = qmc.add_term(GenericMatrixTermEnum::make_uniform(1.0, 2), vec![0]);

        qmc.add_node(0, handle);

        assert_eq!(qmc.num_potential_flip_boundaries(), 1);
        let node = qmc
            .get_node(&0)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());
        assert_eq!(node.input_state[0], false);
        assert_eq!(node.output_state[0], false);

        let mut rng = rand::rng();
        qmc.naive_flip_update_starting_from_timeslice(0, &mut rng);

        let node = qmc
            .get_node(&0)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());
        assert_eq!(node.input_state[0], true);
        assert_eq!(node.output_state[0], true);
    }

    #[test]
    fn test_naive_flip_pass_through() {
        let mut qmc = GenericQMC::<bool>::new(1);
        let handle = qmc.add_term(GenericMatrixTermEnum::make_uniform(1.0, 2), vec![0]);
        let handle_ident = qmc.add_term(GenericMatrixTermEnum::make_identity(2), vec![0]);

        qmc.add_node(0, handle);
        qmc.add_node(1, handle_ident);
        qmc.add_node(2, handle);

        assert_eq!(qmc.num_potential_flip_boundaries(), 2);
        let node = qmc
            .get_node(&0)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());
        let node = qmc
            .get_node(&1)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());
        let node = qmc
            .get_node(&2)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());

        let mut rng = rand::rng();
        qmc.naive_flip_update_starting_from_timeslice(0, &mut rng);

        let node = qmc
            .get_node(&0)
            .expect("Node was added, should not be removed.");
        assert!(!node.is_diagonal());
        let node = qmc
            .get_node(&1)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());
        let node = qmc
            .get_node(&2)
            .expect("Node was added, should not be removed.");
        assert!(!node.is_diagonal());
    }

    #[test]
    fn test_simple_flip_wraparound() {
        let mut qmc = GenericQMC::new_with_state(vec![false]);
        let handle = qmc.add_term(GenericMatrixTermEnum::make_uniform(1.0, 2), vec![0]);

        qmc.add_node(0, handle);
        qmc.add_node(1, handle);

        assert_eq!(qmc.num_potential_flip_boundaries(), 2);
        let node = qmc
            .get_node(&0)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());
        let node = qmc
            .get_node(&1)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());
        assert!(!qmc.get_initial_state()[0]);

        let mut rng = rand::rng();
        qmc.naive_flip_update_starting_from_timeslice(1, &mut rng);

        let node = qmc
            .get_node(&0)
            .expect("Node was added, should not be removed.");
        assert!(!node.is_diagonal());
        let node = qmc
            .get_node(&1)
            .expect("Node was added, should not be removed.");
        assert!(!node.is_diagonal());
        assert!(qmc.get_initial_state()[0]);
    }

    #[test]
    fn test_flip_through_two_dof() {
        let mut qmc = GenericQMC::<bool>::new(2);
        let handle = qmc.add_term(GenericMatrixTermEnum::make_uniform(1.0, 2), vec![0]);
        let handle_ident = qmc.add_term(GenericMatrixTermEnum::make_identity(4), vec![0, 1]);

        qmc.add_node(0, handle);
        qmc.add_node(1, handle_ident);
        qmc.add_node(2, handle);

        assert_eq!(qmc.num_potential_flip_boundaries(), 2);
        let node = qmc
            .get_node(&0)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());
        let node = qmc
            .get_node(&1)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());
        let node = qmc
            .get_node(&2)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());

        let mut rng = rand::rng();
        qmc.naive_flip_update_starting_from_timeslice(0, &mut rng);

        let node = qmc
            .get_node(&0)
            .expect("Node was added, should not be removed.");
        assert!(!node.is_diagonal());
        let node = qmc
            .get_node(&1)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());
        let node = qmc
            .get_node(&2)
            .expect("Node was added, should not be removed.");
        assert!(!node.is_diagonal());
    }

    #[test]
    fn test_flip_two_dof() {
        let mut qmc = GenericQMC::<bool>::new(2);
        let handle = qmc.add_term(GenericMatrixTermEnum::make_uniform(1.0, 4), vec![0, 1]);

        qmc.add_node(0, handle);
        qmc.add_node(1, handle);

        assert_eq!(qmc.num_potential_flip_boundaries(), 2);
        let node = qmc
            .get_node(&0)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());
        let node = qmc
            .get_node(&1)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());

        let mut rng = rand::rng();
        qmc.naive_flip_update_starting_from_timeslice(0, &mut rng);

        let node = qmc
            .get_node(&0)
            .expect("Node was added, should not be removed.");
        assert!(!node.is_diagonal());
        let node = qmc
            .get_node(&1)
            .expect("Node was added, should not be removed.");
        assert!(!node.is_diagonal());
    }

    #[test]
    fn test_flip_two_dof_with_passthrough() {
        let mut qmc = GenericQMC::<bool>::new(2);
        let flip_term = GenericMatrixTermEnum::make_uniform(1.0, 4);

        assert_eq!(
            flip_term.get_number_of_equal_weight_outputs_for_input_distinct_from_output(0, 0),
            3
        );

        let handle = qmc.add_term(flip_term, vec![0, 1]);
        let handle_ident = qmc.add_term(GenericMatrixTermEnum::make_identity(2), vec![0]);

        qmc.add_node(0, handle);
        qmc.add_node(1, handle_ident);
        qmc.add_node(2, handle);

        assert_eq!(qmc.num_potential_flip_boundaries(), 2);
        let node = qmc
            .get_node(&0)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());
        let node = qmc
            .get_node(&1)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());
        let node = qmc
            .get_node(&2)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());

        let mut rng = rand::rng();
        qmc.naive_flip_update_starting_from_timeslice(0, &mut rng);

        let node = qmc
            .get_node(&0)
            .expect("Node was added, should not be removed.");
        assert!(!node.is_diagonal());
        let node = qmc
            .get_node(&1)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());
        let node = qmc
            .get_node(&2)
            .expect("Node was added, should not be removed.");
        assert!(!node.is_diagonal());
    }

    #[test]
    fn test_sparse_uniform_flip() {
        let mut qmc = GenericQMC::<bool>::new(2);

        let flip_term = GenericMatrixTermEnum::make_sparse_uniform(
            1.0,
            4,
            vec![(0, 0), (1, 1), (2, 2), (3, 3), (0, 1), (1, 0)],
        );

        assert_eq!(
            flip_term.get_number_of_equal_weight_outputs_for_input_distinct_from_output(0, 0),
            1
        );

        let handle = qmc.add_term(flip_term, vec![0, 1]);
        let handle_ident = qmc.add_term(GenericMatrixTermEnum::make_identity(2), vec![0]);

        qmc.add_node(0, handle);
        qmc.add_node(1, handle_ident);
        qmc.add_node(2, handle);

        assert_eq!(qmc.num_potential_flip_boundaries(), 2);
        let node = qmc
            .get_node(&0)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());
        let node = qmc
            .get_node(&1)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());
        let node = qmc
            .get_node(&2)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());

        let mut rng = rand::rng();
        qmc.naive_flip_update_starting_from_timeslice(0, &mut rng);

        let node = qmc
            .get_node(&0)
            .expect("Node was added, should not be removed.");
        assert!(!node.is_diagonal());
        assert_eq!(node.output_state, vec![true, false]);
        let node = qmc
            .get_node(&1)
            .expect("Node was added, should not be removed.");
        assert!(node.is_diagonal());
        let node = qmc
            .get_node(&2)
            .expect("Node was added, should not be removed.");
        assert!(!node.is_diagonal());
        assert_eq!(node.input_state, vec![true, false]);
    }

    #[test]
    fn test_flip_ends() {
        let mut qmc = GenericQMC::<bool>::new(2);
        let flip_term = GenericMatrixTermEnum::make_uniform(1.0, 2);

        assert_eq!(
            flip_term.get_number_of_equal_weight_outputs_for_input_distinct_from_output(0, 0),
            1
        );

        let handle = qmc.add_term(flip_term, vec![0]);
        let handle_ident = qmc.add_term(GenericMatrixTermEnum::make_identity(2), vec![0]);

        qmc.add_node(0, handle);
        qmc.add_node(1, handle_ident);

        let mut rng = rand::rng();
        qmc.naive_flip_update_starting_from_timeslice(0, &mut rng);
    }
}

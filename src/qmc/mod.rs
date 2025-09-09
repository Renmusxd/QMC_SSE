use crate::qmc::naive_flip_impl::MatrixTermFlippable;
use crate::traits::graph_traits::{DOFTypeTrait, GraphNode, Link, TimeSlicedGraph};
use num_traits::{One, Zero};
use std::cmp::max;

#[cfg(feature = "autocorrelations")]
pub mod autocorr;
pub mod cluster_impl;
pub mod diagonal_impl;
pub mod graph_mod_impl;
pub mod naive_flip_impl;
pub mod navigator_impl;
pub mod weight_impl;

pub type MatrixTermHandle = usize;

pub struct GenericQMC<
    DOF: DOFTypeTrait = bool,
    TermData: MatrixTermData<f64> = GenericMatrixTermEnum<f64>,
> {
    initial_state: Vec<DOF>,
    indices: Vec<usize>,
    time_slices: Vec<Option<DoublyLinkedNode<DOF>>>,
    first_nodes_for_dofs: Vec<Option<Link<usize>>>,
    last_nodes_for_dofs: Vec<Option<Link<usize>>>,

    // List of all terms in Hamiltonian
    all_terms: Vec<GenericMatrixTerm>,
    all_term_data: Vec<TermData>,

    // Count of terms
    num_non_identity_terms: usize,

    // Keep track of nodes and which are maybe flippable.
    list_of_nodes_by_term: Vec<Vec<usize>>,

    list_of_nodes_with_flippable_outputs: Vec<usize>,
}

impl<DOF: DOFTypeTrait, TermData: MatrixTermData<f64>> GenericQMC<DOF, TermData> {
    pub fn new(num_dofs: usize) -> Self {
        let state = (0..num_dofs).map(|_| DOF::default()).collect();
        Self::new_with_state(state)
    }

    pub fn new_with_state(initial_state: Vec<DOF>) -> Self {
        let n = initial_state.len();
        Self {
            initial_state,
            indices: (0..n).collect(),
            time_slices: vec![],
            first_nodes_for_dofs: vec![None; n],
            last_nodes_for_dofs: vec![None; n],
            all_terms: vec![],
            all_term_data: vec![],
            num_non_identity_terms: 0,
            list_of_nodes_by_term: vec![],
            list_of_nodes_with_flippable_outputs: vec![],
        }
    }

    pub fn get_energy(&self, beta: f64) -> f64 {
        self.num_non_identity_terms as f64 / beta
    }

    pub fn get_expectation_value_of_term(&self, beta: f64, term: &MatrixTermHandle) -> f64 {
        let n = self.get_count_for_term(term);
        n as f64 / beta
    }

    pub fn get_each_expectation_value(&self, beta: f64) -> Vec<f64> {
        (0..self.all_terms.len())
            .map(|i| self.get_expectation_value_of_term(beta, &i))
            .collect()
    }

    pub fn set_minimum_timeslices(&mut self, m: usize) {
        if self.num_time_slices() < m {
            self.time_slices.resize_with(m, || None);
        }
    }

    pub fn maintain_maximum_filling_fraction(&mut self, frac: f64, min_val: usize) {
        let size_to_meet_quota = ((self.num_non_identity_terms as f64) / frac).ceil() as usize;
        self.set_minimum_timeslices(max(size_to_meet_quota, min_val));
        debug_assert!(self.check_consistency());
    }

    pub fn add_term(&mut self, data: TermData, act_on_indices: Vec<usize>) -> MatrixTermHandle {
        debug_assert_eq!(
            data.dim(),
            DOF::local_dimension().pow(act_on_indices.len() as u32)
        );
        debug_assert_eq!(
            {
                let mut indices_clone = act_on_indices.clone();
                indices_clone.dedup();
                indices_clone.len()
            },
            act_on_indices.len(),
            "Indices cannot contain duplicates."
        );

        let matrix_data_entry = self.all_term_data.len();
        self.all_term_data.push(data);
        self.all_terms.push(GenericMatrixTerm {
            act_on_indices,
            matrix_data_entry,
        });

        // Add an entry for lists of nodes
        self.list_of_nodes_by_term.push(vec![]);

        matrix_data_entry
    }

    fn add_node(&mut self, timeslice: usize, matrix_term_handle: MatrixTermHandle) {
        let act_on_indices = self.all_terms[matrix_term_handle].act_on_indices.clone();
        self.insert_node(&timeslice, &act_on_indices, |context| DoublyLinkedNode {
            input_state: context.local_state.clone(),
            output_state: context.local_state.clone(),
            represents_term: GenericMatrixTerm {
                act_on_indices: act_on_indices.clone(),
                matrix_data_entry: matrix_term_handle,
            },
            previous_node_index_for_variable: context.prev_node_slice,
            next_node_index_for_variable: context.next_node_slice,
            timeslice,
            // Handled by .insert_node call.
            index_of_entry_in_node_list_for_term: usize::MAX,
            index_of_entry_into_flippable_list: None,
        });
    }
    fn remove_from_flippable_list(&mut self, node: &DoublyLinkedNode<DOF>) {
        if let Some(index_to_remove) = node.index_of_entry_into_flippable_list {
            Self::handle_flippable_removal(
                index_to_remove,
                &mut self.list_of_nodes_with_flippable_outputs,
                &mut self.time_slices,
            );
        }
    }

    fn handle_flippable_removal(
        index_to_remove: usize,
        flippable_list: &mut Vec<usize>,
        time_slices: &mut [Option<DoublyLinkedNode<DOF>>],
    ) {
        // Fix entry in list of flippables
        let last_index = flippable_list.len() - 1;
        // Check if last entry.
        if index_to_remove == last_index {
            // Easy! pop it.
            flippable_list.pop();
        } else {
            // Swap with last, then pop.
            flippable_list.swap(index_to_remove, last_index);
            let node_to_fix = time_slices[flippable_list[index_to_remove]]
                .as_mut()
                .expect("List entry should never point to None.");
            debug_assert_eq!(
                node_to_fix.index_of_entry_into_flippable_list,
                Some(last_index)
            );
            node_to_fix.index_of_entry_into_flippable_list = Some(index_to_remove);
            flippable_list.pop();
        }
    }

    pub fn get_count_for_term(&self, term: &MatrixTermHandle) -> usize {
        self.list_of_nodes_by_term[*term].len()
    }

    pub fn print_worldlines(&self) {
        let mut worldline = vec!["|"; self.initial_state.len()];
        for (t, slice) in self.time_slices.iter().enumerate() {
            if let Some(node) = slice {
                for i in &node.represents_term.act_on_indices {
                    if node.is_diagonal() {
                        worldline[*i] = "O";
                    } else {
                        worldline[*i] = "X";
                    }
                }
                println!("{}:\t{}", t, worldline.join(""));
                for i in &node.represents_term.act_on_indices {
                    worldline[*i] = "|";
                }
            } else {
                println!("{}:\t{}", t, worldline.join(""))
            }
        }
    }

    pub fn check_node_consistency(&self, timeslice: usize, node: &DoublyLinkedNode<DOF>) -> bool {
        // First check the links for each dof.
        for (rel_index, global_index) in node
            .represents_term
            .act_on_indices
            .iter()
            .copied()
            .enumerate()
        {
            let previous_node = node.previous_node_index_for_variable[rel_index].as_ref();
            let link_to_this_node = match previous_node {
                None => self.first_nodes_for_dofs[global_index]
                    .as_ref()
                    .expect("Head must point to this node."),
                Some(link_to_previous_node) => {
                    assert!(
                        link_to_previous_node.timeslice < timeslice,
                        "Previous timeslice must be lower order."
                    );
                    let previous_node = self.time_slices[link_to_previous_node.timeslice]
                        .as_ref()
                        .expect("Link must point to existing node.");
                    assert_eq!(
                        previous_node.represents_term.act_on_indices
                            [link_to_previous_node.relative_index],
                        global_index,
                        "Link must point to correct dof."
                    );
                    previous_node.next_node_index_for_variable[link_to_previous_node.relative_index]
                        .as_ref()
                        .expect("Link to next node must exist.")
                }
            };
            assert_eq!(
                link_to_this_node.timeslice, timeslice,
                "Link must point to this timeslice."
            );
            assert_eq!(
                link_to_this_node.relative_index, rel_index,
                "Link must point to this relative index."
            );

            let next_node = node.next_node_index_for_variable[rel_index].as_ref();
            let link_to_this_node = match next_node {
                None => self.last_nodes_for_dofs[global_index]
                    .as_ref()
                    .expect("Tail must point to this node."),
                Some(link_to_next_node) => {
                    assert!(
                        link_to_next_node.timeslice > timeslice,
                        "Previous timeslice must be higher order."
                    );
                    let next_node = self.time_slices[link_to_next_node.timeslice]
                        .as_ref()
                        .expect("Link must point to existing node.");
                    assert_eq!(
                        next_node.represents_term.act_on_indices[link_to_next_node.relative_index],
                        global_index,
                        "Link must point to correct dof."
                    );
                    next_node.previous_node_index_for_variable[link_to_next_node.relative_index]
                        .as_ref()
                        .expect("Link to previous node must exist.")
                }
            };
            assert_eq!(
                link_to_this_node.timeslice, timeslice,
                "Link must point to this timeslice."
            );
            assert_eq!(
                link_to_this_node.relative_index, rel_index,
                "Link must point to this relative index."
            );
        }

        // Now check the bookkeeping
        let list_node_should_be_in =
            &self.list_of_nodes_by_term[node.represents_term.matrix_data_entry];
        assert_eq!(
            list_node_should_be_in[node.index_of_entry_in_node_list_for_term],
            timeslice
        );

        let matrix_data = &self.all_term_data[node.represents_term.matrix_data_entry];
        let input = DOF::index_dimension_slice(&node.input_state);
        let output = DOF::index_dimension_slice(&node.output_state);
        let n_flippable_outputs = matrix_data
            .get_number_of_equal_weight_outputs_for_input_distinct_from_output(input, output);
        assert_eq!(
            n_flippable_outputs > 0,
            node.index_of_entry_into_flippable_list.is_some()
        );
        if let Some(entry_in_list_of_flippable) =
            node.index_of_entry_into_flippable_list.as_ref().copied()
        {
            assert_eq!(
                self.list_of_nodes_with_flippable_outputs[entry_in_list_of_flippable],
                timeslice
            );
        }

        true
    }

    pub fn check_consistency(&self) -> bool {
        // Check count of nodes.
        let mut num_nodes = 0;
        for (t, slice) in self.time_slices.iter().enumerate() {
            if let Some(node) = slice {
                self.check_node_consistency(t, node);
                num_nodes += 1;
            }
        }
        assert_eq!(num_nodes, self.num_non_identity_terms);

        // Check count of nodes from node list.
        let num_nodes = self
            .list_of_nodes_by_term
            .iter()
            .map(|x| x.len())
            .sum::<usize>();
        assert_eq!(num_nodes, self.num_non_identity_terms);

        // Now lets check that all the inputs/outputs align.
        let mut state = self.initial_state.clone();
        for node in self.time_slices.iter().filter_map(|slice| slice.as_ref()) {
            for (rel, global) in node.represents_term.act_on_indices.iter().enumerate() {
                assert_eq!(state[*global], node.input_state[rel]);
                state[*global] = node.output_state[rel].clone();
            }
        }
        assert_eq!(state, self.initial_state);

        // Check that the bookkeeping list checks out
        for (i, t) in self.list_of_nodes_with_flippable_outputs.iter().enumerate() {
            let node = self.time_slices[*t]
                .as_ref()
                .expect("List of nodes must point to non-None");
            assert_eq!(node.index_of_entry_into_flippable_list, Some(i));
        }

        true
    }
}

pub struct DoublyLinkedNode<DOF: DOFTypeTrait> {
    input_state: Vec<DOF>,
    output_state: Vec<DOF>,
    represents_term: GenericMatrixTerm,
    previous_node_index_for_variable: Vec<Option<Link<usize>>>,
    next_node_index_for_variable: Vec<Option<Link<usize>>>,
    timeslice: usize,

    // Keep track of where this is being tracked.
    index_of_entry_in_node_list_for_term: usize,
    index_of_entry_into_flippable_list: Option<usize>,
}

pub trait MatrixTermData<T> {
    fn get_matrix_entry(&self, input: usize, output: usize) -> T;
    fn dim(&self) -> usize;
    fn get_number_of_equal_weight_outputs_for_input_distinct_from_output(
        &self,
        input: usize,
        output: usize,
    ) -> usize;

    /// None means no change, Some((old, new)) implies there may be a change.
    /// Can be overridden for improved performance.
    fn get_weight_change_for_diagonal(&self, old_state: usize, new_state: usize) -> Option<(T, T)> {
        let old_weight = self.get_matrix_entry(old_state, old_state);
        let new_weight = self.get_matrix_entry(new_state, new_state);
        Some((old_weight, new_weight))
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GenericMatrixTerm {
    act_on_indices: Vec<usize>,
    matrix_data_entry: MatrixTermHandle,
}

pub enum GenericMatrixTermEnum<T> {
    Identity {
        dim: usize,
    },
    Diagonal {
        data: Vec<T>,
    },
    Uniform {
        data: T,
        dim: usize,
    },
    UniformSparse {
        data: T,
        dim: usize,
        outputs_for_input: Vec<Vec<usize>>,
        inputs_for_output: Vec<Vec<usize>>,
    },
    Generic {
        data: Vec<T>,
        dim: usize,
    },
}

impl<T> GenericMatrixTermEnum<T>
where
    T: One + Zero + Clone,
{
    pub fn make_diagonal(data: Vec<T>) -> Self {
        Self::Diagonal { data }
    }

    pub fn make_identity(dim: usize) -> Self {
        Self::Identity { dim }
    }
    pub fn make_uniform(data: T, dim: usize) -> Self {
        Self::Uniform { data, dim }
    }

    pub fn make_sparse_uniform(data: T, dim: usize, matrix_entries: Vec<(usize, usize)>) -> Self {
        let mut inputs = matrix_entries
            .iter()
            .copied()
            .map(|(a, _)| a)
            .collect::<Vec<_>>();
        inputs.dedup();
        inputs.sort();

        let mut outputs_for_input = Vec::new();
        for input in inputs {
            let mut o_for_i = matrix_entries
                .iter()
                .copied()
                .filter(|(a, _)| input.eq(a))
                .map(|(_, b)| b)
                .collect::<Vec<_>>();
            o_for_i.sort();
            outputs_for_input.push(o_for_i);
        }

        let mut outputs = matrix_entries
            .iter()
            .copied()
            .map(|(a, _)| a)
            .collect::<Vec<_>>();
        outputs.dedup();
        outputs.sort();

        let mut inputs_for_output = Vec::new();
        for output in outputs {
            let mut i_for_o = matrix_entries
                .iter()
                .copied()
                .filter(|(_, b)| output.eq(b))
                .map(|(a, _)| a)
                .collect::<Vec<_>>();
            i_for_o.sort();
            inputs_for_output.push(i_for_o);
        }

        Self::UniformSparse {
            data,
            dim,
            outputs_for_input,
            inputs_for_output,
        }
    }
}

impl<T> MatrixTermData<T> for GenericMatrixTermEnum<T>
where
    T: Zero + One + Clone,
{
    fn get_matrix_entry(&self, input: usize, output: usize) -> T {
        match &self {
            Self::Identity { .. } if input == output => T::one(),
            Self::Identity { .. } => T::zero(),
            Self::Diagonal { data } if input == output => data[input].clone(),
            Self::Diagonal { .. } => T::zero(),
            Self::Uniform { data, .. } => data.clone(),
            Self::Generic { data, dim } => data[output * dim + input].clone(),
            Self::UniformSparse {
                data,
                outputs_for_input,
                ..
            } => {
                if outputs_for_input[input].binary_search(&output).is_ok() {
                    data.clone()
                } else {
                    T::zero()
                }
            }
        }
    }
    fn dim(&self) -> usize {
        match self {
            Self::Identity { dim, .. } => *dim,
            Self::Diagonal { data } => data.len(),
            Self::Uniform { dim, .. } => *dim,
            Self::UniformSparse { dim, .. } => *dim,
            Self::Generic { dim, .. } => *dim,
        }
    }
    /// None means no change, Some((old, new)) implies there may be a change.
    fn get_weight_change_for_diagonal(&self, old_state: usize, new_state: usize) -> Option<(T, T)> {
        match &self {
            Self::Identity { .. } | Self::Uniform { .. } => None,
            Self::Diagonal { data } => Some((data[old_state].clone(), data[new_state].clone())),
            Self::Generic { data, dim } => {
                let old_value = data[old_state * (dim + 1)].clone();
                let new_value = data[new_state * (dim + 1)].clone();
                Some((old_value, new_value))
            }
            Self::UniformSparse {
                data,
                outputs_for_input,
                ..
            } => {
                let binary_old = outputs_for_input[old_state].binary_search(&old_state);
                let binary_new = outputs_for_input[new_state].binary_search(&new_state);
                match (binary_old, binary_new) {
                    (Err(_), Err(_)) | (Ok(_), Ok(_)) => None,
                    (Err(_), Ok(_)) => Some((T::zero(), data.clone())),
                    (Ok(_), Err(_)) => Some((T::zero(), data.clone())),
                }
            }
        }
    }

    fn get_number_of_equal_weight_outputs_for_input_distinct_from_output(
        &self,
        input: usize,
        _output: usize,
    ) -> usize {
        match self {
            GenericMatrixTermEnum::Uniform { dim, .. } => *dim - 1,
            GenericMatrixTermEnum::UniformSparse {
                outputs_for_input, ..
            } => outputs_for_input[input].len() - 1,
            _ => 0,
        }
    }
}

impl<T> MatrixTermFlippable<T> for GenericMatrixTermEnum<T>
where
    T: One + Zero + Clone,
{
    fn is_maybe_flippable(&self) -> bool {
        matches!(
            self,
            GenericMatrixTermEnum::Uniform { .. } | GenericMatrixTermEnum::UniformSparse { .. }
        )
    }

    /// For a fixed output, does changing the inputs result in a change to the matrix entry.
    fn get_weights_for_inputs_given_output(
        &self,
        input_a: usize,
        input_b: usize,
        output: usize,
    ) -> Option<(T, T)> {
        if input_a == input_b {
            return None;
        }
        match &self {
            Self::Identity { .. } if input_a != output && input_b != output => None,
            Self::Identity { .. } if input_a == output => Some((T::one(), T::zero())),
            Self::Identity { .. } if input_b == output => Some((T::zero(), T::one())),
            Self::Identity { .. } => None, // if input_a == input_b
            Self::Uniform { .. } => None,
            Self::Diagonal { .. } if input_a != output && input_b != output => None,
            Self::Generic { .. } | Self::Diagonal { .. } => {
                let ta = self.get_matrix_entry(input_a, input_b);
                let tb = self.get_matrix_entry(input_a, input_b);
                Some((ta, tb))
            }
            Self::UniformSparse {
                data,
                inputs_for_output,
                ..
            } => {
                let inputs = &inputs_for_output[output];
                let input_a_res = inputs.binary_search(&input_a);
                let input_b_res = inputs.binary_search(&input_b);
                match (input_a_res, input_b_res) {
                    (Err(_), Err(_)) | (Ok(_), Ok(_)) => None,
                    (Err(_), Ok(_)) => Some((T::zero(), data.clone())),
                    (Ok(_), Err(_)) => Some((T::zero(), data.clone())),
                }
            }
        }
    }

    fn get_nth_equal_weight_output_for_input_distinct_from_output(
        &self,
        input: usize,
        output: usize,
        n: usize,
    ) -> usize {
        match self {
            GenericMatrixTermEnum::Uniform { .. } => {
                if n < output {
                    n
                } else {
                    n + 1
                }
            }
            GenericMatrixTermEnum::UniformSparse {
                outputs_for_input, ..
            } => {
                let res = outputs_for_input[input][n];
                if res < output {
                    res
                } else {
                    outputs_for_input[input][n + 1]
                }
            }
            _ => input,
        }
    }
}

use crate::traits::graph_traits::{
    DOFTypeTrait, GraphNode, GraphStateNavigator, Link, TimeSlicedGraph,
};
use crate::traits::graph_weights::GraphWeight;

/// Autocorrelation calculations using the fast fourier transform.
#[cfg(feature = "autocorrelations")]
pub mod autocorr;
/// Generalized cluster algorithms, including the loop update.
pub mod cluster_impl;
/// Standard diagonal update.
pub mod diagonal_impl;
/// Functions for manipulating the worldline graph.
pub mod graph_mod_impl;
/// The simple offdiagonal update which flips DOFs between pairs of operators.
pub mod naive_flip_impl;
/// Graph navigation.
pub mod navigator_impl;
/// Functions of adding weights to the graph nodes.
pub mod weight_impl;
/// Simple updates for empty worldlines at high temperatures.
pub mod thermal_update_impl;

/// Matrix terms are addressed using the MatrixTermHandle type.
pub type MatrixTermHandle = usize;

/// The GenericQMC implementation works with local degrees of freedom `DOF`. Operators on those DOFs are
/// represented by `TermData`, which stores matrix terms and defines the allowed update rules.
/// `GraphInformation` gives extra information about the connectivity of the graph when needed (currently unused).
pub struct GenericQMC<DOF: DOFTypeTrait, TermData: MatrixTermData<f64>, GraphInformation = ()> {
    initial_state: Vec<DOF>,
    indices: Vec<usize>,
    time_slices: Vec<Option<DoublyLinkedNode<DOF>>>,
    first_nodes_for_dofs: Vec<Option<Link<usize>>>,
    last_nodes_for_dofs: Vec<Option<Link<usize>>>,

    // List of all terms in Hamiltonian
    all_terms: Vec<MatrixTerm>,
    all_term_data: Vec<TermData>,

    // Count of terms
    num_non_identity_terms: usize,

    // Keep track of nodes and which are maybe flippable.
    list_of_nodes_by_term: Vec<Vec<usize>>,

    list_of_nodes_with_flippable_outputs: Vec<usize>,

    graph_information: GraphInformation,

    // Filling fraction details
    filling_fraction: f64,
    min_space: usize,

    // Total energy offset from terms.
    total_offset: f64,
}

impl<DOF: DOFTypeTrait, TermData: MatrixTermData<f64>> GenericQMC<DOF, TermData> {
    /// Construct a new QMC instance acting on `num_dof` sites.
    pub fn new(num_dofs: usize) -> Self {
        Self::new_with_context(num_dofs, ())
    }

    /// Construct a new QMC instance with an `initial_state`.
    pub fn new_with_state(initial_state: Vec<DOF>) -> Self {
        Self::new_with_state_and_context(initial_state, ())
    }
}

impl<DOF: DOFTypeTrait, TermData: MatrixTermData<f64>, GI> GenericQMC<DOF, TermData, GI> {
    /// Construct a new QMC instance acting on `num_dof` sites and additional graph information (unused).
    pub fn new_with_context(num_dofs: usize, graph_information: GI) -> Self {
        let state = (0..num_dofs).map(|_| DOF::default()).collect();
        Self::new_with_state_and_context(state, graph_information)
    }

    /// Construct a new QMC instance with an `initial_state` and additional graph information (unused).
    pub fn new_with_state_and_context(initial_state: Vec<DOF>, graph_information: GI) -> Self {
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
            graph_information,
            filling_fraction: 0.75,
            min_space: 16,
            total_offset: 0.0,
        }
    }

    /// Sample the graph energy given inverse temperature `beta`.
    pub fn get_energy(&self, beta: f64) -> f64 {
        self.num_non_identity_terms as f64 / (-beta) + self.total_offset
    }

    /// Sample the expectatation value of the operator represented by `term` for fixed inverse temperature `beta`.
    pub fn get_expectation_value_of_term(&self, beta: f64, term: &MatrixTermHandle) -> f64 {
        let n = self.get_count_for_term(term);
        n as f64 / beta
    }

    /// Sample the expectatation value of each operator for fixed inverse temperature `beta`.
    /// The sum of these values is given by `get_energy(beta)`.
    pub fn get_each_expectation_value(&self, beta: f64) -> Vec<f64> {
        (0..self.all_terms.len())
            .map(|i| self.get_expectation_value_of_term(beta, &i))
            .collect()
    }

    /// Set the minimum number of internal timeslices, this has no effect on the physics and is only for optimization purposes.
    pub fn set_minimum_timeslices(&mut self, m: usize) {
        if self.num_time_slices() < m {
            self.time_slices.resize_with(m, || None);
        }
    }

    /// Resize the number of internal timeslices such that the fraction of non-identity operators is no greater than `frac`. 
    /// Recommended to call between diagonal updates with frac~0.75. 
    /// Will automatically resize to no less than `min_val`.
    pub fn maintain_maximum_filling_fraction(&mut self, frac: f64, min_val: usize) {
        let size_to_meet_quota = ((self.num_non_identity_terms as f64) / frac).ceil() as usize;
        self.set_minimum_timeslices(size_to_meet_quota.max(min_val));
        debug_assert!(self.check_consistency());
    }

    /// Adds a term to the Hamiltonian represented by `data`. The term acts on `act_on_indices`.
    /// Returns a handle to the term.
    pub fn add_term<Indices>(&mut self, data: TermData, act_on_indices: Indices) -> MatrixTermHandle where Indices: Into<Vec<usize>> {
        let act_on_indices = act_on_indices.into();
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
        self.total_offset += data.get_natural_offset();

        self.all_term_data.push(data);
        self.all_terms.push(MatrixTerm {
            act_on_indices,
            matrix_data_entry,
        });

        // Add an entry for lists of nodes
        self.list_of_nodes_by_term.push(vec![]);

        matrix_data_entry
    }

    /// Adds a diagonal node to the graph at `timeslice` representing the matrix term pointed to by `matrix_term_handle`.
    pub fn add_node(&mut self, timeslice: usize, matrix_term_handle: MatrixTermHandle) {
        let act_on_indices = self.all_terms[matrix_term_handle].act_on_indices.clone();
        self.insert_node(&timeslice, &act_on_indices, |context| DoublyLinkedNode {
            input_state: context.local_state.clone(),
            output_state: context.local_state.clone(),
            represents_term: MatrixTerm {
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

    /// Remove a node from the list of potentially flippable nodes.
    fn remove_from_flippable_list(&mut self, node: &DoublyLinkedNode<DOF>) {
        if let Some(index_to_remove) = node.index_of_entry_into_flippable_list {
            Self::handle_flippable_removal(
                index_to_remove,
                &mut self.list_of_nodes_with_flippable_outputs,
                &mut self.time_slices,
            );
        }
    }

    /// Perform bookkeeping surrounding a node removal, called by `remove_from_flippable_list`.
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

    /// Returns the number of nodes which represent `term`.
    pub fn get_count_for_term(&self, term: &MatrixTermHandle) -> usize {
        self.list_of_nodes_by_term[*term].len()
    }

    /// Starting at `starting_node_excluded`, step backwards through the worldline on relative index `rel_index` until `filter` returns true.
    pub fn iterate_backwards_through_nodes_until_match<'a, F>(
        &'a self,
        starting_node_excluded: &'a DoublyLinkedNode<DOF>,
        rel_index: usize,
        filter: F,
    ) -> Option<(&'a DoublyLinkedNode<DOF>, usize)>
    where
        F: Fn(&DoublyLinkedNode<DOF>, usize) -> bool,
    {
        self.iterate_through_nodes_until_match(
            starting_node_excluded,
            rel_index,
            filter,
            |node, rel_index| {
                self.get_previous_node_for_relative_dof(node, rel_index)
                    .unwrap_or_else(|| {
                        self.get_last_node_for_dof(&node.get_indices()[rel_index])
                            .expect("We know this DOF must have nodes")
                    })
            },
        )
    }

    /// Starting at `starting_node_excluded`, step forwards through the worldline on relative index `rel_index` until `filter` returns true.
    pub fn iterate_forwards_through_nodes_until_match<'a, F>(
        &'a self,
        starting_node_excluded: &'a DoublyLinkedNode<DOF>,
        rel_index: usize,
        filter: F,
    ) -> Option<(&'a DoublyLinkedNode<DOF>, usize)>
    where
        F: Fn(&DoublyLinkedNode<DOF>, usize) -> bool,
    {
        self.iterate_through_nodes_until_match(
            starting_node_excluded,
            rel_index,
            filter,
            |node, rel_index| {
                self.get_next_node_for_relative_dof(node, rel_index)
                    .unwrap_or_else(|| {
                        self.get_first_node_for_dof(&node.get_indices()[rel_index])
                            .expect("We know this DOF must have nodes")
                    })
            },
        )
    }

    /// Starting at `starting_node_excluded`, step through nodes until `filter` returns true. Stepping behavior is defined by the
    /// `successor` function.
    pub fn iterate_through_nodes_until_match<'a, F, G>(
        &'a self,
        starting_node_excluded: &'a DoublyLinkedNode<DOF>,
        rel_index: usize,
        filter: F,
        successor: G,
    ) -> Option<(&'a DoublyLinkedNode<DOF>, usize)>
    where
        F: Fn(&DoublyLinkedNode<DOF>, usize) -> bool,
        G: Fn(&'a DoublyLinkedNode<DOF>, usize) -> (&'a DoublyLinkedNode<DOF>, usize),
    {
        let mut node = starting_node_excluded;
        let mut rel_index = rel_index;
        let starting_t = node.timeslice;
        let mut current_t = usize::MAX;

        while current_t != starting_t {
            (node, rel_index) = successor(node, rel_index);
            if filter(node, rel_index) {
                return Some((node, rel_index));
            }
            current_t = node.timeslice;
        }

        None
    }

    /// For debugging purposes, print out the worldlines and operators.
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

    /// For debugging purposes, throw an error if an obvious inconsistency exists on `node` at `timeslice`.
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

    /// For debugging purposes, throw an error if an obvious inconsistency exists on any node in the graph.
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
                state[*global] = node.output_state[rel];
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

        for node in self.time_slices.iter().filter_map(|x| x.as_ref()) {
            let element = self.get_matrix_element_from_node(node);
            assert!(element > 0.0);
        }

        true
    }
}

/// A node in the QMC graph which points to all previous and following nodes which share a worldline.
pub struct DoublyLinkedNode<DOF: DOFTypeTrait> {
    input_state: Vec<DOF>,
    output_state: Vec<DOF>,
    represents_term: MatrixTerm,
    previous_node_index_for_variable: Vec<Option<Link<usize>>>,
    next_node_index_for_variable: Vec<Option<Link<usize>>>,
    timeslice: usize,

    // Keep track of where this is being tracked.
    index_of_entry_in_node_list_for_term: usize,
    index_of_entry_into_flippable_list: Option<usize>,
}

/// A term in the Hamiltonian must implement `MatrixTermData` with weights of type `T`. 
pub trait MatrixTermData<T> {
    /// Return the matrix entry connecting an `input` to an `output`, or the coefficient in front of |output><input|.
    fn get_matrix_entry(&self, input: usize, output: usize) -> T;
    /// The dimension of the input/output space.
    fn dim(&self) -> usize;
    /// For a fixed `input`, how distinct outputs, other than `output` have the same weight as `get_matrix_entry(input, output)`.
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

    /// If the term adds an offset for numerical stability, return the offset to "undo" the shift.
    fn get_natural_offset(&self) -> T;
}

/// The type of `GraphWeight::MatrixTermTrait` used for GenericQMC.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MatrixTerm {
    act_on_indices: Vec<usize>,
    matrix_data_entry: MatrixTermHandle,
}

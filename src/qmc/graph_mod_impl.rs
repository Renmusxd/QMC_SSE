use crate::qmc::{DoublyLinkedNode, GenericQMC, MatrixTermData};
use crate::traits::graph_traits::{DOFTypeTrait, GraphContext, GraphNode, Link, LinkedGraphNode, LinkedGraphNodeOutputs, TimeSlicedGraph};

impl<DOF: DOFTypeTrait, Data: MatrixTermData<f64>> TimeSlicedGraph for GenericQMC<DOF, Data> {
    type TimesliceIndex = usize;

    fn num_time_slices(&self) -> usize {
        self.time_slices.len()
    }

    fn iterate_time_slices(&self) -> impl Iterator<Item = Self::TimesliceIndex> {
        0..self.num_time_slices()
    }

    fn get_first_timeslice(&self) -> Option<Self::TimesliceIndex> {
        if self.time_slices.is_empty() {
            None
        } else {
            Some(0)
        }
    }

    fn get_next_timeslice(&self, timeslice: Self::TimesliceIndex) -> Option<Self::TimesliceIndex> {
        let t = timeslice + 1;
        if t < self.num_time_slices() {
            Some(t)
        } else {
            None
        }
    }

    fn get_first_timeslice_for_dof(
        &self,
        dof: &Self::DOFIndex,
    ) -> Option<&Link<Self::TimesliceIndex>> {
        self.first_nodes_for_dofs[*dof].as_ref()
    }

    fn get_node(&self, timeslice: &Self::TimesliceIndex) -> Option<&Self::Node> {
        self.time_slices[*timeslice].as_ref()
    }

    fn get_node_mut(&mut self, timeslice: &Self::TimesliceIndex) -> Option<&mut Self::Node> {
        self.time_slices[*timeslice].as_mut()
    }

    fn remove_node(&mut self, timeslice: &Self::TimesliceIndex) -> Option<Self::Node> {
        if let Some(node) = self.time_slices[*timeslice].take() {
            debug_assert!(node.is_diagonal());

            // For each variable, fix the previous and next nodes to point to each other.
            for (rel_var, global_var) in node
                .represents_term
                .act_on_indices
                .iter()
                .copied()
                .enumerate()
            {
                let link_to_previous = &node.previous_node_index_for_variable[rel_var];
                let link_to_next = &node.next_node_index_for_variable[rel_var];

                if let Some(link_to_previous) = link_to_previous {
                    // Modify previous node to skip over this one.
                    let previous_node_for_variable = self.time_slices[link_to_previous.timeslice]
                        .as_mut()
                        .expect("Previous node pointer to None");
                    debug_assert_eq!(
                        previous_node_for_variable.represents_term.act_on_indices
                            [link_to_previous.relative_index],
                        global_var
                    );
                    previous_node_for_variable.next_node_index_for_variable
                        [link_to_previous.relative_index] = link_to_next.clone()
                } else {
                    // This was the head. Update head to skip over.
                    self.first_nodes_for_dofs[global_var] = link_to_next.clone();
                }

                if let Some(link_to_next) = link_to_next {
                    // Modify next to skip over this one.
                    let next_node_for_variable = self.time_slices[link_to_next.timeslice]
                        .as_mut()
                        .expect("Previous node pointer to None");
                    debug_assert_eq!(
                        next_node_for_variable.represents_term.act_on_indices
                            [link_to_next.relative_index],
                        global_var
                    );
                    next_node_for_variable.previous_node_index_for_variable
                        [link_to_next.relative_index] = link_to_previous.clone();
                } else {
                    // This was the tail. Update tail to skip over.
                    self.last_nodes_for_dofs[global_var] = link_to_previous.clone();
                }
            }

            // Reduce count of nodes by one
            self.num_non_identity_terms -= 1;

            // Fix entry in list of nodes
            let list_entry =
                &mut self.list_of_nodes_by_term[node.represents_term.matrix_data_entry];
            let index_to_remove = node.index_of_entry_in_node_list_for_term;
            if index_to_remove < list_entry.len() - 1 {
                // Swap with last entry before popping.
                let last_index = list_entry.len() - 1;
                list_entry.swap(index_to_remove, last_index);
                list_entry.pop();

                let timeslice_to_fix = list_entry[index_to_remove];
                let node_to_fix = self.time_slices[timeslice_to_fix]
                    .as_mut()
                    .expect("Cannot point to empty timeslice");
                debug_assert_eq!(node_to_fix.index_of_entry_in_node_list_for_term, last_index);
                node_to_fix.index_of_entry_in_node_list_for_term = index_to_remove;
            } else {
                list_entry.pop();
            }

            // Fix entry in list of flippables
            self.remove_from_flippable_list(&node);

            debug_assert!(self.check_consistency());
            Some(node)
        } else {
            None
        }
    }

    fn get_link_to_next_node_by_relative_dof(
        &self,
        node: &Self::Node,
        rel_index: usize,
    ) -> Option<Link<Self::TimesliceIndex>> {
        node.next_node_index_for_variable[rel_index]
            .as_ref()
            .cloned()
    }

    fn get_next_nodes_for_node(
        &self,
        node: &Self::Node,
    ) -> Vec<Option<Link<Self::TimesliceIndex>>> {
        node.next_node_index_for_variable.clone()
    }

    fn insert_node_with_hint<F>(
        &mut self,
        timeslice: &Self::TimesliceIndex,
        variables: &[Self::DOFIndex],
        all_previous_node_indices: &[Option<Link<Self::TimesliceIndex>>],
        constructor: F,
    ) -> &Self::Node
    where
        F: FnOnce(GraphContext<Self::DOFType, Link<Self::TimesliceIndex>>) -> Self::Node,
    {
        // Debug check if the hint is accurate
        debug_assert!({
            all_previous_node_indices
                .iter()
                .enumerate()
                .all(|(global_var, prev_timeslice)| {
                    let pointer_to_next = match prev_timeslice {
                        None => {
                            // No previous.
                            self.first_nodes_for_dofs[global_var].as_ref()
                        }
                        Some(Link {
                            timeslice,
                            relative_index,
                        }) => {
                            let prev_node = self.time_slices[*timeslice]
                                .as_ref()
                                .expect("Backwards pointer should never be to None");
                            prev_node.next_node_index_for_variable[*relative_index].as_ref()
                        }
                    };
                    // Should be after `timeslice` or None.
                    pointer_to_next
                        .map(|h| h.timeslice >= *timeslice)
                        .unwrap_or(true)
                })
        });

        // Resize array to fit new node.
        if *timeslice >= self.time_slices.len() {
            self.time_slices.resize_with(*timeslice + 1, || None)
        }

        // If there was already a node here, remove it.
        self.remove_node(timeslice);
        debug_assert!(self.time_slices[*timeslice].is_none());

        let mut state = vec![];
        let mut links_back = vec![];
        let mut links_forward = vec![];

        variables
            .iter()
            .copied()
            .enumerate()
            .for_each(|(relative_index, global_index)| {
                // Get information from the nodes.
                let link_to_previous_node_for_variable =
                    all_previous_node_indices[global_index].as_ref();
                let (s, b, f) = if let Some(link) = link_to_previous_node_for_variable {
                    let n = self.time_slices[link.timeslice]
                        .as_ref()
                        .expect("Pointer to previous node finds empty timeslice.");
                    let s = n.output_state[link.relative_index];
                    let b = Some(link.clone());
                    let f = n.next_node_index_for_variable[link.relative_index].clone();
                    (s, b, f)
                } else {
                    let s = self.initial_state[global_index];
                    let b = None;
                    let f = self.first_nodes_for_dofs[global_index].clone();
                    (s, b, f)
                };

                debug_assert!(
                    match &b {
                        None => true,
                        Some(t) => t.timeslice < *timeslice,
                    },
                    "Backward pointer must be before timeslice."
                );

                debug_assert!(
                    match &f {
                        None => true,
                        Some(t) => t.timeslice > *timeslice,
                    },
                    "Forward pointer must be after timeslice."
                );

                // Now modify the nodes to point to the new spot.
                let link_to_me = Some(Link {
                    timeslice: *timeslice,
                    relative_index,
                });
                // Modify previous node or head
                if let Some(link) = b.as_ref() {
                    // There's a previous node. Point towards this new position.
                    let prev_node = self.time_slices[link.timeslice]
                        .as_mut()
                        .expect("Pointer to previous node finds empty timeslice.");
                    prev_node.next_node_index_for_variable[link.relative_index] =
                        link_to_me.clone();
                } else {
                    // There's no previous node. We are the new head.
                    self.first_nodes_for_dofs[global_index] = link_to_me.clone();
                }
                // Modify next node or tail.
                if let Some(link) = f.as_ref() {
                    // There's a next node. Point towards this new position.
                    let next_node = self.time_slices[link.timeslice]
                        .as_mut()
                        .expect("Pointer to next node finds empty timeslice.");
                    next_node.previous_node_index_for_variable[link.relative_index] = link_to_me;
                } else {
                    // There's no next node. We are the new tail.
                    self.last_nodes_for_dofs[global_index] = link_to_me;
                }

                // Save information for node construction purposes.
                state.push(s);
                links_back.push(b);
                links_forward.push(f);
            });

        let context = GraphContext {
            local_state: state,
            prev_node_slice: links_back,
            next_node_slice: links_forward,
        };

        let mut node = constructor(context);
        // Keep track of number of total nodes.
        self.num_non_identity_terms += 1;

        // Keep track of node in the list of nodes as well.
        let list_of_nodes_entry =
            &mut self.list_of_nodes_by_term[node.represents_term.matrix_data_entry];
        let index_into_list = list_of_nodes_entry.len();
        list_of_nodes_entry.push(*timeslice);
        node.index_of_entry_in_node_list_for_term = index_into_list;

        // And track total number of maybe flippable nodes.
        let matrix_data = &self.all_term_data[node.represents_term.matrix_data_entry];
        let input = DOF::index_dimension_slice(&node.input_state);
        let output = DOF::index_dimension_slice(&node.output_state);
        let n_flippable_outputs = matrix_data
            .get_number_of_equal_weight_outputs_for_input_distinct_from_output(input, output);
        if n_flippable_outputs > 0 {
            let index_to_insert = self.list_of_nodes_with_flippable_outputs.len();
            self.list_of_nodes_with_flippable_outputs.push(*timeslice);
            node.index_of_entry_into_flippable_list = Some(index_to_insert);
        }

        // Finally insert.
        self.time_slices[*timeslice] = Some(node);
        let res = self.time_slices[*timeslice]
            .as_ref()
            .expect("Node should really be here.");
        debug_assert!(self.check_node_consistency(*timeslice, res));
        res
    }
}

impl<DOF: DOFTypeTrait> LinkedGraphNode for DoublyLinkedNode<DOF> {
    type TimesliceIndex = usize;

    fn iterate_over_outputs(
        &self,
    ) -> impl Iterator<
        Item = LinkedGraphNodeOutputs<'_, Self::DOFIndex, Self::DOFType, Self::TimesliceIndex>,
    > {
        let a = self.represents_term.act_on_indices.iter();
        let b = self.output_state.iter();
        let c = self.next_node_index_for_variable.iter();
        a.zip(b).zip(c).map(|((a, b), c)| LinkedGraphNodeOutputs {
            index: a,
            value: b,
            next_node: c.as_ref(),
        })
    }
}

#[cfg(test)]
mod test_modify_graph {
    use super::*;
    use crate::terms::generic::GenericMatrixTermEnum;
    use crate::traits::graph_traits::GraphStateNavigator;

    #[test]
    fn test_modify_graph_simple() {
        let mut qmc = GenericQMC::<bool, _>::new(3);
        let term = qmc.add_term(GenericMatrixTermEnum::Identity { dim: 2 }, vec![0]);
        qmc.add_node(0, term);

        let first_node = qmc.get_first_node_for_dof(&0);
        assert!(first_node.is_some());

        let last_node = qmc.get_last_node_for_dof(&0);
        assert!(last_node.is_some());
    }

    #[test]
    fn test_modify_graph_chain() {
        let mut qmc = GenericQMC::<bool, _>::new(3);

        for i in 0..3 {
            let vars = (0..i + 1).collect::<Vec<_>>();
            let term = qmc.add_term(
                GenericMatrixTermEnum::Identity {
                    dim: 1 << vars.len(),
                },
                vars.clone(),
            );
            qmc.add_node(i, term);
        }

        // Check first nodes line up
        for i in 0..3 {
            let first_node = qmc.get_first_node_for_dof(&i);
            assert!(first_node.is_some());
            assert_eq!(
                first_node.map(|(n, _)| n.represents_term.matrix_data_entry),
                Some(i)
            );
        }

        // Check the final nodes line up
        for i in 0..3 {
            let last_node = qmc.get_last_node_for_dof(&i);
            assert_eq!(
                last_node.map(|(n, _)| n.represents_term.matrix_data_entry),
                Some(2)
            );
        }

        // Check the correct number of nodes for each in correct order.
        for i in 0..3 {
            let acc = qmc.iterate_over_nodes_for_dof(&i, i, |acc, node, _| {
                assert_eq!(node.represents_term.matrix_data_entry, acc);
                acc + 1
            });
            assert_eq!(acc, 3);
        }
    }
}

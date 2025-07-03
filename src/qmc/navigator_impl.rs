use crate::qmc::{DoublyLinkedNode, GenericQMC, MatrixTermData};
use crate::traits::graph_traits::{DOFTypeTrait, GraphNode, GraphStateNavigator};

impl<DOF: DOFTypeTrait, Data: MatrixTermData<f64>> GraphStateNavigator for GenericQMC<DOF, Data> {
    type Node = DoublyLinkedNode<DOF>;
    type DOFIndex = usize;
    type DOFType = DOF;

    fn get_initial_state(&self) -> &[Self::DOFType] {
        &self.initial_state
    }

    fn get_initial_state_mut(&mut self) -> &mut [Self::DOFType] {
        &mut self.initial_state
    }

    fn get_all_indices(&self) -> &[Self::DOFIndex] {
        &self.indices
    }

    fn get_first_node_for_dof(&self, index: &Self::DOFIndex) -> Option<(&Self::Node, usize)> {
        self.first_nodes_for_dofs[*index]
            .as_ref()
            .and_then(|index| {
                self.time_slices[index.timeslice]
                    .as_ref()
                    .map(|n| (n, index.relative_index))
            })
    }

    fn get_last_node_for_dof(&self, index: &Self::DOFIndex) -> Option<(&Self::Node, usize)> {
        self.last_nodes_for_dofs[*index].as_ref().and_then(|index| {
            self.time_slices[index.timeslice]
                .as_ref()
                .map(|n| (n, index.relative_index))
        })
    }

    fn get_next_node_for_relative_dof(
        &self,
        node: &Self::Node,
        index: usize,
    ) -> Option<(&Self::Node, usize)> {
        node.next_node_index_for_variable[index]
            .as_ref()
            .map(|next_index| {
                let node = self.time_slices[next_index.timeslice]
                    .as_ref()
                    .expect("Disagreement between next_node array and time_slices");
                (node, next_index.relative_index)
            })
    }

    fn get_previous_node_for_relative_dof(
        &self,
        node: &Self::Node,
        index: usize,
    ) -> Option<(&Self::Node, usize)> {
        node.previous_node_index_for_variable[index]
            .as_ref()
            .map(|previous_index| {
                let node = self.time_slices[previous_index.timeslice]
                    .as_ref()
                    .expect("Disagreement between prev_node array and time_slices");
                (node, previous_index.relative_index)
            })
    }

    fn iterate_over_all_nodes(&self) -> impl Iterator<Item = &Self::Node> {
        self.time_slices.iter().filter_map(|node| node.as_ref())
    }
}

impl<DOF: DOFTypeTrait> GraphNode for DoublyLinkedNode<DOF> {
    type DOFIndex = usize;
    type DOFType = DOF;

    fn get_indices(&self) -> &[Self::DOFIndex] {
        &self.represents_term.act_on_indices
    }

    fn get_input_state(&self) -> &[Self::DOFType] {
        &self.input_state
    }

    fn get_output_state(&self) -> &[Self::DOFType] {
        &self.output_state
    }

    fn get_input_state_mut(&mut self) -> &mut [Self::DOFType] {
        &mut self.input_state
    }

    fn get_output_state_mut(&mut self) -> &mut [Self::DOFType] {
        &mut self.output_state
    }

    fn get_relative_variable_index(&self, index: &Self::DOFIndex) -> Result<usize, ()> {
        self.get_indices()
            .iter()
            .copied()
            .find(|v| v.eq(index))
            .ok_or(())
    }
}

#[cfg(test)]
mod navigator_tests {
    use super::*;
    use crate::qmc::GenericQMC;

    #[test]
    fn test_add_node() {
        let mut qmc = GenericQMC::<bool>::new(3);
    }
}

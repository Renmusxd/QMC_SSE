use crate::qmc::GenericQMC;
use crate::traits::graph_traits::DOFTypeTrait;
use crate::traits::naive_flip_update::NaiveFlipUpdater;

impl<DOF: DOFTypeTrait> NaiveFlipUpdater for GenericQMC<DOF> {
    fn num_potential_flip_boundaries(&self) -> usize {
        self.total_maybe_flippable
    }

    fn get_potential_flip_boundary(&self, n: usize) -> Self::TimesliceIndex {
        self.which_terms_are_maybe_flippable
            .iter()
            .copied()
            .zip(self.list_of_nodes.iter())
            .filter(|(a, _)| *a)
            .map(|(_, b)| b)
            .fold(n, |acc, nodes| {
                if acc >= nodes.len() {
                    acc - nodes.len()
                } else {
                    return nodes[acc];
                }
            })
    }

    fn get_nth_equal_weight_output_state(&self, node: &Self::Node, n: usize) -> Vec<Self::DOFType> {
        todo!()
    }

    fn get_number_of_equal_weight_flip_possibilities(&self, node: &Self::Node) -> usize {
        todo!()
    }

    fn can_node_absorb_flip(
        &self,
        node: &Self::Node,
        new_input_state: &[Self::DOFType],
        acts_on_dof: &[Self::DOFIndex],
    ) -> bool {
        todo!()
    }

    fn get_relative_weight_change_for_new_state(
        &self,
        node: &Self::Node,
        new_state: &[Self::DOFType],
    ) -> Option<f64> {
        todo!()
    }
}

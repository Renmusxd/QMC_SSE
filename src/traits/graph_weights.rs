use crate::traits::graph_traits::{GraphNode, GraphStateNavigator};

pub trait GraphWeight: GraphStateNavigator {
    type MatrixTerm: MatrixTermTrait<Index = Self::DOFIndex>;

    fn get_possible_terms(&self) -> &[Self::MatrixTerm];
    fn get_matrix_element_from_term(
        &self,
        term: &Self::MatrixTerm,
        input_state: &[Self::DOFType],
        output_state: &[Self::DOFType],
    ) -> f64;

    fn get_diagonal_matrix_element_from_term(
        &self,
        term: &Self::MatrixTerm,
        state: &[Self::DOFType],
    ) -> f64 {
        self.get_matrix_element_from_term(term, state, state)
    }

    fn get_matrix_term_for_node<'a>(&self, node: &'a Self::Node) -> &'a Self::MatrixTerm;

    fn get_matrix_element_from_node(&self, node: &Self::Node) -> f64 {
        let term = self.get_matrix_term_for_node(node);
        let input_state = node.get_input_state();
        let output_state = node.get_output_state();
        self.get_matrix_element_from_term(term, input_state, output_state)
    }
    fn get_total_graph_weight_from_nodes(&self) -> f64 {
        self.iterate_over_all_nodes().map(|node| {
            self.get_matrix_element_from_node(node)
        }).product()
    }
}

pub trait MatrixTermTrait: Eq + PartialEq + Clone {
    type Index: Eq + PartialEq + Clone;
    fn get_indices_acted_on(&self) -> &[Self::Index];
}

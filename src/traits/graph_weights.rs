use crate::traits::graph_traits::{GraphNode, GraphStateNavigator};

/// A graph with nodes representing matrix elements and legs representing indices.
/// The net result is a total graph weight.
pub trait GraphWeight: GraphStateNavigator {
    /// The type of the matrix terms.
    type MatrixTerm: MatrixTermTrait<Index = Self::DOFIndex>;

    /// Get the list of terms in the Hamiltonian.
    fn get_possible_terms(&self) -> &[Self::MatrixTerm];

    /// Given an input and output for a term, output the matrix element.
    fn get_matrix_element_from_term(
        &self,
        term: &Self::MatrixTerm,
        input_state: &[Self::DOFType],
        output_state: &[Self::DOFType],
    ) -> f64;

    /// Get the diagonal matrix element associated with the state.
    fn get_diagonal_matrix_element_from_term(
        &self,
        term: &Self::MatrixTerm,
        state: &[Self::DOFType],
    ) -> f64 {
        self.get_matrix_element_from_term(term, state, state)
    }

    /// Given a node, output the associated term.
    fn get_matrix_term_for_node<'a>(&self, node: &'a Self::Node) -> &'a Self::MatrixTerm;

    /// Given the state set by input and output legs, output the matrix element.
    fn get_matrix_element_from_node(&self, node: &Self::Node) -> f64 {
        let term = self.get_matrix_term_for_node(node);
        let input_state = node.get_input_state();
        let output_state = node.get_output_state();
        self.get_matrix_element_from_term(term, input_state, output_state)
    }

    /// Get the total weight of the entire graph.
    fn get_total_graph_weight_from_nodes(&self) -> f64 {
        self.iterate_over_all_nodes()
            .map(|node| self.get_matrix_element_from_node(node))
            .product()
    }

    /// Get the number of each matrix term in the graph.
    fn get_counts_for_all_terms(&self) -> Vec<usize>;
}

/// A matrix term which acts on a fixed list of indices.
pub trait MatrixTermTrait: Eq + PartialEq + Clone {
    /// The index type.
    type Index: Eq + PartialEq + Clone;

    /// The fixed list of indices.
    fn get_indices_acted_on(&self) -> &[Self::Index];
}

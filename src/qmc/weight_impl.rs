use crate::qmc::{GenericMatrixTerm, GenericQMC, MatrixTermData};
use crate::traits::graph_traits::DOFTypeTrait;
use crate::traits::graph_weights::{GraphWeight, MatrixTermTrait};

impl<DOF: DOFTypeTrait, Data: MatrixTermData<f64>> GraphWeight for GenericQMC<DOF, Data> {
    type MatrixTerm = GenericMatrixTerm;

    fn get_possible_terms(&self) -> &[Self::MatrixTerm] {
        &self.all_terms
    }
    fn get_matrix_element_from_term(
        &self,
        term: &Self::MatrixTerm,
        input_state: &[Self::DOFType],
        output_state: &[Self::DOFType],
    ) -> f64 {
        let term_entry = term.matrix_data_entry;
        let term_data = &self.all_term_data[term_entry];
        let input_value = Self::DOFType::index_dimension_slice(input_state);
        let output_value = Self::DOFType::index_dimension_slice(output_state);
        term_data.get_matrix_entry(input_value, output_value)
    }

    fn get_matrix_term_for_node<'a>(&self, node: &'a Self::Node) -> &'a Self::MatrixTerm {
        &node.represents_term
    }
}

impl MatrixTermTrait for GenericMatrixTerm {
    type Index = usize;

    fn get_indices_acted_on(&self) -> &[Self::Index] {
        &self.act_on_indices
    }
}

#[cfg(test)]
mod weight_tests {
    use super::*;
    use crate::qmc::{DoublyLinkedNode, GenericMatrixTermEnum, GenericQMC};
    use crate::traits::graph_traits::TimeSlicedGraph;

    #[test]
    fn test_get_matrix_data() {
        let term_enum = GenericMatrixTermEnum::make_diagonal(vec![1, 2]);
        assert_eq!(term_enum.get_matrix_entry(0, 0), 1);
        assert_eq!(term_enum.get_matrix_entry(1, 1), 2);
        assert_eq!(term_enum.get_matrix_entry(1, 0), 0);
        assert_eq!(term_enum.get_matrix_entry(0, 1), 0);
    }

    #[test]
    fn test_get_matrix_terms() {
        let mut qmc = GenericQMC::<bool>::new(3);

        qmc.add_term(
            GenericMatrixTermEnum::make_diagonal(vec![1.0, 2.0]),
            vec![0],
        );
        assert_eq!(qmc.get_possible_terms().len(), 1);
    }

    #[test]
    fn add_node_connected_to_matrix_term() {
        let mut qmc = GenericQMC::<bool>::new(3);
        let term_handle = qmc.add_term(
            GenericMatrixTermEnum::make_diagonal(vec![1.0, 2.0]),
            vec![0],
        );
        qmc.add_node(0, term_handle);

        let node = qmc.get_node(&0).unwrap();
        let a = qmc.get_matrix_element_from_node(node);
        assert!((a - 1.0).abs() < f64::EPSILON);
    }
}

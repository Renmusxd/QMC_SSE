use crate::qmc::{GenericQMC, MatrixTermData};
use crate::traits::graph_traits::{DOFTypeTrait, GraphStateNavigator};
use ndarray::Array2;
use num_traits::Zero;
use sprs::{CsMat, TriMat};
use std::ops::Add;

/// Struct supports giving a Matrix representation of the underlying Hamiltonian.
pub trait GetMatrixRep {
    /// The precision of the representation.
    type P: Clone + Add<Output = Self::P> + Zero;

    /// Get a sparse representation of the Hamiltonian
    fn get_sparse_rep(&self) -> CsMat<Self::P> {
        let hilbert_d = self.get_hilbert_space_dimension();

        let mut mat = TriMat::<Self::P>::new((hilbert_d, hilbert_d));
        self.callback_on_each_matrix_entry(|row, column, value| {
            mat.add_triplet(row, column, value)
        });

        mat.to_csr()
    }

    /// Get a dense representation of the Hamiltonian
    fn get_dense_rep(&self) -> Array2<Self::P> {
        let hilbert_d = self.get_hilbert_space_dimension();

        let mut mat = Array2::zeros((hilbert_d, hilbert_d));
        self.callback_on_each_matrix_entry(|row, column, value| {
            mat[(row, column)] = value;
        });

        mat
    }

    /// Get the system Hilbert space dimension
    fn get_hilbert_space_dimension(&self) -> usize;

    /// Calls `callback` on each nonzero matrix entry, up to once per term in the Hamiltonian.
    fn callback_on_each_matrix_entry<F>(&self, callback: F)
    where
        F: FnMut(usize, usize, Self::P);
}

impl<DOF: DOFTypeTrait, Data: MatrixTermData<f64>, GC> GetMatrixRep for GenericQMC<DOF, Data, GC> {
    type P = f64;

    fn get_hilbert_space_dimension(&self) -> usize {
        let n = self.get_num_dof();
        DOF::local_dimension().pow(n as u32)
    }

    fn callback_on_each_matrix_entry<F>(&self, mut callback: F)
    where
        F: FnMut(usize, usize, Self::P),
    {
        let num_dof = self.get_num_dof();
        let hilbert_d = self.get_hilbert_space_dimension();
        let largest_term_ndof = self.all_terms.iter().map(|t| t.act_on_indices.len()).max();
        if let Some(largest_term_ndof) = largest_term_ndof {
            let mut input_subslice = vec![DOF::default(); largest_term_ndof];
            for input_index in 0..hilbert_d {
                let input_state = DOF::index_to_state_vec(input_index, num_dof);
                for term in &self.all_terms {
                    let mut output_state = input_state.clone();
                    let term_data = &self.all_term_data[term.matrix_data_entry];
                    let indices = &term.act_on_indices;
                    indices
                        .iter()
                        .zip(input_subslice.iter_mut())
                        .for_each(|(index, subslice)| {
                            *subslice = input_state[*index];
                        });

                    // input_subslice is now set.
                    let subslice_input_index =
                        DOF::index_dimension_slice(&input_subslice[0..indices.len()]);
                    for subslice_output_index in 0..term_data.dim() {
                        let matrix_element =
                            term_data.get_matrix_entry(subslice_input_index, subslice_output_index);
                        let subslice_output_vec =
                            DOF::index_to_state_vec(subslice_output_index, indices.len());
                        subslice_output_vec
                            .iter()
                            .zip(indices.iter())
                            .for_each(|(val, index)| {
                                output_state[*index] = val.clone();
                            });
                        // output_state is now set.
                        let output_index = DOF::index_dimension_slice(&output_state);
                        if input_index == output_index {
                            let offset = term_data.get_natural_offset();
                            callback(input_index, input_index, matrix_element - offset);
                        } else {
                            callback(output_index, input_index, matrix_element);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod matrix_rep_tests {
    use super::*;
    use crate::terms::tfim::TFIMTerm;
    use ndarray::Array1;

    #[test]
    fn test_single_qubit_x() {
        let mut qmc = GenericQMC::<bool, _>::new(1);
        qmc.add_term(TFIMTerm::X(1.0), vec![0]);

        let mat = qmc.get_dense_rep();
        let ground_truth = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        assert!(
            mat.iter()
                .zip(ground_truth.iter())
                .all(|(x, y)| (x - y).abs() < f64::EPSILON)
        )
    }

    #[test]
    fn test_double_qubit_x1() {
        let mut qmc = GenericQMC::<bool, _>::new(2);
        qmc.add_term(TFIMTerm::X(1.0), vec![0]);

        let mat = qmc.get_dense_rep();
        let ground_truth = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            ],
        )
        .unwrap();
        assert!(
            mat.iter()
                .zip(ground_truth.iter())
                .all(|(x, y)| (x - y).abs() < f64::EPSILON)
        )
    }

    #[test]
    fn test_double_qubit_x2() {
        let mut qmc = GenericQMC::<bool, _>::new(2);
        qmc.add_term(TFIMTerm::X(1.0), vec![1]);

        let mat = qmc.get_dense_rep();
        let ground_truth = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        println!("{:?}", mat);
        assert!(
            mat.iter()
                .zip(ground_truth.iter())
                .all(|(x, y)| (x - y).abs() < f64::EPSILON)
        )
    }

    #[test]
    fn test_double_qubit_zz() {
        let mut qmc = GenericQMC::<bool, _>::new(2);
        qmc.add_term(TFIMTerm::ZZ(1.0), vec![0, 1]);

        let mat = qmc.get_dense_rep();
        let d = Array1::from_vec(vec![-1.0, 1.0, 1.0, -1.0]);
        let ground_truth = Array2::from_diag(&d);
        assert!(
            mat.iter()
                .zip(ground_truth.iter())
                .all(|(x, y)| (x - y).abs() < f64::EPSILON)
        )
    }
}

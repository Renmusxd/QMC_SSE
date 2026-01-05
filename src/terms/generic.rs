use crate::qmc::MatrixTermData;
use crate::qmc::naive_flip_impl::MatrixTermFlippable;
use num_traits::{One, Zero};

/// A general term in a Hamiltonian, makes few assumptions and thus offers few speedups.
pub enum GenericMatrixTermEnum<T> {
    /// An identity operator acting on a Hilbert space of size `dim`.
    Identity {
        /// The dimension of the Hilbert sub-space
        dim: usize,
        /// The number of dofs acted on.
        num_dof: usize
    },
    /// A diagonal operator.
    Diagonal {
        /// The diagonal of the operator, each entry in the data vector is a matrix element.
        data: Vec<T>,
        /// The number of dofs acted on.
        num_dof: usize
    },
    /// A uniform operator, meaning all matrix entries are identical.
    Uniform {
        /// The value of the matrix entries.
        data: T,
        /// The dimension of the Hilbert sub-space
        dim: usize,
        /// The number of dofs acted on.
        num_dof: usize
    },
    /// An operator which can be expressed as a scale times a binary operator (only 0 and 1).
    UniformSparse {
        /// The scale of the operator
        data: T,
        /// The dimension of the Hilbert sub-space
        dim: usize,
        /// The various values of |b> for a given <a|.
        outputs_for_input: Vec<Vec<usize>>,
        /// The various values of <a| for a given |b>.
        inputs_for_output: Vec<Vec<usize>>,
        /// The number of dofs acted on.
        num_dof: usize
    },
    /// An operator represented as a matrix.
    Generic {
        /// Matrix entries in row-major form.
        data: Vec<T>,
        /// The dimension of the Hilbert sub-space.
        dim: usize,
        /// The number of dofs acted on.
        num_dof: usize
    },
}

impl<T> GenericMatrixTermEnum<T>
where
    T: One + Zero + Clone,
{
    /// Make a diagonal operator given the diagonal `data`.
    pub fn make_diagonal<VT>(data: VT, num_dof: usize) -> Self where VT: Into<Vec<T>> {
        Self::Diagonal { data: data.into(), num_dof }
    }

    /// Make an identity operator acting on a Hilbert sub-space of dimension `dim`.
    pub fn make_identity(dim: usize, num_dof: usize) -> Self {
        Self::Identity { dim, num_dof }
    }

    /// Make a uniform operator with all entries given by `data`.
    pub fn make_uniform(data: T, dim: usize, num_dof: usize) -> Self {
        Self::Uniform { data, dim, num_dof }
    }

    /// Make an operator with only 0 and `data` entries from a list of tuples:
    /// (input, output): |output><input|
    pub fn make_sparse_uniform(data: T, dim: usize, matrix_entries: Vec<(usize, usize)>, num_dof: usize) -> Self {
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
            num_dof,
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
            Self::Diagonal { data, .. } if input == output => data[input].clone(),
            Self::Diagonal { .. } => T::zero(),
            Self::Uniform { data, .. } => data.clone(),
            Self::Generic { data, dim, .. } => data[output * dim + input].clone(),
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
            Self::Diagonal { data, .. } => data.len(),
            Self::Uniform { dim, .. } => *dim,
            Self::UniformSparse { dim, .. } => *dim,
            Self::Generic { dim, .. } => *dim,
        }
    }
    /// None means no change, Some((old, new)) implies there may be a change.
    fn get_weight_change_for_diagonal(&self, old_state: usize, new_state: usize) -> Option<(T, T)> {
        match &self {
            Self::Identity { .. } | Self::Uniform { .. } => None,
            Self::Diagonal { data, .. } => Some((data[old_state].clone(), data[new_state].clone())),
            Self::Generic { data, dim, .. } => {
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

    fn get_natural_offset(&self) -> T {
        T::zero()
    }

    fn num_dof(&self) -> usize {
        match self {
            GenericMatrixTermEnum::Identity { num_dof, .. } |
            GenericMatrixTermEnum::Diagonal { num_dof, .. } |
            GenericMatrixTermEnum::Uniform { num_dof, .. } |
            GenericMatrixTermEnum::UniformSparse { num_dof, .. } |
            GenericMatrixTermEnum::Generic { num_dof, .. } => *num_dof,
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

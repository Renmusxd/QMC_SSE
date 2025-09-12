use num_traits::{One, Zero};
use crate::qmc::MatrixTermData;
use crate::qmc::naive_flip_impl::MatrixTermFlippable;

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
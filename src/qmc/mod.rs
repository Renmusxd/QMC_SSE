use crate::traits::graph_traits::{DOFTypeTrait, Link, TimeSlicedGraph};
use num_traits::{One, Zero};

mod diagonal_impl;
mod graph_mod_impl;
mod naive_flip_impl;
mod navigator_impl;
mod weight_impl;

pub type MatrixTermHandle = usize;

pub struct GenericQMC<DOF: DOFTypeTrait = bool> {
    initial_state: Vec<DOF>,
    indices: Vec<usize>,
    time_slices: Vec<Option<DoublyLinkedNode<DOF>>>,
    first_nodes_for_dofs: Vec<Option<Link<usize>>>,
    last_nodes_for_dofs: Vec<Option<Link<usize>>>,

    // List of all terms in Hamiltonian
    all_terms: Vec<GenericMatrixTerm>,
    all_term_data: Vec<GenericMatrixTermEnum<f64>>,
    which_terms_are_maybe_flippable: Vec<bool>,

    // Count of terms
    num_non_identity_terms: usize,

    // Keep track of nodes and which are maybe flippable.
    list_of_nodes: Vec<Vec<usize>>,
    total_maybe_flippable: usize,
}

impl<DOF: DOFTypeTrait> GenericQMC<DOF> {
    fn new(num_dofs: usize) -> Self {
        let state = (0..num_dofs).map(|_| DOF::default()).collect();
        Self::new_with_state(state)
    }

    fn new_with_state(initial_state: Vec<DOF>) -> Self {
        let n = initial_state.len();
        Self {
            initial_state,
            indices: (0..n).collect(),
            time_slices: vec![],
            first_nodes_for_dofs: vec![None; n],
            last_nodes_for_dofs: vec![None; n],
            all_terms: vec![],
            all_term_data: vec![],
            which_terms_are_maybe_flippable: vec![],
            num_non_identity_terms: 0,
            list_of_nodes: vec![],
            total_maybe_flippable: 0,
        }
    }

    fn set_minimum_timeslices(&mut self, m: usize) {
        if self.num_time_slices() < m {
            self.time_slices.resize_with(m, || None);
        }
    }

    fn add_term<Data: Into<GenericMatrixTermEnum<f64>>>(
        &mut self,
        data: Data,
        act_on_indices: Vec<usize>,
    ) -> MatrixTermHandle {
        let data = data.into();
        debug_assert_eq!(
            data.dim(),
            DOF::local_dimension().pow(act_on_indices.len() as u32)
        );

        let matrix_data_entry = self.all_term_data.len();
        self.all_term_data.push(data);
        self.all_terms.push(GenericMatrixTerm {
            act_on_indices,
            matrix_data_entry,
        });

        // Add an entry for lists of nodes
        self.which_terms_are_maybe_flippable.push(false); // TODO fix
        self.list_of_nodes.push(vec![]);

        matrix_data_entry
    }
}

pub struct DoublyLinkedNode<DOF: DOFTypeTrait> {
    input_state: Vec<DOF>,
    output_state: Vec<DOF>,
    represents_term: GenericMatrixTerm,
    previous_node_index_for_variable: Vec<Option<Link<usize>>>,
    next_node_index_for_variable: Vec<Option<Link<usize>>>,

    // Keep track of where this is being tracked.
    index_of_entry_in_node_list_for_term: usize,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GenericMatrixTerm {
    act_on_indices: Vec<usize>,
    matrix_data_entry: MatrixTermHandle,
}

pub enum GenericMatrixTermEnum<T> {
    Identity { dim: usize },
    Diagonal { data: Vec<T> },
    Uniform { data: T, dim: usize },
    UniformSparse { data: T, dim: usize, outputs_for_input: Vec<usize>, inputs_for_output: Vec<usize> },
    Generic { data: Vec<T>, dim: usize },
}

impl<T> GenericMatrixTermEnum<T>
where
    T: One + Zero + Clone,
{
    pub fn get_matrix_entry(&self, input: usize, output: usize) -> T {
        match &self {
            Self::Identity { .. } if input == output => T::one(),
            Self::Identity { .. } => T::zero(),
            Self::Diagonal { data } if input == output => data[input].clone(),
            Self::Diagonal { .. } => T::zero(),
            Self::Uniform { data, .. } => data.clone(),
            Self::Generic { data, dim } => data[output * dim + input].clone(),
            Self::UniformSparse { data, outputs_for_input, .. } => {
                
            }
        }
    }

    pub fn make_diagonal(data: Vec<T>) -> Self {
        Self::Diagonal { data }
    }
    pub fn dim(&self) -> usize {
        match self {
            Self::Identity { dim, .. } => *dim,
            Self::Diagonal { data } => data.len(),
            Self::Uniform { dim, .. } => *dim,
            Self::UniformSparse { dim, .. } => *dim,
            Self::Generic { dim, .. } => *dim,
        }
    }
}

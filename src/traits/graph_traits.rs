use rand::Rng;
use rand::distr::Uniform;
use std::cmp::Ordering;
use std::fmt::Debug;

/// This graph can be navigated by following nodes.
pub trait GraphStateNavigator {
    /// The type of nodes in the graph.
    type Node: GraphNode<DOFIndex = Self::DOFIndex, DOFType = Self::DOFType>;
    /// The type of the index.
    type DOFIndex: Eq + PartialEq + Clone + Ord + PartialOrd + Into<usize> + Debug;
    /// The type of DOF
    type DOFType: DOFTypeTrait;

    /// Get the initial state at the 0th timeslice.
    fn get_initial_state(&self) -> &[Self::DOFType];
    /// Get a mutable reference to the initial state.
    fn get_initial_state_mut(&mut self) -> &mut [Self::DOFType];
    /// Set the initial state at a given index.
    fn set_initial_state(&mut self, dof: &Self::DOFIndex, val: Self::DOFType) {
        let initial_state = self.get_initial_state_mut();
        initial_state[dof.clone().into()] = val;
    }

    /// Get the list of all indices, each of which maps to a worldline.
    fn get_all_indices(&self) -> &[Self::DOFIndex];

    /// Get the first node along the worldline associated with a given index.
    fn get_first_node_for_dof(&self, index: &Self::DOFIndex) -> Option<(&Self::Node, usize)>;

    /// Get the last node along the worldline associated with a given index.
    fn get_last_node_for_dof(&self, index: &Self::DOFIndex) -> Option<(&Self::Node, usize)>;

    /// Get the next node following the worldline indexed by the relative `index`. None if the last.
    fn get_next_node_for_relative_dof(
        &self,
        node: &Self::Node,
        rel_index: usize,
    ) -> Option<(&Self::Node, usize)>;

    /// Get the previous node following the worldline indexed by the relative `index`. None if first.
    fn get_previous_node_for_relative_dof(
        &self,
        node: &Self::Node,
        rel_index: usize,
    ) -> Option<(&Self::Node, usize)>;

    /// Get the number of DOFs in the graph.
    fn get_num_dof(&self) -> usize {
        self.get_all_indices().len()
    }

    /// Get the list of initial nodes, up to one per index.
    fn get_all_initial_nodes(&self) -> Vec<Option<(&Self::Node, usize)>> {
        self.get_all_indices()
            .iter()
            .map(|index| self.get_first_node_for_dof(index))
            .collect()
    }

    /// Get the next node following the worldline of the DOF indexed by absolute `index`.
    fn get_next_node_for_absolute_dof(
        &self,
        node: &Self::Node,
        index: &Self::DOFIndex,
    ) -> Option<Option<(&Self::Node, usize)>> {
        node.get_relative_variable_index(index)
            .map(|index| self.get_next_node_for_relative_dof(node, index))
    }

    /// Get the previous node following the worldline of the DOF indexed by absolute `index`.
    fn get_previous_node_for_absolute_dof(
        &self,
        node: &Self::Node,
        index: &Self::DOFIndex,
    ) -> Option<Option<(&Self::Node, usize)>> {
        node.get_relative_variable_index(index)
            .map(|index| self.get_previous_node_for_relative_dof(node, index))
    }

    /// Follow a worldline and apply callback to each node. Folds with `init`.
    fn iterate_over_nodes_for_dof<K, F>(&self, dof: &Self::DOFIndex, init: K, callback: F) -> K
    where
        F: Fn(K, &Self::Node, usize) -> K,
    {
        let mut node = self.get_first_node_for_dof(dof);
        let mut k = init;
        while let Some((n, rel_index)) = node {
            k = callback(k, n, rel_index);
            node = self.get_next_node_for_relative_dof(n, rel_index);
        }
        k
    }

    /// Iterate over all nodes in the graph.
    fn iterate_over_all_nodes(&self) -> impl Iterator<Item = &Self::Node>;

    /// Check the graph consistency, panics or returns false on failure.
    fn check_graph_consistency(&self) -> bool;
}

/// A node in the graph
pub trait GraphNode {
    /// The type used as an absolute index.
    type DOFIndex: Eq + PartialEq + Into<usize> + Clone;
    /// The type of DOFs.
    type DOFType: DOFTypeTrait;

    /// Get the list of absolute indices acted on by this node.
    fn get_indices(&self) -> &[Self::DOFIndex];

    /// Get the input state.
    fn get_input_state(&self) -> &[Self::DOFType];

    /// Get the input state as a mutable reference.
    // fn get_input_state_mut(&mut self) -> &mut [Self::DOFType];

    /// Get the output state.
    fn get_output_state(&self) -> &[Self::DOFType];

    /// Get the output state as a mutable reference.
    // fn get_output_state_mut(&mut self) -> &mut [Self::DOFType];

    /// Return an iterator over indices and input/output DOF refs for that index.
    fn iterate_over_indices_and_states(
        &self,
    ) -> impl IntoIterator<Item = (&Self::DOFIndex, &Self::DOFType, &Self::DOFType)>;

    /// Return an iterator over indices and mutable input/output DOF refs for that index.
    // fn iterate_over_indices_and_states_mut(
    //     &mut self,
    // ) -> impl IntoIterator<Item = (&Self::DOFIndex, &mut Self::DOFType, &mut Self::DOFType)>;

    /// Get the relative index of an absolute index if it is in the node.
    fn get_relative_variable_index(&self, index: &Self::DOFIndex) -> Option<usize>;

    /// Return true if the node is currently diagonal, meaning the input is equal to the output.
    fn is_diagonal(&self) -> bool {
        self.get_input_state() == self.get_output_state()
    }
}

/// Degrees of freedom must implement this trait.
pub trait DOFTypeTrait: Eq + PartialEq + Clone + Copy + Default + Debug {
    /// The integer local Hilbert space dimension.
    fn local_dimension() -> usize;

    /// Convert the DOF to a usize representation
    fn to_index(&self) -> usize;

    /// Convert the usize representation back to DOF
    fn from_index(index: usize) -> Self;

    /// Iterate through the values the DOF can take.
    fn iterate_through_values() -> impl Iterator<Item = Self>;

    /// Convert an iterator of DOFs into an integer.
    fn index_dimension<It>(it: It) -> usize
    where
        It: IntoIterator<Item = Self>,
    {
        it.into_iter()
            .fold((1, 0), |(mut mult, mut acc), v| {
                acc += mult * v.to_index();
                mult *= Self::local_dimension();
                (mult, acc)
            })
            .1
    }

    /// Convert a slice of DOFs into an integer.
    fn index_dimension_slice(dofs: &[Self]) -> usize {
        Self::index_dimension(dofs.iter().cloned())
    }

    /// Convert an integer into a vector of DOFs.
    fn index_to_state_vec(mut dof_index: usize, n_dof: usize) -> Vec<Self> {
        let d = Self::local_dimension();
        let mut output = vec![Self::default(); n_dof];

        for o in output.iter_mut() {
            *o = Self::from_index(dof_index % d);
            dof_index /= d;
        }
        output
    }

    /// Convert an integer into an array of DOFs.
    fn index_to_state<const N: usize>(mut dof_index: usize) -> [Self; N] {
        let d = Self::local_dimension();
        let mut output = [Self::default(); N];

        for o in output.iter_mut() {
            *o = Self::from_index(dof_index % d);
            dof_index /= d;
        }
        output
    }

    /// Get a random DOF value.
    fn get_random<R>(rng: &mut R) -> Self
    where
        R: Rng,
    {
        let choice = rng.sample(Uniform::new(0, Self::local_dimension()).unwrap());
        Self::iterate_through_values()
            .take(choice + 1)
            .last()
            .unwrap()
    }

    /// Get a random DOF value distinct from self.
    fn get_distinct_random<R>(&self, rng: &mut R) -> Self
    where
        R: Rng;
}

/// Graph context necessary for node insertion.
pub struct GraphContext<A, B> {
    /// The local state of a node.
    pub local_state: Vec<A>,
    /// The previous nodes, if any, for each DOF.
    pub prev_node_slice: Vec<Option<B>>,
    /// The next nodes, if any, for each DOF.
    pub next_node_slice: Vec<Option<B>>,
}

/// A graph with time slices indexed by `TimesliceIndex`.
pub trait TimeSlicedGraph: GraphStateNavigator
where
    Self::Node: LinkedGraphNode,
{
    /// The type of the index.
    type TimesliceIndex: Eq + PartialEq + Ord + PartialOrd + Clone + Debug;

    /// The number of time slices.
    fn num_time_slices(&self) -> usize;

    /// Iterate over all time slice indices.
    fn iterate_time_slices(&self) -> impl Iterator<Item = Self::TimesliceIndex>;

    /// Get the first timeslice.
    fn get_first_timeslice(&self) -> Option<Self::TimesliceIndex>;

    /// Get the next timeslice.
    fn get_next_timeslice(&self, timeslice: Self::TimesliceIndex) -> Option<Self::TimesliceIndex>;

    /// Get the first timeslice with a node for the chosen DOF on it.
    fn get_first_timeslice_for_dof(
        &self,
        dof: &Self::DOFIndex,
    ) -> Option<&Link<Self::TimesliceIndex>>;

    /// Get the node at the timeslice index.
    fn get_node(&self, timeslice: &Self::TimesliceIndex) -> Option<&Self::Node>;

    /// Get a mutable reference to the node at the timeslice index.
    fn get_node_mut(&mut self, timeslice: &Self::TimesliceIndex) -> Option<&mut Self::Node>;

    /// Remove the node at the given timeslice.
    fn remove_node(&mut self, timeslice: &Self::TimesliceIndex) -> Option<Self::Node>;

    /// Get the node at the timeslice index.
    fn get_link_to_next_node_by_relative_dof(
        &self,
        node: &Self::Node,
        rel_index: usize,
    ) -> Option<Link<Self::TimesliceIndex>>;

    /// Get all links "connected" to this node.
    fn get_next_nodes_for_node(&self, node: &Self::Node)
    -> Vec<Option<Link<Self::TimesliceIndex>>>;

    /// Insert a node at the given timeslice using a constructor.
    fn insert_node<F>(
        &mut self,
        timeslice: &Self::TimesliceIndex,
        variables: &[Self::DOFIndex],
        constructor: F,
    ) -> &Self::Node
    where
        F: FnOnce(GraphContext<Self::DOFType, Link<Self::TimesliceIndex>>) -> Self::Node,
    {
        let mut last_nodes: Vec<Option<Link<Self::TimesliceIndex>>> =
            vec![None; self.get_num_dof()];
        let mut tt = self.get_first_timeslice();
        while let Some(t) = tt {
            if t.eq(timeslice) {
                break;
            }

            if let Some(node) = self.get_node(&t) {
                node.iterate_over_outputs().enumerate().for_each(
                    |(
                        rel_index,
                        LinkedGraphNodeOutputs {
                            index: global_index,
                            ..
                        },
                    )| {
                        let global_index = global_index.clone().into();
                        last_nodes[global_index] = Some(Link {
                            timeslice: t.clone(),
                            relative_index: rel_index,
                        });
                    },
                );
            };

            tt = self.get_next_timeslice(t);
        }

        self.insert_node_with_hint(timeslice, variables, &last_nodes, constructor)
    }

    /// Insert a node using a hint of the previous nodes and variables.
    fn insert_node_with_hint<F>(
        &mut self,
        timeslice: &Self::TimesliceIndex,
        variables: &[Self::DOFIndex],
        all_previous_node_indices: &[Option<Link<Self::TimesliceIndex>>],
        constructor: F,
    ) -> &Self::Node
    where
        F: FnOnce(GraphContext<Self::DOFType, Link<Self::TimesliceIndex>>) -> Self::Node;
}

/// A node which can point to the next nodes
pub trait LinkedGraphNode: GraphNode {
    /// The type of the timeslice index.
    type TimesliceIndex: Eq + PartialEq + Ord + PartialOrd + Clone + Debug;

    /// Iterate over the output legs, the values of DOFs, and the next nodes.
    fn iterate_over_outputs(
        &self,
    ) -> impl Iterator<
        Item = LinkedGraphNodeOutputs<'_, Self::DOFIndex, Self::DOFType, Self::TimesliceIndex>,
    >;
}

/// A link to an output node.
pub struct LinkedGraphNodeOutputs<'a, DOFIndex, DOFType, TimesliceIndex>
where
    TimesliceIndex: Clone + Ord,
{
    /// The DOF index of the leg.
    pub index: &'a DOFIndex,
    /// The DOF value.
    pub value: &'a DOFType,
    /// The next node, if any, linked to.
    pub next_node: Option<&'a Link<TimesliceIndex>>,
}

/// A timeslice and a relative index.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Link<T>
where
    T: Eq + PartialEq + Ord + PartialOrd + Clone,
{
    /// The timeslice of a node
    pub timeslice: T,
    /// The relative index of a DOF.
    pub relative_index: usize,
}

impl<T> PartialOrd<Self> for Link<T>
where
    T: Eq + Ord + PartialEq + PartialOrd + Clone,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Link<T>
where
    T: Eq + PartialEq + Ord + PartialOrd + Clone,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.timeslice.cmp(&other.timeslice)
    }
}

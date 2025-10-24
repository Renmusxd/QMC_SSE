use std::cmp::Ordering;
use std::fmt::Debug;

pub trait GraphStateNavigator {
    type Node: GraphNode<DOFIndex = Self::DOFIndex, DOFType = Self::DOFType>;
    type DOFIndex: Eq + PartialEq + Clone + Ord + PartialOrd + Into<usize> + Debug;
    type DOFType: DOFTypeTrait;

    fn get_initial_state(&self) -> &[Self::DOFType];
    fn get_n_dof(&self) -> usize {
        self.get_initial_state().len()
    }
    fn get_initial_state_mut(&mut self) -> &mut [Self::DOFType];

    fn set_initial_state(&mut self, dof: &Self::DOFIndex, val: Self::DOFType) {
        let initial_state = self.get_initial_state_mut();
        initial_state[dof.clone().into()] = val;
    }

    fn get_all_indices(&self) -> &[Self::DOFIndex];
    fn get_first_node_for_dof(&self, index: &Self::DOFIndex) -> Option<(&Self::Node, usize)>;
    fn get_last_node_for_dof(&self, index: &Self::DOFIndex) -> Option<(&Self::Node, usize)>;
    fn get_next_node_for_relative_dof(
        &self,
        node: &Self::Node,
        index: usize,
    ) -> Option<(&Self::Node, usize)>;
    fn get_previous_node_for_relative_dof(
        &self,
        node: &Self::Node,
        index: usize,
    ) -> Option<(&Self::Node, usize)>;

    fn get_num_dof(&self) -> usize {
        self.get_all_indices().len()
    }

    fn get_all_initial_nodes(&self) -> Vec<Option<(&Self::Node, usize)>> {
        self.get_all_indices()
            .iter()
            .map(|index| self.get_first_node_for_dof(index))
            .collect()
    }

    fn get_next_node_for_absolute_dof(
        &self,
        node: &Self::Node,
        index: &Self::DOFIndex,
    ) -> Option<Option<(&Self::Node, usize)>> {
        node.get_relative_variable_index(index)
            .map(|index| self.get_next_node_for_relative_dof(node, index))
    }

    fn get_previous_node_for_absolute_dof(
        &self,
        node: &Self::Node,
        index: &Self::DOFIndex,
    ) -> Option<Option<(&Self::Node, usize)>> {
        node.get_relative_variable_index(index)
            .map(|index| self.get_previous_node_for_relative_dof(node, index))
    }

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

    fn iterate_over_all_nodes(&self) -> impl Iterator<Item = &Self::Node>;

    fn check_graph_consistency(&self) -> bool;
}

pub trait GraphNode {
    type DOFIndex: Eq + PartialEq + Into<usize> + Clone;
    type DOFType: DOFTypeTrait;

    fn get_indices(&self) -> &[Self::DOFIndex];
    fn get_input_state(&self) -> &[Self::DOFType];
    fn get_input_state_mut(&mut self) -> &mut [Self::DOFType];
    fn get_output_state(&self) -> &[Self::DOFType];
    fn get_output_state_mut(&mut self) -> &[Self::DOFType];

    fn iterate_over_indices_and_states(
        &self,
    ) -> impl IntoIterator<Item = (&Self::DOFIndex, &Self::DOFType, &Self::DOFType)>;
    fn iterate_over_indices_and_states_mut(
        &mut self,
    ) -> impl IntoIterator<Item = (&Self::DOFIndex, &mut Self::DOFType, &mut Self::DOFType)>;

    fn get_relative_variable_index(&self, index: &Self::DOFIndex) -> Option<usize>;
    fn is_diagonal(&self) -> bool {
        self.get_input_state() == self.get_output_state()
    }
}

pub trait DOFTypeTrait: Eq + PartialEq + Clone + Copy + Default + Debug {
    fn local_dimension() -> usize;

    fn to_index(&self) -> usize;
    fn from_index(index: usize) -> Self;

    fn iterate_through_values() -> impl Iterator<Item = Self>;

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

    fn index_dimension_slice(dofs: &[Self]) -> usize {
        Self::index_dimension(dofs.iter().cloned())
    }

    fn index_to_state_vec(mut dof_index: usize, n_dof: usize) -> Vec<Self> {
        let d = Self::local_dimension();
        let mut output = vec![Self::default(); n_dof];

        for o in output.iter_mut() {
            *o = Self::from_index(dof_index % d);
            dof_index /= d;
        }
        output
    }

    fn index_to_state<const N: usize>(mut dof_index: usize) -> [Self; N] {
        let d = Self::local_dimension();
        let mut output = [Self::default(); N];

        for o in output.iter_mut() {
            *o = Self::from_index(dof_index % d);
            dof_index /= d;
        }
        output
    }
}

pub struct GraphContext<A, B> {
    pub local_state: Vec<A>,
    pub prev_node_slice: Vec<Option<B>>,
    pub next_node_slice: Vec<Option<B>>,
}

pub trait TimeSlicedGraph: GraphStateNavigator
where
    Self::Node: LinkedGraphNode,
{
    type TimesliceIndex: Eq + PartialEq + Ord + PartialOrd + Clone + Debug;

    fn num_time_slices(&self) -> usize;

    fn iterate_time_slices(&self) -> impl Iterator<Item = Self::TimesliceIndex>;

    fn get_first_timeslice(&self) -> Option<Self::TimesliceIndex>;
    fn get_next_timeslice(&self, timeslice: Self::TimesliceIndex) -> Option<Self::TimesliceIndex>;

    fn get_first_timeslice_for_dof(
        &self,
        dof: &Self::DOFIndex,
    ) -> Option<&Link<Self::TimesliceIndex>>;

    fn get_node(&self, timeslice: &Self::TimesliceIndex) -> Option<&Self::Node>;
    fn get_node_mut(&mut self, timeslice: &Self::TimesliceIndex) -> Option<&mut Self::Node>;

    fn remove_node(&mut self, timeslice: &Self::TimesliceIndex) -> Option<Self::Node>;

    fn get_link_to_next_node_by_relative_dof(
        &self,
        node: &Self::Node,
        rel_index: usize,
    ) -> Option<Link<Self::TimesliceIndex>>;

    fn get_next_nodes_for_node(&self, node: &Self::Node)
    -> Vec<Option<Link<Self::TimesliceIndex>>>;

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

pub struct LinkedGraphNodeOutputs<'a, DOFIndex, DOFType, TimesliceIndex>
where
    TimesliceIndex: Clone + Ord,
{
    pub index: &'a DOFIndex,
    pub value: &'a DOFType,
    pub next_node: Option<&'a Link<TimesliceIndex>>,
}

pub trait LinkedGraphNode: GraphNode {
    type TimesliceIndex: Eq + PartialEq + Ord + PartialOrd + Clone + Debug;
    fn iterate_over_outputs(
        &self,
    ) -> impl Iterator<
        Item = LinkedGraphNodeOutputs<'_, Self::DOFIndex, Self::DOFType, Self::TimesliceIndex>,
    >;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Link<T>
where
    T: Eq + PartialEq + Ord + PartialOrd + Clone,
{
    pub timeslice: T,
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

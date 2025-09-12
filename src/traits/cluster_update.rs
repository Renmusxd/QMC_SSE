use crate::traits::graph_traits::{GraphNode, LinkedGraphNode, TimeSlicedGraph};
use crate::traits::graph_weights::GraphWeight;
use rand::Rng;
use std::hash::Hash;
use std::ops::MulAssign;

pub trait ClusterUpdater: TimeSlicedGraph + GraphWeight
where
    Self::Node: LinkedGraphNode + HasTimeslice<Self::TimesliceIndex>,
    Self::TimesliceIndex: Hash + Eq,
    Self::Node: 'static,
    Self::DOFType: 'static,
{
    type ChangeRecord;
    type ClusterManager<'a>: ClusterManager<
            &'a Self::Node,
            Self::DOFType,
            Self::DOFIndex,
            ChangeRecord = Self::ChangeRecord,
        >;

    fn cluster_update_starting_from_timeslice<R>(
        &mut self,
        timeslice: &Self::TimesliceIndex,
        direction: DirectionEnum,
        relative_index: usize,
        new_value: Self::DOFType,
        rng: &mut R,
    ) -> Result<bool, String>
    where
        R: Rng,
    {
        debug_assert!(self.check_graph_consistency());

        let mut cluster = self.get_cluster_manager();

        // We have to flip the spin to get started
        let node = self.get_node(timeslice).ok_or("Timeslice does not contain node.".to_string())?;
        let val = cluster.push_cluster_leg(Leg::new(node, direction, relative_index), new_value);
        let leg_changes = self.output_changes_for_spin_flip_with_default_state(
            node,
            direction,
            relative_index,
            val,
            rng,
        );
        let mut weight_change = leg_changes.get_weight_change();
        for (direction, relative_index, val) in leg_changes.get_iterator() {
            cluster.push_cluster_leg(Leg::new(node, direction, relative_index), val);
        }

        while let Some((leg, value)) = cluster.pop_cluster_leg() {
            let value = *value;
            debug_assert_eq!(
                Some(&value),
                cluster.get_leg_value(&leg),
                "Popped a leg with an invalid leg value. This should have been removed from the queue."
            );

            // Follow leg direction. Input legs connect to output legs and vice versa.
            let leg = self.follow_leg(leg);
            if let FollowResult::WrapBoundary(leg) = &leg {
                let absolute_index = &leg.get_node().get_indices()[leg.get_relative_index()];
                cluster.set_initial_state_value(absolute_index, value);
            }
            let leg = leg.get_value();

            // Let the cluster know a leg is coming into a node. If there's another leg which
            // cancels this out then returns None and closes the cluster, if they don't cancel then
            // we prioritize this incoming one and overwrite things.
            if let Some((leg, value)) = cluster.incoming_leg(leg, value) {
                let node = *leg.get_node();
                let direction = leg.get_direction();
                let relative_index = leg.get_relative_index();
                let matrix_term = self.get_matrix_term_for_node(node);

                let input_state = cluster
                    .get_input_state(node)
                    .unwrap_or_else(|| node.get_input_state());
                let output_state = cluster
                    .get_output_state(node)
                    .unwrap_or_else(|| node.get_output_state());
                let leg_changes = self.output_changes_for_spin_flip(
                    matrix_term,
                    input_state,
                    output_state,
                    direction,
                    relative_index,
                    &value,
                    rng,
                );
                weight_change *= leg_changes.get_weight_change();
                if weight_change.zero_weight() {
                    break;
                }
                cluster.set_leg_value(&leg, value);
                for (direction, relative_index, val) in leg_changes.get_iterator() {
                    cluster.push_cluster_leg(Leg::new(node, direction, relative_index), val);
                }
            }
        }

        let make_changes = match weight_change {
            WeightChange::NoChange => true,
            WeightChange::ZeroWeight => false,
            WeightChange::Factor(x) => rng.random::<f64>() < x,
        };

        if make_changes {
            let change_record = cluster.produce_change_record();
            self.apply_cluster_changes(change_record);
        }
        Ok(make_changes)
    }

    fn follow_leg(&self, leg: Leg<&Self::Node>) -> FollowResult<Leg<&Self::Node>> {
        match leg {
            Leg::Input(NodeLink {
                node,
                relative_index,
            }) => {
                let prev_node = self.get_previous_node_for_relative_dof(node, relative_index);
                let wrap = prev_node.is_none();
                let (node, relative_index) = prev_node.unwrap_or_else(|| {
                    let absolute_index = &node.get_indices()[relative_index];
                    self.get_last_node_for_dof(absolute_index)
                        .expect("Worldline cannot be empty.")
                });
                FollowResult::new(
                    Leg::Output(NodeLink {
                        node,
                        relative_index,
                    }),
                    wrap,
                )
            }
            Leg::Output(NodeLink {
                node,
                relative_index,
            }) => {
                let next_node = self.get_next_node_for_relative_dof(node, relative_index);
                let wrap = next_node.is_none();
                let (node, relative_index) = next_node.unwrap_or_else(|| {
                    let absolute_index = &node.get_indices()[relative_index];
                    self.get_first_node_for_dof(absolute_index)
                        .expect("Worldline cannot be empty.")
                });
                FollowResult::new(
                    Leg::Input(NodeLink {
                        node,
                        relative_index,
                    }),
                    wrap,
                )
            }
        }
    }

    fn output_changes_for_spin_flip_with_default_state<'a, R>(
        &'a self,
        node: &'a Self::Node,
        direction: DirectionEnum,
        relative_index: usize,
        new_value: &Self::DOFType,
        rng: &mut R,
    ) -> impl NodeClusterExpansion<Self::DOFType> + 'a
    where
        R: Rng,
    {
        let matrix_term = self.get_matrix_term_for_node(node);
        let input_state = node.get_input_state();
        let output_state = node.get_output_state();
        self.output_changes_for_spin_flip(
            matrix_term,
            input_state,
            output_state,
            direction,
            relative_index,
            new_value,
            rng,
        )
    }

    fn output_changes_for_spin_flip<R>(
        &self,
        term: &Self::MatrixTerm,
        input_state: &[Self::DOFType],
        output_state: &[Self::DOFType],
        direction: DirectionEnum,
        relative_index: usize,
        new_value: &Self::DOFType,
        rng: &mut R,
    ) -> impl NodeClusterExpansion<Self::DOFType> + '_
    where
        R: Rng;

    fn get_cluster_manager<'a>(&self) -> Self::ClusterManager<'a>;

    fn apply_cluster_changes(&mut self, manager: Self::ChangeRecord);
}

pub trait HasTimeslice<T> {
    fn get_timeslice(&self) -> &T;
}

pub enum FollowResult<T> {
    WrapBoundary(T),
    WithinBulk(T),
}

impl<T> FollowResult<T> {
    pub fn new(t: T, wrap: bool) -> Self {
        if wrap {
            Self::WrapBoundary(t)
        } else {
            Self::WithinBulk(t)
        }
    }

    pub fn get_value(self) -> T {
        match self {
            FollowResult::WrapBoundary(x) => x,
            FollowResult::WithinBulk(x) => x,
        }
    }

    pub fn value_ref(&self) -> &T {
        match self {
            FollowResult::WrapBoundary(x) => x,
            FollowResult::WithinBulk(x) => x,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DirectionEnum {
    Input,
    Output,
}

impl DirectionEnum {
    pub fn swap_direction(&self) -> Self {
        match self {
            DirectionEnum::Input => DirectionEnum::Output,
            DirectionEnum::Output => DirectionEnum::Input,
        }
    }
}

pub struct NodeLink<N> {
    node: N,
    relative_index: usize,
}

pub enum Leg<N> {
    Input(NodeLink<N>),
    Output(NodeLink<N>),
}

impl<N> Leg<N> {
    pub fn new(node: N, direction_enum: DirectionEnum, relative_index: usize) -> Self {
        match direction_enum {
            DirectionEnum::Input => Self::Input(NodeLink {
                node,
                relative_index,
            }),
            DirectionEnum::Output => Self::Output(NodeLink {
                node,
                relative_index,
            }),
        }
    }
    pub fn get_node(&self) -> &N {
        match self {
            Leg::Input(NodeLink { node, .. }) => node,
            Leg::Output(NodeLink { node, .. }) => node,
        }
    }
    pub fn get_relative_index(&self) -> usize {
        match self {
            Leg::Input(NodeLink { relative_index, .. }) => *relative_index,
            Leg::Output(NodeLink { relative_index, .. }) => *relative_index,
        }
    }

    pub fn get_direction(&self) -> DirectionEnum {
        match self {
            Leg::Input(_) => DirectionEnum::Input,
            Leg::Output(_) => DirectionEnum::Output,
        }
    }
}

pub trait ClusterManager<N, DOF, DOFIndex> {
    type ChangeRecord;

    /// Return error if there's a disagreement between existing value and new value
    fn push_cluster_leg(&mut self, leg: Leg<N>, value: DOF) -> &DOF;
    fn pop_cluster_leg(&mut self) -> Option<(Leg<N>, &DOF)>;
    /// Return error if there's a disagreement between existing value and new value
    fn set_leg_value(&mut self, leg: &Leg<N>, value: DOF) -> &DOF;
    fn get_leg_value(&self, leg: &Leg<N>) -> Option<&DOF>;
    /// Get the input state for a node.
    fn get_input_state(&self, node: N) -> Option<&[DOF]>;
    fn get_output_state(&self, node: N) -> Option<&[DOF]>;
    /// Sets leg value, and if an existing leg completes the path then returns None to prevent
    /// further cluster expansion.
    fn incoming_leg(&mut self, leg: Leg<N>, value: DOF) -> Option<(Leg<N>, DOF)>;
    /// Mark any changes which need to be made to the initial state.
    fn set_initial_state_value(&mut self, absolute_index: &DOFIndex, value: DOF);

    /// Dissolve the cluster manager to produce a lifetime-free record of changes.
    fn produce_change_record(self) -> Self::ChangeRecord;
}

pub trait NodeClusterExpansion<DOF> {
    fn get_weight_change(&self) -> WeightChange;
    fn get_iterator(self) -> impl IntoIterator<Item = (DirectionEnum, usize, DOF)>;
}

#[derive(Clone, Copy, Debug)]
pub enum WeightChange {
    NoChange,
    ZeroWeight,
    Factor(f64),
}

impl WeightChange {
    pub fn get_factor_mut(&mut self) -> Option<&mut f64> {
        match self {
            WeightChange::NoChange | WeightChange::ZeroWeight => None,
            WeightChange::Factor(x) => Some(x),
        }
    }

    pub fn zero_weight(&self) -> bool {
        matches!(self, WeightChange::ZeroWeight | WeightChange::Factor(0.0))
    }
}

impl MulAssign<WeightChange> for WeightChange {
    fn mul_assign(&mut self, rhs: WeightChange) {
        match (self, rhs) {
            (_, WeightChange::NoChange) => {}
            (WeightChange::ZeroWeight, _) => {}
            (x, WeightChange::ZeroWeight) => *x = WeightChange::ZeroWeight,
            (x @ WeightChange::NoChange, WeightChange::Factor(f)) => {
                *x = WeightChange::Factor(f);
            }
            (WeightChange::Factor(x), WeightChange::Factor(y)) => *x *= y,
        }
    }
}

impl<DOF, T> NodeClusterExpansion<DOF> for T
where
    T: IntoIterator<Item = (DirectionEnum, usize, DOF)>,
{
    fn get_weight_change(&self) -> WeightChange {
        WeightChange::NoChange
    }

    fn get_iterator(self) -> impl IntoIterator<Item = (DirectionEnum, usize, DOF)> {
        self
    }
}

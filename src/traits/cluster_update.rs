use crate::traits::WeightChange;
use crate::traits::graph_traits::{GraphNode, LinkedGraphNode, TimeSlicedGraph};
use crate::traits::graph_weights::GraphWeight;
use rand::Rng;
use std::hash::Hash;
use thiserror::Error;

/// A cluster updater propagates changes through worldlines to expand a cluster.
/// This encompasses loop updates as well as Wolf style clusters. Upon visiting a nodes and changing
/// and input/output, the node responds with resulting changes to the other legs, propagating the
/// cluster. It may also assign a weight cost to the change.
/// The net cluster update is accepted with a metropolis step using the net weight change.
pub trait ClusterUpdater: TimeSlicedGraph + GraphWeight
where
    Self::Node: LinkedGraphNode + HasTimeslice<Self::TimesliceIndex>,
    Self::TimesliceIndex: Hash + Eq,
    Self::Node: 'static,
    Self::DOFType: 'static,
{
    /// Details of what changes must be made to the graph to flip the cluster.
    type ChangeRecord;
    /// The object used to track the changes made by the cluster.
    /// When the cluster is done, uses the `ChangeRecord` struct to tell the graph
    /// how to update.
    type ClusterManager<'a>: ClusterManager<
            &'a Self::Node,
            Self::DOFType,
            Self::DOFIndex,
            ChangeRecord = Self::ChangeRecord,
        >;

    /// Run a "cluster update". This typically involves choosing a starting
    /// node and leg at random then calling `cluster_update_starting_from_timeslice`.
    fn cluster_update<R>(&mut self, rng: &mut R) -> Result<bool, ClusterError>
    where
        R: Rng;

    /// The machinery of the cluster update comes from this function. Given a starting leg, flip a
    /// DOF and then track the implications to other node legs. Repeat until the graph contains no
    /// inconsistencies.
    fn cluster_update_starting_from_timeslice<R>(
        &mut self,
        timeslice: &Self::TimesliceIndex,
        direction: DirectionEnum,
        relative_index: usize,
        new_value: Self::DOFType,
        rng: &mut R,
    ) -> Result<bool, ClusterError>
    where
        R: Rng,
    {
        debug_assert!(self.check_graph_consistency());

        let mut cluster = self.get_cluster_manager();

        // We have to flip the spin to get started
        let node = self
            .get_node(timeslice)
            .ok_or(ClusterError::TimesliceMissingNode)?;
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
                let absolute_index = &leg.get_node().get_indices()[*leg.get_relative_index()];
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
                    *relative_index,
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

        #[cfg(debug_assertions)]
        let before_total_weight = self.get_total_graph_weight_from_nodes();
        if make_changes {
            let change_record = cluster.produce_change_record();
            self.apply_cluster_changes(change_record);

            #[cfg(debug_assertions)]
            let after_total_weight = self.get_total_graph_weight_from_nodes();
            #[cfg(debug_assertions)]
            debug_assert!(
                ((after_total_weight / before_total_weight)
                    - weight_change.get_weight().unwrap_or(0.0))
                .abs()
                    < 1e-6
            );
        } else {
            #[cfg(debug_assertions)]
            let after_total_weight = self.get_total_graph_weight_from_nodes();
            #[cfg(debug_assertions)]
            debug_assert!((before_total_weight - after_total_weight).abs() < 1e-6);
        }

        Ok(make_changes)
    }

    /// Follow a `Leg`, meaning find the Output leg which connects to an Input on another node.
    fn follow_leg(&self, leg: Leg<&Self::Node>) -> FollowResult<Leg<&Self::Node>> {
        match leg {
            Leg::Input {
                node,
                relative_index,
            } => {
                let prev_node = self.get_previous_node_for_relative_dof(node, relative_index);
                let wrap = prev_node.is_none();
                let (node, relative_index) = prev_node.unwrap_or_else(|| {
                    let absolute_index = &node.get_indices()[relative_index];
                    self.get_last_node_for_dof(absolute_index)
                        .expect("Worldline cannot be empty.")
                });
                FollowResult::new(
                    Leg::Output {
                        node,
                        relative_index,
                    },
                    wrap,
                )
            }
            Leg::Output {
                node,
                relative_index,
            } => {
                let next_node = self.get_next_node_for_relative_dof(node, relative_index);
                let wrap = next_node.is_none();
                let (node, relative_index) = next_node.unwrap_or_else(|| {
                    let absolute_index = &node.get_indices()[relative_index];
                    self.get_first_node_for_dof(absolute_index)
                        .expect("Worldline cannot be empty.")
                });
                FollowResult::new(
                    Leg::Input {
                        node,
                        relative_index,
                    },
                    wrap,
                )
            }
        }
    }

    /// Given a change to a leg on a node, output the resulting changes to other nodes. Assumes
    /// the states if the inputs and outputs is otherwise not changed.
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

    /// Given a change to a leg on a node, output the resulting changes to other nodes. Takes the
    /// node's modified input and output states (from potential previous graph modifications).
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

    /// Get a cluster manager for use during the cluster update.
    fn get_cluster_manager<'a>(&self) -> Self::ClusterManager<'a>;

    /// Apply the changes as prescribed by the cluster manager.
    fn apply_cluster_changes(&mut self, manager: Self::ChangeRecord);
}

/// Indicates that this node keeps track of the timeslice it has been assigned to.
pub trait HasTimeslice<T> {
    /// Return the timeslice for the node.
    fn get_timeslice(&self) -> &T;
}

/// Following a leg can either:
pub enum FollowResult<T> {
    /// Cross the periodic boundary (such as large T back to 0).
    WrapBoundary(T),
    /// Remain in the bulk.
    WithinBulk(T),
}

impl<T> FollowResult<T> {
    /// Construct a new follow result, ending at the leg T.
    pub fn new(t: T, wrap: bool) -> Self {
        if wrap {
            Self::WrapBoundary(t)
        } else {
            Self::WithinBulk(t)
        }
    }

    /// Get the resulting leg.
    pub fn get_value(self) -> T {
        match self {
            FollowResult::WrapBoundary(x) => x,
            FollowResult::WithinBulk(x) => x,
        }
    }

    /// Get a reference to the resulting leg.
    pub fn value_ref(&self) -> &T {
        match self {
            FollowResult::WrapBoundary(x) => x,
            FollowResult::WithinBulk(x) => x,
        }
    }
}

/// Legs may either be inputs or outputs to the node.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DirectionEnum {
    /// The leg points to smaller timeslices.
    Input,
    /// The leg points to larger timeslices.
    Output,
}

impl DirectionEnum {
    /// Swap an input for an output or vice versa.
    pub fn swap_direction(&self) -> Self {
        match self {
            DirectionEnum::Input => DirectionEnum::Output,
            DirectionEnum::Output => DirectionEnum::Input,
        }
    }
}

/// A leg, either an input or an output to a node. Contains a reference to a node and the
/// relative index of the leg.
pub enum Leg<N> {
    /// An input leg.
    Input {
        /// The node referenced.
        node: N,
        /// The relative index of the leg
        relative_index: usize,
    },
    /// An output leg.
    Output {
        /// The node referenced.
        node: N,
        /// The relative index of the leg
        relative_index: usize,
    },
}

impl<N> Leg<N> {
    /// Make a new leg given a direction, a node, and a relative index.
    pub fn new(node: N, direction_enum: DirectionEnum, relative_index: usize) -> Self {
        match direction_enum {
            DirectionEnum::Input => Self::Input {
                node,
                relative_index,
            },
            DirectionEnum::Output => Self::Output {
                node,
                relative_index,
            },
        }
    }
    /// Get a reference to the node.
    pub fn get_node(&self) -> &N {
        match self {
            Leg::Input { node, .. } => node,
            Leg::Output { node, .. } => node,
        }
    }
    /// Get a reference to the relative index.
    pub fn get_relative_index(&self) -> &usize {
        match self {
            Leg::Input { relative_index, .. } => relative_index,
            Leg::Output { relative_index, .. } => relative_index,
        }
    }

    /// Get the direction the leg is facing.
    pub fn get_direction(&self) -> DirectionEnum {
        match self {
            Leg::Input { .. } => DirectionEnum::Input,
            Leg::Output { .. } => DirectionEnum::Output,
        }
    }
}

/// A cluster manager tracks which legs are still inconsistent, and which DOFs must be changed.
pub trait ClusterManager<N, DOF, DOFIndex> {
    /// When done, it outputs a ChangeRecord to instruct the graph on how to change.
    type ChangeRecord;

    /// Return error if there's a disagreement between existing value and new value
    fn push_cluster_leg(&mut self, leg: Leg<N>, value: DOF) -> &DOF;
    /// Return an inconsistent cluster leg.
    fn pop_cluster_leg(&mut self) -> Option<(Leg<N>, &DOF)>;
    /// Return error if there's a disagreement between existing value and new value
    fn set_leg_value(&mut self, leg: &Leg<N>, value: DOF) -> &DOF;
    /// Get the value of a leg if changed, or None if the cluster hasn't edited it.
    fn get_leg_value(&self, leg: &Leg<N>) -> Option<&DOF>;
    /// Get the input state for a node.
    fn get_input_state(&self, node: N) -> Option<&[DOF]>;
    /// Get the full output state of a node.
    fn get_output_state(&self, node: N) -> Option<&[DOF]>;
    /// Sets leg value, and if an existing leg completes the path then returns None to prevent
    /// further cluster expansion.
    fn incoming_leg(&mut self, leg: Leg<N>, value: DOF) -> Option<(Leg<N>, DOF)>;
    /// Mark any changes which need to be made to the initial state.
    fn set_initial_state_value(&mut self, absolute_index: &DOFIndex, value: DOF);

    /// Dissolve the cluster manager to produce a lifetime-free record of changes.
    fn produce_change_record(self) -> Self::ChangeRecord;
}

/// The result of a leg change, iterates over other leg changes and stores the associated weight
/// factor.
pub trait NodeClusterExpansion<DOF> {
    /// Get the weight factor of this set of changes.
    fn get_weight_change(&self) -> WeightChange;
    /// Iterator over other leg changes on this node.
    fn get_iterator(self) -> impl IntoIterator<Item = (DirectionEnum, usize, DOF)>;
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

/// Error types for cluster building.
#[derive(Error, Debug, Copy, Clone)]
pub enum ClusterError {
    /// An error for a timeslice missing a node.
    #[error("the cluster starting timeslice does not contain a node")]
    TimesliceMissingNode,
}

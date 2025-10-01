mod sampling;
pub mod utils;

use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, Mul};

use crate::traits::graph_traits::{GraphNode, Link, LinkedGraphNode, TimeSlicedGraph};
use crate::traits::graph_weights::GraphWeight;
use crate::traits::term_rotation_cluster_update::sampling::{SampleFraction, SampleFromWeights};
use crate::traits::term_rotation_cluster_update::utils::{
    allocate_terms_to_timeslices, get_weights_for_flips,
};
use crate::utils::logwrapper::LogWrapper;
use num_traits::{Float, One};
use rand::Rng;

pub trait MatrixTermRotationUpdate: TimeSlicedGraph + GraphWeight
where
    Self::Node: LinkedGraphNode,
{
    type SliceContext;
    type Cluster;
    type ClusterSlice: ClusterSliceData<Self::TimesliceIndex>;
    type ClusterFlipLabel;

    fn get_classical_slice_context(&self, timeslice: &Self::TimesliceIndex) -> Self::SliceContext;

    /// Returns none only if there are no operators in this worldline.
    /// Otherwise returns node and relative index.
    fn get_starting_operator_for_flippable_region_on_index(
        &self,
        context: &Self::SliceContext,
        index: &Self::DOFIndex,
    ) -> Option<(&Self::Node, usize)>;

    /// Returns none only if there are no operators in this worldline.
    /// Otherwise returns node and relative index.
    fn get_ending_operator_for_flippable_region_on_index(
        &self,
        context: &Self::SliceContext,
        index: &Self::DOFIndex,
    ) -> Option<(&Self::Node, usize)>;

    fn get_cluster_from_starting_index<R: Rng>(
        &self,
        context: &Self::SliceContext,
        index: &Self::DOFIndex,
        rng: &mut R,
    ) -> Self::Cluster;

    fn get_indices_in_cluster_at_timeslice(
        &self,
        context: &Self::SliceContext,
        cluster: &Self::Cluster,
        timeslice: &Self::TimesliceIndex,
    ) -> impl IntoIterator<Item = Self::DOFIndex>;

    fn get_flip_possibilities_for_cluster(
        &self,
        context: &Self::SliceContext,
        cluster: &Self::Cluster,
    ) -> impl IntoIterator<Item = Self::ClusterFlipLabel>;

    fn get_dof_value_for_index_and_flip(
        &self,
        context: &Self::SliceContext,
        index: &Self::DOFIndex,
        flip: &Self::ClusterFlipLabel,
    ) -> Self::DOFType;

    fn get_slice_for_start_of_cluster(&self, cluster: &Self::Cluster) -> Self::ClusterSlice;
    /// Returns the next slice, or None if the end of the cluster is reached.
    /// Should never return the start of cluster slice.
    fn get_next_slice(
        &self,
        cluster: &Self::Cluster,
        existing_cluster_slice: Self::ClusterSlice,
    ) -> Option<Self::ClusterSlice>;

    /// Using the output weights of any nodes at this timeslices, calculate the total weight of all diagonal terms which could be added to the state.
    fn get_weight_associated_with_term_at_slice_with_flip(
        &self,
        cluster: &Self::Cluster,
        cluster_slice: &Self::ClusterSlice,
        flip: &Self::ClusterFlipLabel,
        term: &Self::MatrixTerm,
    ) -> f64;

    fn get_rotatable_terms_for_cluster(
        &self,
        cluster: &Self::Cluster,
    ) -> impl IntoIterator<Item = &Self::MatrixTerm> {
        self.get_possible_terms()
    }

    fn is_term_rotatable(&self, cluster: &Self::Cluster, term: &Self::MatrixTerm) -> bool {
        true
    }

    fn get_total_weight_for_cluster_flip(
        &self,
        cluster: &Self::Cluster,
        cluster_slice: &Self::ClusterSlice,
        flip: &Self::ClusterFlipLabel,
    ) -> f64 {
        self.get_rotatable_terms_for_cluster(cluster)
            .into_iter()
            .map(|term| {
                self.get_weight_associated_with_term_at_slice_with_flip(
                    cluster,
                    cluster_slice,
                    flip,
                    term,
                )
            })
            .sum::<f64>()
    }

    fn get_slices_and_cluster_weights<P>(
        &self,
        context: &Self::SliceContext,
        cluster: &Self::Cluster,
    ) -> Vec<ClusterWeights<Self::TimesliceIndex, P>>
    where
        P: From<f64>,
    {
        let mut cluster_slice = self.get_slice_for_start_of_cluster(cluster);
        let first_timeslice = cluster_slice.get_timeslice().clone();

        let mut sections = vec![];
        let mut section = ClusterWeights::default();
        loop {
            let nodetype = NodeType::from(self.get_node(cluster_slice.get_timeslice()));
            if matches!(nodetype, NodeType::Diagonal | NodeType::Empty) {
                if section.weights_per_flip_label.is_empty() {
                    section.weights_per_flip_label = self
                        .get_flip_possibilities_for_cluster(context, cluster)
                        .into_iter()
                        .map(|flip| {
                            P::from(self.get_total_weight_for_cluster_flip(
                                cluster,
                                &cluster_slice,
                                &flip,
                            ))
                        })
                        .collect();
                }
                if matches!(nodetype, NodeType::Diagonal) {
                    section.n_occupied_slices += 1;
                }
                section
                    .timeslices
                    .push(cluster_slice.get_timeslice().clone());
            } else {
                sections.push(section);
                section = ClusterWeights::default();
            }

            cluster_slice = if let Some(cluster_slice) = self.get_next_slice(cluster, cluster_slice)
            {
                cluster_slice
            } else {
                if !section.timeslices.is_empty() {
                    sections.push(section);
                }
                break sections;
            };
            debug_assert_ne!(
                cluster_slice.get_timeslice(),
                &first_timeslice,
                "Cannot cycle back to start of cluster."
            );
        }
    }

    fn get_context_and_cluster_from_index_at_slice<R>(
        &self,
        timeslice: &Self::TimesliceIndex,
        starting_index: &Self::DOFIndex,
        rng: &mut R,
    ) -> (Self::SliceContext, Self::Cluster)
    where
        R: Rng,
    {
        let context = self.get_classical_slice_context(timeslice);
        let cluster = self.get_cluster_from_starting_index(&context, starting_index, rng);
        (context, cluster)
    }

    fn perform_cluster_update_on_index_at_slice<R>(
        &mut self,
        timeslice: &Self::TimesliceIndex,
        starting_index: &Self::DOFIndex,
        rng: &mut R,
    ) -> bool
    where
        R: Rng,
    {
        let (context, cluster) =
            self.get_context_and_cluster_from_index_at_slice(timeslice, starting_index, rng);
        let flip_labels = self
            .get_flip_possibilities_for_cluster(&context, &cluster)
            .into_iter()
            .collect::<Vec<_>>();
        let weights = self.get_slices_and_cluster_weights::<LogWrapper<f64>>(&context, &cluster);
        let (flip_label, timeslices) = get_term_allocation_for_weights(&flip_labels, &weights, rng);
        self.perform_cluster_update(cluster, flip_label, timeslices)
    }

    fn term_toggles_cluster_dofs(
        &self,
        cluster: &Self::Cluster,
        term: &Self::MatrixTerm,
    ) -> impl IntoIterator<Item = Self::DOFIndex>;

    fn get_variable_and_node_information_from_context(
        &self,
        context: Self::ClusterSlice,
        cluster: &Self::Cluster,
    ) -> (Vec<Self::DOFType>, Vec<Link<Self::TimesliceIndex>>);

    fn get_timeslices_of_end_of_cluster(&self, cluster: &Self::Cluster) -> Self::TimesliceIndex;

    fn perform_cluster_update(
        &mut self,
        cluster: Self::Cluster,
        cluster_flip_label: &Self::ClusterFlipLabel,
        term_timeslices: Vec<&Self::TimesliceIndex>,
    ) -> bool {
        let mut context = self.get_slice_for_start_of_cluster(&cluster);
        let mut t = Some(context.get_timeslice().clone());
        let (mut variables, mut last_nodes) =
            self.get_variable_and_node_information_from_context(context, &cluster);
        let mut dof_in_cluster = vec![false; self.get_n_dof()];

        let mut insertion_index = 0;
        while let Some(timeslice) = t {
            let node = self.get_node(&timeslice);
            if let Some(node) = node {
                // First, check if the node changes the cluster.

                // Set the previous node values.
                node.get_indices()
                    .iter()
                    .enumerate()
                    .for_each(|(relative_index, global_index)| {
                        let link = Link {
                            timeslice: timeslice.clone(),
                            relative_index,
                        };
                        let global_index: usize = global_index.clone().into();
                        last_nodes[global_index] = link;
                    });

                // Now, set the variable context.
                if !node.is_diagonal() {
                    let term = self.get_matrix_term_for_node(node);
                    self.term_toggles_cluster_dofs(&cluster, &term)
                        .into_iter()
                        .for_each(|global_index| {
                            let index = global_index.into();
                            dof_in_cluster[index] = !dof_in_cluster[index];
                        });

                    node.get_indices()
                        .iter()
                        .zip(node.get_input_state_mut().iter().zip(node.get_output_state().iter()))
                        .enumerate()
                        .for_each(|(relative_index, (global_index, (input_dof_value, output_dof_value))| {
                            // Set the previous nodes and set the variable values.
                            let global_index: usize = global_index.clone().into();
                            *input_dof_value = variables[global_index].clone();

                            if dof_in_cluster[global_index] {
                                variables[global_index] = dof_value.flip(cluster_flip_label);
                            } else {
                                variables[global_index] = dof_value.clone();
                            }
                        });
                }
            }

            t = self.get_next_timeslice(timeslice);
        }

        // let node = self.insert_node_with_hint(
        //     &t,
        //     &variables_for_term,
        //     &last_nodes,
        //     |context| Self::construct_node(&t, context, term_to_try),
        // );
        todo!()
    }
}

pub struct ClusterWeights<T, P = f64> {
    timeslices: Vec<T>,
    n_occupied_slices: usize,
    weights_per_flip_label: Vec<P>,
}

impl<T, P> Default for ClusterWeights<T, P> {
    fn default() -> Self {
        Self {
            timeslices: vec![],
            n_occupied_slices: 0,
            weights_per_flip_label: vec![],
        }
    }
}

pub trait ClusterSliceData<T> {
    fn get_timeslice(&self) -> &T;

    fn get_number_of_cluster_dof(&self) -> usize;
}

enum NodeType {
    Diagonal,
    Offdiagonal,
    Empty,
}

impl<GN> From<Option<&GN>> for NodeType
where
    GN: GraphNode,
{
    fn from(value: Option<&GN>) -> Self {
        match value {
            None => Self::Empty,
            Some(value) if value.is_diagonal() => Self::Diagonal,
            _ => Self::Offdiagonal,
        }
    }
}

impl<GN> From<&GN> for NodeType
where
    GN: GraphNode,
{
    fn from(value: &GN) -> Self {
        match value {
            value if value.is_diagonal() => Self::Diagonal,
            _ => Self::Offdiagonal,
        }
    }
}

fn get_term_allocation_for_weights<'a, 'b, F, T, R, P>(
    flip_labels: &'a [F],
    weights: &'b [ClusterWeights<T, P>],
    rng: &mut R,
) -> (&'a F, Vec<&'b T>)
where
    R: Rng,
    P: Mul<P, Output = P> + One + Clone + Debug + SampleFraction + SampleFromWeights + Sum,
    for<'c> P: Add<&'c P, Output = P>,
    for<'c, 'd> &'c P: Mul<&'d P, Output = P>,
{
    let total_weights = get_weights_for_flips(flip_labels.len(), weights)
        .into_iter()
        .collect::<Vec<_>>();
    let flip_index = P::sample_from_weights(&total_weights, rng);
    let flip_choice = &flip_labels[flip_index];
    let allocations = allocate_terms_to_timeslices(weights, flip_index, rng);
    (flip_choice, allocations)
}

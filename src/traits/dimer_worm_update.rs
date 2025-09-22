use crate::traits::WeightChange;
use crate::traits::graph_traits::{GraphContext, Link, LinkedGraphNode, TimeSlicedGraph};
use crate::traits::graph_weights::GraphWeight;
use rand::Rng;
use std::ops::Range;

pub trait MatrixTermRotationUpdate: TimeSlicedGraph + GraphWeight
where
    Self::Node: LinkedGraphNode,
{
    type SliceContext;
    type Cluster;
    type ClusterSlice;
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

    fn get_cluster_from_starting_index(
        &self,
        context: &Self::SliceContext,
        index: &Self::DOFIndex,
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

    fn get_flip_weights_for_cluster(
        &self,
        context: &Self::SliceContext,
        cluster: &Self::Cluster,
        flip: &Self::ClusterFlipLabel,
    ) -> WeightChange;

    /// Gives both flip possibilities and weights. Can be overridden for improved performance.
    fn get_flip_possibilities_and_weights_for_cluster(
        &self,
        context: &Self::SliceContext,
        cluster: &Self::Cluster,
    ) -> impl IntoIterator<Item = FlipInformation<Self::ClusterFlipLabel>> {
        self.get_flip_possibilities_for_cluster(context, cluster)
            .into_iter()
            .map(|flip_label| {
                let weight_change_inside_cluster =
                    self.get_flip_weights_for_cluster(context, cluster, &flip_label);
                FlipInformation {
                    flip_label,
                    weight_change_inside_cluster,
                }
            })
    }

    fn get_dof_value_for_index_and_flip(
        &self,
        context: &Self::SliceContext,
        index: &Self::DOFIndex,
        flip: &Self::ClusterFlipLabel,
    ) -> Self::DOFType;

    fn get_cluster_by_timeslice(
        &self,
        context: &Self::SliceContext,
        cluster: &Self::Cluster,
    ) -> impl IntoIterator<Item=(Self::TimesliceIndex, Self::ClusterSlice)>;

    fn get_graph_context_at_cluster_start(
        &self,
        original_slice_context: &Self::SliceContext,
        cluster: &Self::Cluster,
    ) -> GraphContext<Self::DOFType, Link<Self::TimesliceIndex>>;

    /// Fast forward to return the next "relevant" slice. A "relevant" change is one which may
    /// change the weights of any of the terms in `get_terms_affected_by_cluster_flips`. This
    /// typically means changes to any of the indices in the cluster or the values of DOF in or
    /// nearby the cluster.
    fn get_next_relevant_context_for_cluster(&self, context: GraphContext<Self::DOFType, Link<Self::TimesliceIndex>>, cluster: &Self::Cluster) -> GraphContext<Self::DOFType, Link<Self::TimesliceIndex>>;

    fn get_terms_affected_by_cluster_flips(&self, cluster: &Self::Cluster) -> impl IntoIterator<Item=Self::MatrixTerm>;

    fn get_timeslice_for_start_of_cluster(&self, context: &Self::SliceContext, cluster: &Self::Cluster) -> Self::TimesliceIndex;

    fn perform_cluster_update_on_index_at_slice<R>(
        &mut self,
        timeslice: &Self::TimesliceIndex,
        starting_index: &Self::DOFIndex,
        rng: &mut R,
    ) where
        R: Rng,
    {
        todo!()
    }
}

pub trait DimerWormUpdate: MatrixTermRotationUpdate
where
    Self::Node: LinkedGraphNode,
{
    type Bond: Eq;

    fn navigate_to_next_bond(
        &self,
        context: &Self::SliceContext,
        from_term: &Self::Bond,
    ) -> Self::Bond;

    fn get_cluster_from_bonds_list(
        &self,
        context: &Self::SliceContext,
        bonds: Vec<Self::Bond>,
    ) -> Self::Cluster;

    fn get_bonds_list(
        &self,
        context: &Self::SliceContext,
        starting_bond: Self::Bond,
    ) -> Vec<Self::Bond> {
        let next_bond = self.navigate_to_next_bond(context, &starting_bond);
        let mut bond_list = vec![starting_bond, next_bond];
        let mut ref_to_last = &bond_list[1];
        loop {
            let next_bond = self.navigate_to_next_bond(context, ref_to_last);
            if bond_list[0].eq(&next_bond) {
                return bond_list;
            }
            bond_list.push(next_bond);
            ref_to_last = &bond_list.last().as_ref().unwrap();
        }
    }

    fn get_cluster(
        &self,
        context: &Self::SliceContext,
        starting_bond: Self::Bond,
    ) -> Self::Cluster {
        let bond_list = self.get_bonds_list(&context, starting_bond);
        self.get_cluster_from_bonds_list(context, bond_list)
    }

    fn perform_cluster_update_on_slice<R>(
        &mut self,
        timeslice: &Self::TimesliceIndex,
        starting_bond: Self::Bond,
        rng: &mut R,
    ) where
        R: Rng,
    {
        todo!()
    }
}

pub struct FlipInformation<FlipLabel> {
    flip_label: FlipLabel,
    weight_change_inside_cluster: WeightChange,
}

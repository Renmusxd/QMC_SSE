use rand::Rng;
use crate::qmc::{GenericQMC, MatrixTermData};
use crate::traits::term_rotation_cluster_update::{ClusterSliceData, MatrixTermRotationUpdate};
use crate::traits::graph_traits::{DOFTypeTrait, GraphNode, GraphStateNavigator, Link};
use crate::traits::graph_weights::GraphWeight;

impl<DOF, Data, GI> MatrixTermRotationUpdate
    for GenericQMC<DOF, Data, GI>
where
    DOF: DOFTypeTrait + ClusterDOF,
    Data: MatrixTermData<f64> + ClusterEndcap,
    GI: SimpleClusterBuilder<DOF, Self::DOFIndex>,
{
    type SliceContext = GenericSliceContext<DOF>;
    type Cluster = GenericClusterInformation;

    type ClusterSlice = GenericClusterSlice;

    type ClusterFlipLabel = DOF::ClusterLabel;

    fn get_classical_slice_context(&self, timeslice: &Self::TimesliceIndex) -> Self::SliceContext {
        let mut num_unset = self.get_num_dof();
        let mut last_op_for_index = vec![None; self.get_num_dof()];

        for t in *timeslice..=0 {
            if let Some(op) = self.time_slices[t].as_ref() {
                for (relative_index, index) in op.get_indices().into_iter().copied().enumerate() {
                    if last_op_for_index[index].is_none() {
                        last_op_for_index[index] = Some(Link::<usize> {
                            timeslice: t,
                            relative_index,
                        });
                        num_unset -= 1;
                    }
                }
            }
            if num_unset == 0 {
                break;
            }
        }

        let state_at_slice = last_op_for_index
            .iter()
            .enumerate()
            .map(|(index, c)| {
                if let Some(ce) = c {
                    self.time_slices[ce.timeslice]
                        .as_ref()
                        .unwrap()
                        .output_state[ce.relative_index]
                        .clone()
                } else {
                    self.initial_state[index].clone()
                }
            })
            .collect::<Vec<_>>();

        let op_endcaps = last_op_for_index
            .iter()
            .enumerate()
            .map(|(index, op)| {
                match op {
                    Some(op) => Some(op),
                    None => self.first_nodes_for_dofs[index].as_ref(),
                }
                .and_then(|link| {
                    let starting_op = self.time_slices[link.timeslice]
                        .as_ref()
                        .expect("Link and context should guarantee op.");
                    let start_term = self.get_matrix_term_for_node(starting_op);
                    let start_data = &self.all_term_data[start_term.matrix_data_entry];

                    let top_endcap = if start_data.get_node_type().can_flip_states_arbitrarily() {
                        Some((starting_op, link.relative_index))
                    } else {
                        self.iterate_backwards_through_nodes_until_match(
                            starting_op,
                            link.relative_index,
                            |node, _| {
                                let term = self.get_matrix_term_for_node(node);
                                let data = &self.all_term_data[term.matrix_data_entry];
                                data.get_node_type().can_flip_states_arbitrarily()
                            },
                        )
                    }
                    .map(|(node, relative_index)| Link {
                        timeslice: node.timeslice,
                        relative_index,
                    })?;

                    let bottom_endcap = self
                        .iterate_forwards_through_nodes_until_match(
                            starting_op,
                            link.relative_index,
                            |node, _| {
                                let term = self.get_matrix_term_for_node(node);
                                let data = &self.all_term_data[term.matrix_data_entry];
                                data.get_node_type().can_flip_states_arbitrarily()
                            },
                        )
                        .map(|(node, relative_index)| Link {
                            timeslice: node.timeslice,
                            relative_index,
                        })
                        .expect("If there's a top endcap, there must be a bottom.");

                    Some(GenericClusterEndcaps {
                        start_op: top_endcap,
                        end_op: bottom_endcap,
                    })
                })
            })
            .collect::<Vec<_>>();

        Self::SliceContext {
            timeslice: *timeslice,
            state: state_at_slice,
            previous_ops_for_indices: last_op_for_index,
            op_endcaps,
        }
    }

    fn get_starting_operator_for_flippable_region_on_index(
        &self,
        context: &Self::SliceContext,
        index: &Self::DOFIndex,
    ) -> Option<(&Self::Node, usize)> {
        context.op_endcaps[*index].as_ref().and_then(|endcap| {
            let start_link = &endcap.start_op;
            self.time_slices[start_link.timeslice]
                .as_ref()
                .map(|node| (node, start_link.relative_index))
        })
    }

    fn get_ending_operator_for_flippable_region_on_index(
        &self,
        context: &Self::SliceContext,
        index: &Self::DOFIndex,
    ) -> Option<(&Self::Node, usize)> {
        context.op_endcaps[*index].as_ref().and_then(|endcap| {
            let end_link = &endcap.end_op;
            self.time_slices[end_link.timeslice]
                .as_ref()
                .map(|node| (node, end_link.relative_index))
        })
    }

    fn get_cluster_from_starting_index<R>(&self, _: &Self::SliceContext, _: &Self::DOFIndex, _: &mut R) -> Self::Cluster where R: Rng { todo!() }

    fn get_indices_in_cluster_at_timeslice(
        &self,
        context: &Self::SliceContext,
        cluster: &Self::Cluster,
        timeslice: &Self::TimesliceIndex,
    ) -> impl IntoIterator<Item = Self::DOFIndex> {
        let timeslice = *timeslice;
        cluster
            .indices_at_slice
            .iter()
            .copied()
            .filter(move |index| {
                if let Some(endcap) = context.op_endcaps[*index].as_ref() {
                    let start_t = endcap.start_op.timeslice;
                    let end_t = endcap.end_op.timeslice;
                    if start_t < end_t {
                        // t must be between the two
                        if timeslice <= end_t && timeslice > start_t {
                            true
                        } else {
                            false
                        }
                    } else if end_t < start_t {
                        // t must not be between the two
                        if timeslice < start_t && timeslice >= end_t {
                            false
                        } else {
                            true
                        }
                    } else {
                        // if they are equal, include the whole line
                        true
                    }
                } else {
                    true
                }
            })
    }

    fn get_flip_possibilities_for_cluster(
        &self,
        _context: &Self::SliceContext,
        _cluster: &Self::Cluster,
    ) -> impl IntoIterator<Item = Self::ClusterFlipLabel> {
        DOF::get_cluster_labels()
    }

    fn get_dof_value_for_index_and_flip(
        &self,
        context: &Self::SliceContext,
        index: &Self::DOFIndex,
        flip: &Self::ClusterFlipLabel,
    ) -> Self::DOFType {
        context.state[*index].new_value_for_cluster(flip)
    }

    fn get_slice_for_start_of_cluster(&self, cluster: &Self::Cluster) -> Self::ClusterSlice {
        todo!()
    }

    fn get_next_slice(&self, _: &<Self as MatrixTermRotationUpdate>::Cluster, _: <Self as MatrixTermRotationUpdate>::ClusterSlice) -> Option<<Self as MatrixTermRotationUpdate>::ClusterSlice> { todo!() }

    fn get_weight_associated_with_term_at_slice_with_flip(&self, cluster: &Self::Cluster, cluster_slice: &Self::ClusterSlice, flip: &Self::ClusterFlipLabel, term: &Self::MatrixTerm) -> f64 {
        todo!()
    }

    fn perform_cluster_update<R>(&mut self, cluster: Self::Cluster, cluster_flip_label: Self::ClusterFlipLabel) -> bool {
        todo!()
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum ClusterNodeType {
    DiagonalMatrixTerm,
    ArbitraryFlipsAllowed,
}

impl ClusterNodeType {
    fn can_flip_states_arbitrarily(&self) -> bool {
        matches!(self, ClusterNodeType::ArbitraryFlipsAllowed)
    }
}

pub trait ClusterEndcap {
    fn get_node_type(&self) -> ClusterNodeType;
}

pub trait SimpleClusterBuilder<DOF, I> {
    fn get_cluster_around_and_including_site(
        &self,
        state: &[DOF],
        index: &I,
    ) -> impl IntoIterator<Item = I>;
    fn get_cluster_around_and_excluding_site(
        &self,
        state: &[DOF],
        index: &I,
    ) -> impl IntoIterator<Item = I>;
}

impl<DOF, I> SimpleClusterBuilder<DOF, I> for ()
where
    I: Clone,
{
    fn get_cluster_around_and_including_site(
        &self,
        _state: &[DOF],
        index: &I,
    ) -> impl IntoIterator<Item = I> {
        Some(index.clone())
    }

    fn get_cluster_around_and_excluding_site(
        &self,
        _state: &[DOF],
        _index: &I,
    ) -> impl IntoIterator<Item = I> {
        None
    }
}

pub trait ClusterDOF {
    type ClusterLabel;

    fn new_value_for_cluster(&self, label: &Self::ClusterLabel) -> Self;

    fn get_cluster_labels() -> impl IntoIterator<Item=Self::ClusterLabel>;
}

pub struct GenericSliceContext<T> {
    timeslice: usize,
    state: Vec<T>,
    previous_ops_for_indices: Vec<Option<Link<usize>>>,
    op_endcaps: Vec<Option<GenericClusterEndcaps>>,
}

pub struct GenericClusterEndcaps {
    start_op: Link<usize>,
    end_op: Link<usize>,
}

pub struct GenericClusterInformation {
    indices_at_slice: Vec<usize>,
}

pub struct GenericClusterSlice;

impl<T> ClusterSliceData<T> for GenericClusterSlice {
    fn get_timeslice(&self) -> &T {
        todo!()
    }

    fn get_number_of_cluster_dof(&self) -> usize {
        todo!()
    }
}
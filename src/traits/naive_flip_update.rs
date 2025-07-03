use crate::traits::graph_traits::{GraphNode, Link, LinkedGraphNode, TimeSlicedGraph};
use crate::traits::graph_weights::GraphWeight;
use rand::Rng;
use std::cmp::Ordering;

pub trait NaiveFlipUpdater: TimeSlicedGraph + GraphWeight
where
    Self::Node: LinkedGraphNode,
{
    fn num_potential_flip_boundaries(&self) -> usize;
    fn get_potential_flip_boundary(&self, n: usize) -> Self::TimesliceIndex;
    fn is_node_potentially_flippable(&self, node: &Self::Node) -> bool;
    fn get_nth_equal_weight_output_state(&self, node: &Self::Node, n: usize) -> Vec<Self::DOFType>;
    fn get_number_of_equal_weight_flip_possibilities(&self, node: &Self::Node) -> usize;
    fn can_node_absorb_flip(
        &self,
        node: &Self::Node,
        new_input_state: &[Self::DOFType],
        acts_on_dof: &[Self::DOFIndex],
        originating_matrix_term: &Self::MatrixTerm,
    ) -> bool;

    /// Returns the relative change in weight for updating a diagonal node to a new state.
    /// None implies no change (equivalent to 1.0).
    fn get_relative_weight_change_for_new_state(
        &self,
        node: &Self::Node,
        new_state: &[Self::DOFType],
    ) -> Option<f64>;

    fn naive_flip_update<R>(&mut self, mut rng: R)
    where
        R: Rng,
    {
        // First, find a potential starting position for the flip.
        let n = self.num_potential_flip_boundaries();
        if n == 0 {
            return;
        }
        let starting_node_number = rng.sample(rand::distr::Uniform::new(0, n).unwrap());
        let start_pos = self.get_potential_flip_boundary(starting_node_number);
        debug_assert_eq!(
            self.get_node(&start_pos).map(|node| {
                self.is_node_potentially_flippable(node)
            }), Some(true), "Flip boundary term is not flippable."
        );
        self.naive_flip_update_starting_from_timeslice(start_pos, rng)
    }

    fn naive_flip_update_starting_from_timeslice<R>(
        &mut self,
        start_pos: Self::TimesliceIndex,
        mut rng: R,
    ) where
        R: Rng,
    {
        // Only in debug.
        let weight_before_update = self.get_total_graph_weight_from_nodes();
        debug_assert!(weight_before_update > f64::EPSILON, "Weight going into flip is zero");

        let node = self
            .get_node(&start_pos)
            .expect("potential flip boundary cannot point to empty timeslice");
        let originating_term = self.get_matrix_term_for_node(node).clone();

        // What variables are we working on
        let acting_on_dofs = node.get_indices().to_vec();

        // Get the trial output to flip to
        let num_configs = self.get_number_of_equal_weight_flip_possibilities(node);
        let flip_config_number = rng.sample(rand::distr::Uniform::new(0, num_configs).unwrap());
        let flip_config = self.get_nth_equal_weight_output_state(node, flip_config_number);
        if &flip_config == node.get_output_state() {
            // Already in correct configuration.
            return;
        }

        // Start working our way down the nodes
        let next_nodes = self.get_next_nodes_for_node(node);
        let first_result = check_from_starting_point(
            self,
            &originating_term,
            &flip_config,
            &acting_on_dofs,
            next_nodes,
        );
        let (end_flip_location, weight_change_on_flip) = match first_result {
            // If we hit the end of the line, just return. We made no changes and owe nothing.
            FlipCheckReturnValues::FailedToFlip => return,
            // If we hit the end of the timeline, got back to the start and find the next flip position.
            FlipCheckReturnValues::HitEndOfTimeline(w1) => {
                let beginning_nodes = acting_on_dofs
                    .iter()
                    .map(|dof| {
                        let res = self.get_first_timeslice_for_dof(dof).cloned();
                        debug_assert_ne!(res, None);
                        res
                    })
                    .collect::<Vec<_>>();
                let from_start_result = check_from_starting_point(
                    self,
                    &originating_term,
                    &flip_config,
                    &acting_on_dofs,
                    beginning_nodes,
                );
                match from_start_result {
                    FlipCheckReturnValues::FailedToFlip => return,
                    FlipCheckReturnValues::HitEndOfTimeline(_) => {
                        unreachable!("Cannot wrap around again, there should be at least one node!")
                    }
                    FlipCheckReturnValues::FoundEndOfFlipSection(t, w2) => {
                        (t, multiply_weight_changes(w1, w2))
                    }
                }
            }
            // If we found the end of the flippable section without wrapping, good for us.
            FlipCheckReturnValues::FoundEndOfFlipSection(t, w) => (t, w),
        };

        // MH step if needed.
        if let Some(w) = weight_change_on_flip {
            if w < 1.0 {
                let should_flip = rng.sample(rand::distr::Uniform::new(0.0, 1.0).unwrap()) < w;
                if !should_flip {
                    return;
                }
            }
        }

        // Ok, we're flipping bonds.
        edit_dof_from_starting_point(
            self,
            &flip_config,
            &acting_on_dofs,
            &start_pos,
            &end_flip_location,
        );

        let weight_after_update = self.get_total_graph_weight_from_nodes();
        debug_assert!(weight_after_update > f64::EPSILON, "Weight after flip is zero. Flip started at {:?} ended at {:?}", start_pos, end_flip_location);

        let target = weight_change_on_flip.unwrap_or(1.0);
        debug_assert!(
            ((weight_after_update / weight_before_update) - target) < f64::EPSILON,
            "Weights changed in an unexpected way: \t {:.3} -> {:.3} vs expected {:.3}\tFlip started at {:?} ended at {:?}",
            weight_before_update, weight_after_update, target, start_pos, end_flip_location
        )
    }
}

fn check_from_starting_point<G>(
    graph: &G,
    originating_term: &G::MatrixTerm,
    flip_config: &[G::DOFType],
    acting_on_dofs: &[G::DOFIndex],
    mut next_nodes: Vec<Option<Link<G::TimesliceIndex>>>,
) -> FlipCheckReturnValues<G::TimesliceIndex>
where
    G: NaiveFlipUpdater + ?Sized,
    G::Node: LinkedGraphNode,
{
    let mut total_weight_change = None;

    // Now repeatedly look at the next link.
    while let Some((rel_index, link)) = get_argmin_and_min_in_slice(&next_nodes) {
        let current_timeslice = link.timeslice.clone();
        let node_at_timeslice = graph
            .get_node(&link.timeslice)
            .expect("Cannot have link pointing to empty timeslice.");
        debug_assert_eq!(node_at_timeslice.get_indices()[link.relative_index], acting_on_dofs[rel_index]);

        if graph.can_node_absorb_flip(
            node_at_timeslice,
            &flip_config,
            &acting_on_dofs,
            originating_term,
        ) {
            // This will represent the end of the flipped region.
            return FlipCheckReturnValues::FoundEndOfFlipSection(
                current_timeslice,
                total_weight_change,
            );
        } else if node_at_timeslice.is_diagonal() {
            // Accumulate all the input state changes. What are the overlapping variables?
            let mut node_state = node_at_timeslice.get_input_state().to_vec();

            // Check all nodes in next_nodes. If they also point to this timeslice then we
            // should update the relevant state.
            next_nodes
                .iter()
                .zip(flip_config.iter())
                .enumerate()
                .filter_map(|(i, (ll, b))| ll.as_ref().map(|ll| (i, ll, b)))
                .filter(|(_, ll, _)| ll.timeslice == current_timeslice)
                .for_each(|(i, link, val)| {
                    debug_assert_eq!(
                        node_at_timeslice.get_indices()[link.relative_index],
                        acting_on_dofs[i]
                    );
                    node_state[link.relative_index] = val.clone();
                });
            let weight_change =
                graph.get_relative_weight_change_for_new_state(node_at_timeslice, &node_state);
            if weight_change == Some(0.0) {
                return FlipCheckReturnValues::FailedToFlip;
            }
            total_weight_change = multiply_weight_changes(total_weight_change, weight_change);

            // Update next_nodes
            next_nodes
                .iter_mut()
                .filter(|ll| {
                    if let Some(ll) = ll {
                        ll.timeslice == current_timeslice
                    } else {
                        false
                    }
                })
                .for_each(| link| {
                    *link = graph
                        .get_link_to_next_node_by_relative_dof(
                            node_at_timeslice,
                            link.as_ref().expect("Already checked that this was Some").relative_index,
                        );
                });
        } else {
            // Cannot handle offdiagonal overlaps.
            return FlipCheckReturnValues::FailedToFlip;
        }
    }
    FlipCheckReturnValues::HitEndOfTimeline(total_weight_change)
}

fn edit_dof_from_starting_point<G>(
    graph: &mut G,
    flip_config: &[G::DOFType],
    acting_on_dofs: &[G::DOFIndex],
    start_at_and_edit_output_of: &G::TimesliceIndex,
    stop_at_and_edit_input_of: &G::TimesliceIndex,
) where
    G: NaiveFlipUpdater + ?Sized,
    G::Node: LinkedGraphNode,
{
    // We do these one dof at a time.
    let it = acting_on_dofs.iter().zip(flip_config.iter()).enumerate();
    it.for_each(|(rel_index, (dof_index, dof_value))| {
        let node_at_start = graph
            .get_node_mut(start_at_and_edit_output_of)
            .expect("Cannot be None");
        debug_assert_eq!(&node_at_start.get_indices()[rel_index], dof_index);
        node_at_start.get_output_state_mut()[rel_index] = dof_value.clone();

        let node_at_start = graph
            .get_node(start_at_and_edit_output_of)
            .expect("Cannot be None");
        let mut link_to_next_node =
            graph.get_link_to_next_node_by_relative_dof(node_at_start, rel_index);

        // Loop until the link is pointing at the stop position.
        while link_to_next_node
            .as_ref()
            .map(|t| t.timeslice.ne(stop_at_and_edit_input_of))
            .unwrap_or(true)
        {
            if let Some(link) = link_to_next_node.as_ref() {
                // We hit a non-ending node. Edit the state and continue.
                let node_to_edit = graph
                    .get_node_mut(&link.timeslice)
                    .expect("Link should not point to empty timeslice.");
                node_to_edit.get_input_state_mut()[link.relative_index] = dof_value.clone();
                node_to_edit.get_output_state_mut()[link.relative_index] = dof_value.clone();
                // Now get pointer to next node.
                let node_to_edit = graph
                    .get_node(&link.timeslice)
                    .expect("Link should not point to empty timeslice.");
                link_to_next_node =
                    graph.get_link_to_next_node_by_relative_dof(node_to_edit, link.relative_index);
            } else {
                // We hit the end of the timeline without hitting the stop_at_end.
                debug_assert!(stop_at_and_edit_input_of <= start_at_and_edit_output_of);
                let initial_state = graph.get_initial_state_mut();
                initial_state[dof_index.clone().into()] = dof_value.clone();
                // Go back to start.
                link_to_next_node = graph.get_first_timeslice_for_dof(dof_index).cloned();
            }
        }

        // Now edit the last position.
        let link_to_next_node = link_to_next_node.expect("Link pointing to end should not be None");
        debug_assert_eq!(&link_to_next_node.timeslice, stop_at_and_edit_input_of);
        let node_to_edit = graph
            .get_node_mut(&link_to_next_node.timeslice)
            .expect("Link should not point to empty timeslice.");
        node_to_edit.get_input_state_mut()[link_to_next_node.relative_index] = dof_value.clone();
    });
}

enum FlipCheckReturnValues<T> {
    FailedToFlip,
    HitEndOfTimeline(Option<f64>),
    FoundEndOfFlipSection(T, Option<f64>),
}

fn multiply_weight_changes(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (None, None) => None,
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (Some(a), Some(b)) => Some(a * b),
    }
}

fn get_argmin_and_min_in_slice<T: Ord>(slice: &[Option<T>]) -> Option<(usize, &T)> {
    slice
        .iter()
        .enumerate()
        .min_by(|(_, x), (_, y)| match (x, y) {
            (None, None) => Ordering::Equal,
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (Some(x), Some(y)) => x.cmp(y),
        })
        .and_then(|(i, x)| x.as_ref().map(|x| (i, x)))
}

#[cfg(test)]
mod test_utilities_for_flipping {
    use super::*;

    #[test]
    fn test_get_argmin_and_min() {
        let res = get_argmin_and_min_in_slice(&[None, Some(13)]);
        assert_eq!(res, Some((1, &13)))
    }
}
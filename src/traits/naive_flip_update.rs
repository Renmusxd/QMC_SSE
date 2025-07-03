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
    fn get_nth_equal_weight_output_state(&self, node: &Self::Node, n: usize) -> Vec<Self::DOFType>;
    fn get_number_of_equal_weight_flip_possibilities(&self, node: &Self::Node) -> usize;
    fn can_node_absorb_flip(
        &self,
        node: &Self::Node,
        new_input_state: &[Self::DOFType],
        acts_on_dof: &[Self::DOFIndex],
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
        let start_pos = rng.sample(rand::distr::Uniform::new(0, n).unwrap());
        let start_pos = self.get_potential_flip_boundary(start_pos);
        let node = self
            .get_node(&start_pos)
            .expect("potential flip boundary cannot point to empty timeslice");

        // What variables are we working on
        let acting_on_dofs = node.get_indices().to_vec();

        // Get the trial output to flip to
        let flip_config_number = rng.sample(
            rand::distr::Uniform::new(0, self.get_number_of_equal_weight_flip_possibilities(node))
                .unwrap(),
        );
        let flip_config = self.get_nth_equal_weight_output_state(node, flip_config_number);

        // Start working our way down the nodes
        let next_nodes = self.get_next_nodes_for_node(node);
        let first_result =
            check_from_starting_point(self, &flip_config, &acting_on_dofs, next_nodes);
        let (end_flip_location, weight_change_on_flip, edit_initial_state) = match first_result {
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
                let from_start_result =
                    check_from_starting_point(self, &flip_config, &acting_on_dofs, beginning_nodes);
                match from_start_result {
                    FlipCheckReturnValues::FailedToFlip => return,
                    FlipCheckReturnValues::HitEndOfTimeline(_) => {
                        unreachable!("Cannot wrap around again, there should be at least one node!")
                    }
                    FlipCheckReturnValues::FoundEndOfFlipSection(t, w2) => {
                        (t, multiply_weight_changes(w1, w2), true)
                    }
                }
            }
            // If we found the end of the flippable section without wrapping, good for us.
            FlipCheckReturnValues::FoundEndOfFlipSection(t, w) => (t, w, false),
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
    }
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
    });
}

fn check_from_starting_point<G>(
    graph: &G,
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
        let node_at_timeslice = graph
            .get_node(&link.timeslice)
            .expect("Cannot have link pointing to empty timeslice.");

        if graph.can_node_absorb_flip(node_at_timeslice, &flip_config, &acting_on_dofs) {
            // This will represent the end of the flipped region.
            return FlipCheckReturnValues::FoundEndOfFlipSection(
                link.timeslice.clone(),
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
                .filter(|(_, ll, _)| ll.timeslice == link.timeslice)
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
                break;
            }
            total_weight_change = multiply_weight_changes(total_weight_change, weight_change);
        } else {
            // Cannot handle offdiagonal overlaps.
            return FlipCheckReturnValues::FailedToFlip;
        }
    }
    FlipCheckReturnValues::HitEndOfTimeline(total_weight_change)
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
            (Some(_), None) => Ordering::Greater,
            (None, Some(_)) => Ordering::Less,
            (Some(x), Some(y)) => x.cmp(y),
        })
        .and_then(|(i, x)| x.as_ref().map(|x| (i, x)))
}

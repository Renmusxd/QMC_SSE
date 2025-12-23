use crate::qmc::{DoublyLinkedNode, GenericQMC, MatrixTermData};
use crate::traits::cluster_update::{
    ClusterManager, ClusterUpdater, DirectionEnum, HasTimeslice, Leg, NodeClusterExpansion,
};
use crate::traits::graph_traits::{DOFTypeTrait, GraphNode, TimeSlicedGraph};
use rand::Rng;
use std::collections::HashMap;
use num_traits::Zero;
use rand::distr::Uniform;

impl<DOF: DOFTypeTrait, Data: MatrixTermData<f64>, GC> ClusterUpdater for GenericQMC<DOF, Data, GC>
where
    Data: TermClusterExpander<DOF>,
    DOF: 'static,
{
    type ChangeRecord = (
        HashMap<usize, Self::DOFType>,
        HashMap<usize, ManagerData<Self::DOFType>>,
    );
    type ClusterManager<'a> = GenericClusterManager<'a, Self::DOFType>;

    fn cluster_update<R>(&mut self, rng: &mut R) -> Result<bool, String>
    where
        R: Rng
    {
        if self.num_non_identity_terms.is_zero() {
            return Ok(true);
        }

        let choice = rng.sample(Uniform::new(0, self.num_non_identity_terms).unwrap());
        let choice_timeslice = self.list_of_nodes_by_term.iter().try_fold(choice, |choice, node_list| {
            if choice < node_list.len() {
                Err(node_list[choice])
            } else {
                Ok(choice - node_list.len())
            }
        }).expect_err("Choice must always select a node.");

        let direction = if rng.random::<bool>() { DirectionEnum::Input } else { DirectionEnum::Output };
        let node = self.get_node(&choice_timeslice).expect("Choice timeslice must point to node");
        let relative_index = rng.sample(Uniform::new(0, node.input_state.len()).unwrap());
        let old_dof_value = match direction {
            DirectionEnum::Input => &node.input_state[relative_index],
            DirectionEnum::Output => &node.output_state[relative_index],
        };
        let new_dof_value = old_dof_value.get_distinct_random(rng);
        self.cluster_update_starting_from_timeslice(&choice_timeslice, direction, relative_index, new_dof_value, rng)
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
        R: Rng,
    {
        let term_data = &self.all_term_data[term.matrix_data_entry];
        term_data.output_changes_for_spin_flip(
            input_state,
            output_state,
            direction,
            relative_index,
            new_value,
            rng,
        )
    }

    fn get_cluster_manager<'a>(&self) -> Self::ClusterManager<'a> {
        GenericClusterManager::default()
    }

    fn apply_cluster_changes<'a>(&mut self, changes: Self::ChangeRecord) {
        let (initial_state, timeline_changes) = changes;
        for (absolute_index, value) in initial_state {
            self.initial_state[absolute_index] = value;
        }
        for (timeslice, data) in timeline_changes {
            let node = self.time_slices[timeslice]
                .as_mut()
                .expect("There should be a node here since it was in the cluster.");

            node.input_state = data.input;
            node.output_state = data.output;
        }
    }
}

impl<T> HasTimeslice<usize> for DoublyLinkedNode<T>
where
    T: DOFTypeTrait,
{
    fn get_timeslice(&self) -> &usize {
        &self.timeslice
    }
}

#[derive(Default)]
pub struct GenericClusterManager<'a, T>
where
    T: DOFTypeTrait,
{
    initial_state: HashMap<usize, T>,
    cluster_legs: Vec<Leg<&'a DoublyLinkedNode<T>>>,
    timeslice_to_leg: HashMap<(usize, DirectionEnum, usize), usize>,
    timeslice_to_data: HashMap<usize, ManagerData<T>>,
}

#[derive(Default, Clone)]
pub struct ManagerData<T> {
    input: Vec<T>,
    output: Vec<T>,
}

impl<T> ManagerData<T> {
    fn set(&mut self, direction_enum: DirectionEnum, relative_index: usize, value: T) -> &T {
        match direction_enum {
            DirectionEnum::Input => {
                self.input[relative_index] = value;
                &self.input[relative_index]
            }
            DirectionEnum::Output => {
                self.output[relative_index] = value;
                &self.output[relative_index]
            }
        }
    }

    fn get(&self, direction_enum: DirectionEnum, relative_index: usize) -> &T {
        match direction_enum {
            DirectionEnum::Input => &self.input[relative_index],
            DirectionEnum::Output => &self.output[relative_index],
        }
    }

    fn get_input(&self) -> &[T] {
        &self.input
    }
    fn get_output(&self) -> &[T] {
        &self.output
    }
}

impl<'a, T> ClusterManager<&'a DoublyLinkedNode<T>, T, usize> for GenericClusterManager<'a, T>
where
    T: DOFTypeTrait,
{
    type ChangeRecord = (HashMap<usize, T>, HashMap<usize, ManagerData<T>>);

    fn push_cluster_leg(&mut self, leg: Leg<&'a DoublyLinkedNode<T>>, value: T) -> &T {
        let timeslice = *leg.get_node().get_timeslice();
        let direction = leg.get_direction();
        let relative_index = leg.get_relative_index();
        let key = (timeslice, direction, relative_index);

        // First check if there's a leg already there.
        if let Some(leg_index) = self.timeslice_to_leg.get(&key) {
            // There's already an open leg, reuse and overwrite.
            self.cluster_legs[*leg_index] = leg;
        } else {
            // If there's no leg, push to end of array.
            let leg_index = self.cluster_legs.len();
            self.cluster_legs.push(leg);
            self.timeslice_to_leg.insert(key, leg_index);
        }

        let entry = self.timeslice_to_data.entry(timeslice).or_insert_with(|| {
            let (input, output) = self
                .cluster_legs
                .last()
                .map(|leg| {
                    let node = leg.get_node();
                    (node.get_input_state(), node.get_output_state())
                })
                .expect("We just added a leg. Where did it go?");
            ManagerData {
                input: input.to_vec(),
                output: output.to_vec(),
            }
        });
        entry.set(direction, relative_index, value)
    }

    fn pop_cluster_leg(&mut self) -> Option<(Leg<&'a DoublyLinkedNode<T>>, &T)> {
        let leg = self.cluster_legs.pop();
        leg.map(|leg| {
            let value = self
                .timeslice_to_data
                .get(leg.get_node().get_timeslice())
                .map(|data| data.get(leg.get_direction(), leg.get_relative_index()))
                .expect("Leg cannot exist which points to empty data");
            (leg, value)
        })
    }

    fn set_leg_value(&mut self, leg: &Leg<&'a DoublyLinkedNode<T>>, value: T) -> &T {
        let timeslice = *leg.get_node().get_timeslice();
        let direction = leg.get_direction();
        let relative_index = leg.get_relative_index();
        let entry = self
            .timeslice_to_data
            .entry(timeslice)
            .or_insert_with(|| ManagerData {
                input: leg.get_node().input_state.clone(),
                output: leg.get_node().output_state.clone(),
            });
        entry.set(direction, relative_index, value)
    }

    fn get_leg_value(&self, leg: &Leg<&'a DoublyLinkedNode<T>>) -> Option<&T> {
        let timeslice = *leg.get_node().get_timeslice();
        let direction = leg.get_direction();
        let relative_index = leg.get_relative_index();
        self.timeslice_to_data
            .get(&timeslice)
            .map(|data| data.get(direction, relative_index))
    }

    fn get_input_state(&self, node: &'a DoublyLinkedNode<T>) -> Option<&[T]> {
        let timeslice = node.get_timeslice();
        self.timeslice_to_data
            .get(timeslice)
            .map(|data| data.get_input())
    }

    fn get_output_state(&self, node: &'a DoublyLinkedNode<T>) -> Option<&[T]> {
        let timeslice = node.get_timeslice();
        self.timeslice_to_data
            .get(timeslice)
            .map(|data| data.get_output())
    }

    fn incoming_leg(
        &mut self,
        leg: Leg<&'a DoublyLinkedNode<T>>,
        value: T,
    ) -> Option<(Leg<&'a DoublyLinkedNode<T>>, T)> {
        let timeslice = *leg.get_node().get_timeslice();
        let direction = leg.get_direction();
        let relative_index = leg.get_relative_index();
        let key = (timeslice, direction, relative_index);

        // Check if we are going to hit a leg
        if let Some(existing_leg_index) = self.timeslice_to_leg.get(&key) {
            let data = self
                .timeslice_to_data
                .get_mut(&timeslice)
                .expect("Data should not be empty given a leg exists");
            let existing_value = data.get(direction, relative_index);
            if value.eq(existing_value) {
                // We can annihilate this leg and resolve the cluster ends.
                self.remove_leg(*existing_leg_index);
                None
            } else {
                // If not equal, overrun the old leg and continue.
                // This is probably not a good sign.
                self.remove_leg(*existing_leg_index);
                Some((leg, value))
            }
        } else {
            Some((leg, value))
        }
    }

    fn set_initial_state_value(&mut self, absolute_index: &usize, value: T) {
        self.initial_state.insert(*absolute_index, value);
    }

    fn produce_change_record(self) -> Self::ChangeRecord {
        (self.initial_state, self.timeslice_to_data)
    }
}

impl<'a, T> GenericClusterManager<'a, T>
where
    T: DOFTypeTrait,
{
    fn remove_leg(&mut self, leg_index: usize) -> Option<Leg<&'a DoublyLinkedNode<T>>> {
        let leg_to_remove = &self.cluster_legs[leg_index];
        let timeslice_for_leg_to_remove = *leg_to_remove.get_node().get_timeslice();
        let direction_for_leg_to_remove = leg_to_remove.get_direction();
        let relative_index_for_leg_to_remove = leg_to_remove.get_relative_index();
        let key_to_remove = (
            timeslice_for_leg_to_remove,
            direction_for_leg_to_remove,
            relative_index_for_leg_to_remove,
        );

        let last_index = self.cluster_legs.len() - 1;
        if leg_index >= self.cluster_legs.len() {
            return None;
        } else if leg_index == last_index {
            // If it's at the end, just pop it. Great job.
        } else {
            // Otherwise swap with the end and then pop.
            let last_leg = self
                .cluster_legs
                .last()
                .expect("There must be a leg at the end");
            let timeslice = *last_leg.get_node().get_timeslice();
            let direction = last_leg.get_direction();
            let relative_index = last_leg.get_relative_index();
            let key = (timeslice, direction, relative_index);

            self.cluster_legs.swap(leg_index, last_index);
            let existing_leg_index = self
                .timeslice_to_leg
                .get_mut(&key)
                .expect("This should be pointing to the end of the leg list.");
            debug_assert_eq!(*existing_leg_index, last_index);
            *existing_leg_index = leg_index;
        }
        let removed = self.timeslice_to_leg.remove(&key_to_remove);
        debug_assert_eq!(removed, Some(leg_index));
        self.cluster_legs.pop()
    }
}

pub trait TermClusterExpander<DOF> {
    fn output_changes_for_spin_flip<'a, R>(
        &self,
        input_state: &[DOF],
        output_state: &[DOF],
        direction: DirectionEnum,
        relative_index: usize,
        new_value: &DOF,
        rng: &mut R,
    ) -> impl NodeClusterExpansion<DOF> + 'a
    where
        R: Rng;
}

#[cfg(test)]
mod cluster_tests {
    use super::*;
    use crate::qmc::naive_flip_impl::MatrixTermFlippable;
    use crate::traits::WeightChange;
    use crate::traits::graph_traits::{GraphStateNavigator, TimeSlicedGraph};

    struct EyePlusXMatrixTerm;

    impl MatrixTermData<f64> for EyePlusXMatrixTerm {
        fn get_matrix_entry(&self, _input: usize, _output: usize) -> f64 {
            1.0
        }
        fn dim(&self) -> usize {
            2
        }
        fn get_weight_change_for_diagonal(
            &self,
            _old_state: usize,
            _new_state: usize,
        ) -> Option<(f64, f64)> {
            None
        }
        fn get_number_of_equal_weight_outputs_for_input_distinct_from_output(
            &self,
            _input: usize,
            _output: usize,
        ) -> usize {
            1
        }

        fn get_natural_offset(&self) -> f64 {
            0.0
        }
    }
    impl MatrixTermFlippable<f64> for EyePlusXMatrixTerm {
        fn is_maybe_flippable(&self) -> bool {
            true
        }
        fn get_weights_for_inputs_given_output(
            &self,
            _input_a: usize,
            _input_b: usize,
            _output: usize,
        ) -> Option<(f64, f64)> {
            None
        }
        fn get_nth_equal_weight_output_for_input_distinct_from_output(
            &self,
            _input: usize,
            output: usize,
            _n: usize,
        ) -> usize {
            1 - output
        }
    }
    struct SimpleClusterExpander;
    impl NodeClusterExpansion<bool> for SimpleClusterExpander {
        fn get_weight_change(&self) -> WeightChange {
            WeightChange::NoChange
        }
        fn get_iterator(self) -> impl IntoIterator<Item = (DirectionEnum, usize, bool)> {
            None
        }
    }

    impl TermClusterExpander<bool> for EyePlusXMatrixTerm {
        fn output_changes_for_spin_flip<'a, R>(
            &self,
            _input_state: &[bool],
            _output_state: &[bool],
            _direction: DirectionEnum,
            _relative_index: usize,
            _new_value: &bool,
            _rng: &mut R,
        ) -> impl NodeClusterExpansion<bool> + 'a
        where
            R: Rng,
        {
            SimpleClusterExpander
        }
    }

    #[test]
    fn check_simple_cluster() -> Result<(), String> {
        let mut qmc = GenericQMC::new(1);
        let handle = qmc.add_term(EyePlusXMatrixTerm, vec![0]);
        qmc.add_node(0, handle);
        qmc.add_node(2, handle);

        let mut rng = rand::rng();
        qmc.cluster_update_starting_from_timeslice(&0, DirectionEnum::Output, 0, true, &mut rng)?;

        let first_node_value = qmc
            .get_node(&0)
            .map(|node| (node.input_state[0], node.output_state[0]));
        debug_assert_eq!(first_node_value, Some((false, true)));

        let second_node_value = qmc
            .get_node(&2)
            .map(|node| (node.input_state[0], node.output_state[0]));
        debug_assert_eq!(second_node_value, Some((true, false)));

        Ok(())
    }

    #[test]
    fn check_simple_cluster_wrap() -> Result<(), String> {
        let mut qmc = GenericQMC::new(1);
        let handle = qmc.add_term(EyePlusXMatrixTerm, vec![0]);
        qmc.add_node(0, handle);
        qmc.add_node(2, handle);

        let mut rng = rand::rng();
        qmc.cluster_update_starting_from_timeslice(&0, DirectionEnum::Input, 0, true, &mut rng)?;

        let first_node_value = qmc
            .get_node(&0)
            .map(|node| (node.input_state[0], node.output_state[0]));
        debug_assert_eq!(first_node_value, Some((true, false)));

        let second_node_value = qmc
            .get_node(&2)
            .map(|node| (node.input_state[0], node.output_state[0]));
        debug_assert_eq!(second_node_value, Some((false, true)));

        debug_assert!(qmc.get_initial_state()[0]);

        Ok(())
    }

    #[test]
    fn check_simple_cluster_wrap_single_op() -> Result<(), String> {
        let mut qmc = GenericQMC::new(1);
        let handle = qmc.add_term(EyePlusXMatrixTerm, vec![0]);
        qmc.add_node(0, handle);

        let mut rng = rand::rng();
        qmc.cluster_update_starting_from_timeslice(&0, DirectionEnum::Input, 0, true, &mut rng)?;

        let first_node_value = qmc
            .get_node(&0)
            .map(|node| (node.input_state[0], node.output_state[0]));
        debug_assert_eq!(first_node_value, Some((true, true)));

        debug_assert!(qmc.get_initial_state()[0]);

        Ok(())
    }

    struct EyeEyePlusXXMatrixTerm;

    impl MatrixTermData<f64> for EyeEyePlusXXMatrixTerm {
        fn get_matrix_entry(&self, input: usize, output: usize) -> f64 {
            match (input, output) {
                (i, o) if i == o => 1.0,
                (i, o) if i == !o & 0b11  => 1.0,
                _ => 0.0
            }
        }
        fn dim(&self) -> usize {
            4
        }

        fn get_number_of_equal_weight_outputs_for_input_distinct_from_output(
            &self,
            _input: usize,
            _output: usize,
        ) -> usize {
            1
        }

        fn get_natural_offset(&self) -> f64 {
            0.0
        }
    }
    impl MatrixTermFlippable<f64> for EyeEyePlusXXMatrixTerm {
        fn is_maybe_flippable(&self) -> bool {
            true
        }
        fn get_weights_for_inputs_given_output(
            &self,
            _input_a: usize,
            _input_b: usize,
            _output: usize,
        ) -> Option<(f64, f64)> {
            todo!()
        }
        fn get_nth_equal_weight_output_for_input_distinct_from_output(
            &self,
            input: usize,
            _output: usize,
            _n: usize,
        ) -> usize {
            match input {
                0b00 => 0b11,
                0b01 => 0b10,
                0b10 => 0b01,
                0b11 => 0b00,
                _ => unreachable!(),
            }
        }
    }

    impl TermClusterExpander<bool> for EyeEyePlusXXMatrixTerm {
        fn output_changes_for_spin_flip<'a, R>(
            &self,
            input_state: &[bool],
            output_state: &[bool],
            direction: DirectionEnum,
            relative_index: usize,
            new_value: &bool,
            _rng: &mut R,
        ) -> impl NodeClusterExpansion<bool> + 'a
        where
            R: Rng,
        {
            let existing_value = match direction {
                DirectionEnum::Input => &input_state[relative_index],
                DirectionEnum::Output => &output_state[relative_index],
            };
            debug_assert_ne!(new_value, existing_value);

            let new_index = 1 - relative_index;
            let new_direction = direction;
            let new_value = match new_direction {
                DirectionEnum::Input => !input_state[new_index],
                DirectionEnum::Output => !output_state[new_index],
            };
            Some((new_direction, new_index, new_value))
        }
    }

    #[test]
    fn check_simple_twobody_cluster() -> Result<(), String> {
        let mut qmc = GenericQMC::new(2);
        let handle = qmc.add_term(EyeEyePlusXXMatrixTerm, vec![0, 1]);
        qmc.add_node(0, handle);
        qmc.add_node(2, handle);

        let mut rng = rand::rng();
        qmc.cluster_update_starting_from_timeslice(&0, DirectionEnum::Output, 0, true, &mut rng)?;

        let node_value = qmc
            .get_node(&0)
            .map(|node| (&node.input_state, &node.output_state));
        debug_assert_eq!(node_value, Some((&vec![false, false], &vec![true, true])));

        let node_value = qmc
            .get_node(&2)
            .map(|node| (&node.input_state, &node.output_state));
        debug_assert_eq!(node_value, Some((&vec![true, true], &vec![false, false])));

        Ok(())
    }

    #[test]
    fn check_simple_twobody_cluster_wrap_single_op_worm() -> Result<(), String> {
        let mut qmc = GenericQMC::new(2);
        let handle = qmc.add_term(EyeEyePlusXXMatrixTerm, vec![0, 1]);
        qmc.add_node(0, handle);

        let mut rng = rand::rng();
        qmc.cluster_update_starting_from_timeslice(&0, DirectionEnum::Output, 0, true, &mut rng)?;

        let node_value = qmc
            .get_node(&0)
            .map(|node| (&node.input_state, &node.output_state));
        debug_assert_eq!(node_value, Some((&vec![true, true], &vec![true, true])));

        debug_assert_eq!(qmc.get_initial_state(), vec![true, true]);

        Ok(())
    }

    struct EyeEyePlusZZMatrixTerm;

    impl MatrixTermData<f64> for EyeEyePlusZZMatrixTerm {
        fn get_matrix_entry(&self, input: usize, output: usize) -> f64 {
            if input != output { return 0.0; }
            match input {
                0b00 | 0b11 => 2.0,
                0b01 | 0b10 => 0.0,
                _ => panic!("Should not be reachable.")
            }
        }
        fn dim(&self) -> usize {
            4
        }
        fn get_number_of_equal_weight_outputs_for_input_distinct_from_output(
            &self,
            _input: usize,
            _output: usize,
        ) -> usize {
            0
        }

        fn get_natural_offset(&self) -> f64 {
            0.0
        }
    }
    impl MatrixTermFlippable<f64> for EyeEyePlusZZMatrixTerm {
        fn is_maybe_flippable(&self) -> bool {
            false
        }
        fn get_weights_for_inputs_given_output(
            &self,
            _input_a: usize,
            _input_b: usize,
            _output: usize,
        ) -> Option<(f64, f64)> {
            unimplemented!()
        }
        fn get_nth_equal_weight_output_for_input_distinct_from_output(
            &self,
            _input: usize,
            _output: usize,
            _n: usize,
        ) -> usize {
            unimplemented!()
        }
    }
    impl TermClusterExpander<bool> for EyeEyePlusZZMatrixTerm {
        fn output_changes_for_spin_flip<'a, R>(
            &self,
            _input_state: &[bool],
            _output_state: &[bool],
            direction: DirectionEnum,
            relative_index: usize,
            new_value: &bool,
            _rng: &mut R,
        ) -> impl NodeClusterExpansion<bool> + 'a
        where
            R: Rng,
        {
            let other_direction = direction.swap_direction();
            let other_index = 1 - relative_index;
            [
                (direction, other_index, *new_value),
                (other_direction, relative_index, *new_value),
                (other_direction, other_index, *new_value),
            ]
        }
    }

    #[test]
    fn check_simple_twobody_cluster_wrap_single_op_ising_cluster() -> Result<(), String> {
        let mut qmc = GenericQMC::new(2);
        let handle = qmc.add_term(EyeEyePlusZZMatrixTerm, vec![0, 1]);
        qmc.add_node(0, handle);

        let mut rng = rand::rng();
        qmc.cluster_update_starting_from_timeslice(&0, DirectionEnum::Output, 0, true, &mut rng)?;

        let node_value = qmc
            .get_node(&0)
            .map(|node| (&node.input_state, &node.output_state));
        debug_assert_eq!(node_value, Some((&vec![true, true], &vec![true, true])));

        debug_assert_eq!(qmc.get_initial_state(), vec![true, true]);

        Ok(())
    }

    #[test]
    fn check_staggered_twobody_cluster() -> Result<(), String> {
        let mut qmc = GenericQMC::new(3);
        let handle_a = qmc.add_term(EyeEyePlusZZMatrixTerm, vec![0, 1]);
        let handle_b = qmc.add_term(EyeEyePlusZZMatrixTerm, vec![1, 2]);
        qmc.add_node(0, handle_a);
        qmc.add_node(2, handle_b);

        let mut rng = rand::rng();
        qmc.cluster_update_starting_from_timeslice(&0, DirectionEnum::Output, 0, true, &mut rng)?;

        let node_value = qmc
            .get_node(&0)
            .map(|node| (&node.input_state, &node.output_state));
        debug_assert_eq!(node_value, Some((&vec![true, true], &vec![true, true])));

        let node_value = qmc
            .get_node(&2)
            .map(|node| (&node.input_state, &node.output_state));
        debug_assert_eq!(node_value, Some((&vec![true, true], &vec![true, true])));

        debug_assert_eq!(qmc.get_initial_state(), vec![true, true, true]);

        Ok(())
    }
}

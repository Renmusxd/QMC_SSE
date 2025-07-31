use QmcSSE::qmc::MatrixTermData;
use num_traits::{One, Zero};

#[derive(Clone)]
pub struct RingExchangeData<T>
where
    T: Clone,
{
    scale: T,
    exchangeable_state_a: usize,
    exchangeable_state_b: usize,
    dim: usize,
}

impl<T: Clone> RingExchangeData<T> {
    pub fn new(
        scale: T,
        exchangeable_state_a: usize,
        exchangeable_state_b: usize,
        dim: usize,
    ) -> Self {
        Self {
            scale,
            exchangeable_state_a,
            exchangeable_state_b,
            dim,
        }
    }
}

impl<T> MatrixTermData<T> for RingExchangeData<T>
where
    T: Zero + One + Clone + PartialEq,
{
    fn get_matrix_entry(&self, input: usize, output: usize) -> T {
        if input == output {
            self.scale.clone()
        } else if input == self.exchangeable_state_a && output == self.exchangeable_state_b {
            self.scale.clone()
        } else if output == self.exchangeable_state_a && input == self.exchangeable_state_b {
            self.scale.clone()
        } else {
            T::zero()
        }
    }

    fn is_maybe_flippable(&self) -> bool {
        true
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn get_weight_change_for_diagonal(
        &self,
        _old_state: usize,
        _new_state: usize,
    ) -> Option<(T, T)> {
        None
    }

    fn get_weight_change_for_inputs_given_output(
        &self,
        input_a: usize,
        input_b: usize,
        output: usize,
    ) -> Option<(T, T)> {
        let wa = self.get_matrix_entry(input_a, output);
        let wb = self.get_matrix_entry(input_b, output);
        if wa == wb { None } else { Some((wa, wb)) }
    }

    fn get_number_of_equal_weight_outputs_for_input_distinct_from_output(
        &self,
        input: usize,
        output: usize,
    ) -> usize {
        if input == self.exchangeable_state_a || input == self.exchangeable_state_b {
            1
        } else {
            0
        }
    }

    fn get_nth_equal_weight_output_for_input_distinct_from_output(
        &self,
        input: usize,
        output: usize,
        n: usize,
    ) -> usize {
        debug_assert_eq!(n, 0);
        debug_assert!(input == self.exchangeable_state_a || input == self.exchangeable_state_b);
        if output == self.exchangeable_state_a {
            self.exchangeable_state_b
        } else {
            self.exchangeable_state_a
        }
    }
}

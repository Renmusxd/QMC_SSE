use num_traits::{One, Zero};
use rand::prelude::*;

use QmcSSE::qmc::naive_flip_impl::MatrixTermFlippable;
use QmcSSE::qmc::{GenericQMC, MatrixTermData};
use QmcSSE::traits::diagonal_update::DiagonalUpdate;
use QmcSSE::traits::graph_traits::GraphStateNavigator;
use QmcSSE::traits::naive_flip_update::NaiveFlipUpdater;

fn main() {
    let l = 3;

    let mut qmc = GenericQMC::<bool, RingExchangeData<f64>>::new_with_state(vec![false; l]);

    let term = RingExchangeData::<f64> {
        scale: 1.0,
        exchangeable_state_a: 0b00,
        exchangeable_state_b: 0b11,
        dim: 4,
    };

    for i in 0..l - 1 {
        qmc.add_term(term.clone(), vec![i, i + 1]);
    }

    let beta = 16.0;
    let mut rng = SmallRng::seed_from_u64(12345);

    let thermalization_steps = 1024;
    for _ in 0..thermalization_steps {
        qmc.maintain_maximum_filling_fraction(0.75, 16);
        qmc.diagonal_update(beta, &mut rng);
        for _ in 0..l {
            qmc.naive_flip_update(&mut rng);
        }
    }

    let mut energies = vec![];
    let samples = 128;
    let autocorr_time = 4;
    for _ in 0..samples {
        for _ in 0..autocorr_time {
            qmc.maintain_maximum_filling_fraction(0.75, 16);
            qmc.diagonal_update(beta, &mut rng);
            for _ in 0..l {
                qmc.naive_flip_update(&mut rng);
            }
        }
        energies.push(qmc.get_energy(beta));
    }
    let avg_energy = energies.iter().sum::<f64>() / (samples as f64);
    let variance = energies.iter().map(|x| x.powi(2)).sum::<f64>() / (samples as f64);
    println!(
        "Avg: {:.3} +/- {:.3}",
        avg_energy,
        variance.sqrt() / (samples as f64).sqrt()
    );
}

#[derive(Clone)]
struct RingExchangeData<T>
where
    T: Clone,
{
    scale: T,
    exchangeable_state_a: usize,
    exchangeable_state_b: usize,
    dim: usize,
}

impl<T> MatrixTermFlippable<T> for RingExchangeData<T>
where
    T: Zero + One + Clone + PartialEq,
{
    fn is_maybe_flippable(&self) -> bool {
        true
    }

    fn get_weights_for_inputs_given_output(
        &self,
        input_a: usize,
        input_b: usize,
        output: usize,
    ) -> Option<(T, T)> {
        let wa = self.get_matrix_entry(input_a, output);
        let wb = self.get_matrix_entry(input_b, output);
        if wa == wb { None } else { Some((wa, wb)) }
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
}

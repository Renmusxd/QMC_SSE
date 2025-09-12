use rand::prelude::*;

use qmc_sse::qmc::{GenericQMC};
use qmc_sse::terms::generic::GenericMatrixTermEnum;
use qmc_sse::traits::diagonal_update::DiagonalUpdate;
use qmc_sse::traits::naive_flip_update::NaiveFlipUpdater;

fn main() {
    let n = 3;

    let mut qmc = GenericQMC::<bool, _>::new(n);

    for i in 0..n {
        qmc.add_term(
            GenericMatrixTermEnum::make_diagonal(vec![1.0, 0.0, 0.0, 1.0]),
            vec![i, (i + 1) % n],
        );
    }
    for i in 0..n {
        qmc.add_term(GenericMatrixTermEnum::make_uniform(1.0, 2), vec![i]);
    }

    let beta = 16.0;
    let mut rng = SmallRng::seed_from_u64(12345);

    let thermalization_steps = 128;
    for _ in 0..thermalization_steps {
        qmc.maintain_maximum_filling_fraction(0.5, 16);
        qmc.diagonal_update(beta, &mut rng);
        qmc.naive_flip_update(&mut rng);
    }
    qmc.print_worldlines();

    let mut num_operators = vec![];
    let samples = 1024;
    let autocorr_time = 16;
    for _ in 0..samples {
        for _ in 0..autocorr_time {
            qmc.maintain_maximum_filling_fraction(0.5, 16);
            qmc.diagonal_update(beta, &mut rng);
            qmc.naive_flip_update(&mut rng);
        }
        debug_assert!(qmc.check_consistency());
        let n_nonzero = qmc.get_number_of_non_identity_operators();
        num_operators.push(n_nonzero);
    }
    qmc.print_worldlines();

    let energies = num_operators
        .into_iter()
        .map(|x| x as f64 / beta)
        .collect::<Vec<_>>();
    let avg_energy = energies.iter().sum::<f64>() / (samples as f64);
    let variance = energies.iter().map(|x| x.powi(2)).sum::<f64>() / (samples as f64);
    println!(
        "Avg: {:.3} +/- {:.3}",
        avg_energy,
        variance.sqrt() / (samples as f64).sqrt()
    );
}

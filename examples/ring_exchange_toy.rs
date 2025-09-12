use rand::prelude::*;

use qmc_sse::qmc::GenericQMC;
use qmc_sse::terms::ring_exchange::RingExchangeData;
use qmc_sse::traits::diagonal_update::DiagonalUpdate;
use qmc_sse::traits::naive_flip_update::NaiveFlipUpdater;

fn main() {
    let l = 3;

    let mut qmc = GenericQMC::new_with_state(vec![false; l]);
    let term = RingExchangeData::new(
         1.0,
         0b00,
         0b11,
         4,
    );

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
pub mod pyrochlore_helper;

use crate::pyrochlore_helper::PyrochloreLatticeHelper;
use qmc_sse::qmc::GenericQMC;
use qmc_sse::traits::diagonal_update::DiagonalUpdate;
use qmc_sse::traits::naive_flip_update::NaiveFlipUpdater;
use rand::SeedableRng;
use rand::prelude::SmallRng;
use qmc_sse::terms::ring_exchange::RingExchangeData;

fn main() {
    let l = 4;

    let pyrochlore_helper = PyrochloreLatticeHelper::new(l, l, l);

    let initial_state = pyrochlore_helper.get_initial_state();
    let v = initial_state.len();

    let mut qmc = GenericQMC::<bool, RingExchangeData<f64>>::new_with_state(initial_state);

    for (term, indices) in pyrochlore_helper.get_terms() {
        qmc.add_term(term, indices);
    }

    let mut rng = SmallRng::seed_from_u64(12345);
    let beta = 16.0;

    // Diagonal therm
    let diagonal_therm = 128;
    for _ in 0..diagonal_therm {
        qmc.maintain_maximum_filling_fraction(0.75, 16);
        qmc.diagonal_update(beta, &mut rng);
    }

    // Thermalization with off-diagonals.
    let full_therm = 128;
    for _ in 0..full_therm {
        qmc.maintain_maximum_filling_fraction(0.75, 16);
        qmc.diagonal_update(beta, &mut rng);
        for _ in 0..v {
            qmc.naive_flip_update(&mut rng);
        }
    }

    // Measure autocorrelation
    let autocorr_data = qmc.autocorr_for_terms(beta, 128, |qmc| {
        qmc.maintain_maximum_filling_fraction(0.75, 16);
        qmc.diagonal_update(beta, &mut rng);
        for _ in 0..v {
            qmc.naive_flip_update(&mut rng);
        }
    });

    println!(
        "{:?}",
        autocorr_data
            .into_iter()
            .map(|x| format!("{:.3}", x))
            .collect::<Vec<_>>()
            .join(", ")
    );
}

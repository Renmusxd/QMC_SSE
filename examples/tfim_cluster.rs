use rand::prelude::*;
use std::env::var;

use QmcSSE::qmc::{GenericMatrixTerm, GenericMatrixTermEnum, GenericQMC};
use QmcSSE::traits::diagonal_update::DiagonalUpdate;
use QmcSSE::traits::naive_flip_update::NaiveFlipUpdater;

fn main() {
    let n = 3;

    let mut qmc = GenericQMC::<bool>::new(n);

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
    for i in 0..thermalization_steps {
        qmc.maintain_maximum_filling_fraction(0.5, 16);
        qmc.diagonal_update(beta, &mut rng);
    }
    qmc.print_worldlines();
}

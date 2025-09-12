use rand::prelude::*;

use qmc_sse::qmc::GenericQMC;
use qmc_sse::terms::tfim::TFIMTerm;
use qmc_sse::traits::diagonal_update::DiagonalUpdate;

fn main() {
    let n = 5;
    let bond_j = 1.0;
    let gamma = 1.0;

    let mut qmc = GenericQMC::<bool,_>::new(n);

    for i in 0..n {
        qmc.add_term(TFIMTerm::X(gamma),vec![i],);
        qmc.add_term(TFIMTerm::ZZ(bond_j), vec![i, (i + 1) % n]);
    }

    let beta = 16.0;
    let mut rng = SmallRng::seed_from_u64(12345);

    let thermalization_steps = 128;
    for _ in 0..thermalization_steps {
        qmc.maintain_maximum_filling_fraction(0.5, 16);
        qmc.diagonal_update(beta, &mut rng);
    }
    qmc.print_worldlines();
}

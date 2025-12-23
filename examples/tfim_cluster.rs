use rand::prelude::*;

use qmc::qmc::GenericQMC;
use qmc::terms::tfim::TFIMTerm;
use qmc::traits::cluster_update::ClusterUpdater;
use qmc::traits::diagonal_update::DiagonalUpdate;
use qmc::traits::thermal_update::ThermalUpdate;



fn main() -> Result<(), String> {
    env_logger::init();
    let n = 5;
    let bond_j = 0.5;
    let gamma = 1.0;

    let mut qmc = GenericQMC::<bool, _>::new(n);

    for i in 0..n {
        qmc.add_term(TFIMTerm::X(gamma), [i]);
        qmc.add_term(TFIMTerm::ZZ(-bond_j), [i, (i + 1) % n]);
    }

    let beta = 4.0;
    let mut rng = SmallRng::from_os_rng(); // SmallRng::seed_from_u64(12345);

    let thermalization_steps = 1024;
    for _ in 0..thermalization_steps {
        qmc.thermal_update(&mut rng);
        qmc.diagonal_update(beta, &mut rng);
        qmc.check_consistency();
    }

    let mut energies = vec![];
    let samples = 128;
    let autocorr_time = 32;
    for _ in 0..samples {
        for _ in 0..autocorr_time {
            qmc.diagonal_update(beta, &mut rng);
            qmc.cluster_update(&mut rng)?;
            qmc.check_consistency();
        }
        energies.push(qmc.get_energy(beta));
    }
    let avg_energy = energies.iter().sum::<f64>() / (samples as f64);
    let variance = energies.iter().map(|x| (x - avg_energy).powi(2)).sum::<f64>() / (samples as f64);
    println!(
        "Avg: {:.3} +/- {:.3}",
        avg_energy,
        variance.sqrt() / (samples as f64).sqrt()
    );

    Ok(())
}

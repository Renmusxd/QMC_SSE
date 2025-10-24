use rand::prelude::*;

use qmc_sse::qmc::GenericQMC;
use qmc_sse::terms::generic::GenericMatrixTermEnum;
use qmc_sse::terms::tfim::TFIMTerm;
use qmc_sse::traits::diagonal_update::DiagonalUpdate;
use qmc_sse::traits::graph_traits::TimeSlicedGraph;
use qmc_sse::traits::naive_flip_update::NaiveFlipUpdater;
use qmc_sse::traits::term_rotation_cluster_update::MatrixTermRotationUpdate;

fn main() {
    let n = 3;

    let mut qmc = GenericQMC::<bool, _>::new(n);

    let mut x_handles = vec![];
    let mut zz_handles = vec![];
    for i in 0..n {
        let hx = qmc.add_term(TFIMTerm::X(1.0), vec![i]);
        let hzz = qmc.add_term(TFIMTerm::ZZ(1.0), vec![i, (i + 1) % n]);
        x_handles.push(hx);
        zz_handles.push(hzz);
    }

    let beta = 16.0;
    let mut rng = SmallRng::seed_from_u64(12345);

    qmc.add_node(0, x_handles[0]);
    qmc.add_node(1, zz_handles[0]);
    qmc.add_node(2, x_handles[0]);

    qmc.print_worldlines();

    qmc.perform_cluster_update_on_index_at_slice(&0, &0, &mut rng);

    qmc.print_worldlines();
}

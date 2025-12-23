use crate::qmc::{GenericQMC, MatrixTermData};
use crate::traits::graph_traits::DOFTypeTrait;
use crate::traits::thermal_update::ThermalUpdate;

impl<DOF: DOFTypeTrait, Data: MatrixTermData<f64>, GC> ThermalUpdate for GenericQMC<DOF, Data, GC>{
    fn apply_function_to_empty_worldlines<T, F>(&mut self, mut context: T, f: F)
    where
        F: Fn(&mut T, &mut Self::DOFType)
    {
        self.first_nodes_for_dofs.iter().zip(self.initial_state.iter_mut()).filter(|(first_node, _)| {
            first_node.is_none()
        }).for_each(|(_, value)| {
            f(&mut context, value)
        });
    }
}
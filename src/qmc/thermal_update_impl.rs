use crate::qmc::{GenericQMC, MatrixTermData};
use crate::traits::graph_traits::DOFTypeTrait;
use crate::traits::thermal_update::ThermalUpdate;

impl<DOF: DOFTypeTrait, Data: MatrixTermData<f64>, GC> ThermalUpdate for GenericQMC<DOF, Data, GC>{
    fn apply_function_to_empty_worldlines<T, F>(&mut self, mut context: T, f: F)
    where
        F: Fn(&mut T, &mut Self::DOFType)
    {
        self.initial_state.iter_mut().enumerate().filter(|(index, _)| {
            self.first_nodes_for_dofs[*index].is_none()
        }).for_each(|(_, value)| {
            f(&mut context, value)
        });
    }
}
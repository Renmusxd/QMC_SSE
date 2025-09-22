use std::ops::MulAssign;

pub mod cluster_update;
pub mod diagonal_update;
pub mod dimer_worm_update;
pub mod graph_traits;
pub mod graph_weights;
pub mod naive_flip_update;
pub mod spin_systems;

#[derive(Clone, Copy, Debug)]
pub enum WeightChange {
    NoChange,
    ZeroWeight,
    Factor(f64),
}

impl WeightChange {
    pub fn get_factor(&mut self) -> Option<&f64> {
        match self {
            WeightChange::NoChange | WeightChange::ZeroWeight => None,
            WeightChange::Factor(x) => Some(x),
        }
    }

    pub fn get_factor_mut(&mut self) -> Option<&mut f64> {
        match self {
            WeightChange::NoChange | WeightChange::ZeroWeight => None,
            WeightChange::Factor(x) => Some(x),
        }
    }

    pub fn zero_weight(&self) -> bool {
        matches!(self, WeightChange::ZeroWeight | WeightChange::Factor(0.0))
    }
}

impl MulAssign<WeightChange> for WeightChange {
    fn mul_assign(&mut self, rhs: WeightChange) {
        match (self, rhs) {
            (_, WeightChange::NoChange) => {}
            (WeightChange::ZeroWeight, _) => {}
            (x, WeightChange::ZeroWeight) => *x = WeightChange::ZeroWeight,
            (x @ WeightChange::NoChange, WeightChange::Factor(f)) => {
                *x = WeightChange::Factor(f);
            }
            (WeightChange::Factor(x), WeightChange::Factor(y)) => *x *= y,
        }
    }
}

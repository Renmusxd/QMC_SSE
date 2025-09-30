use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Mul, MulAssign};

pub mod cluster_update;
pub mod diagonal_update;
pub mod term_rotation_cluster_update;
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
    pub fn get_factor(&self) -> Option<&f64> {
        match self {
            WeightChange::NoChange | WeightChange::ZeroWeight => None,
            WeightChange::Factor(x) => Some(x),
        }
    }

    pub fn get_weight(&self) -> Option<f64> {
        match self {
            WeightChange::ZeroWeight => None,
            WeightChange::NoChange => Some(1.0),
            WeightChange::Factor(x) => Some(*x)
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

impl Mul for WeightChange {
    type Output = WeightChange;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= rhs;
        self
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

impl Add for WeightChange {
    type Output = WeightChange;
    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}


impl AddAssign for WeightChange {
    fn add_assign(&mut self, rhs: Self) {
        match (self, rhs) {
            (_, WeightChange::ZeroWeight) => {}
            (x @ WeightChange::ZeroWeight, y) => {*x = y;}
            (x , y) => {
                let fx = x.get_factor().unwrap_or(&0.0);
                let fy = y.get_factor().unwrap_or(&0.0);
                *x = WeightChange::Factor(fx + fy);
            }
        }
    }
}


impl Product for WeightChange {
    fn product<I: Iterator<Item=Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a * b).unwrap_or(WeightChange::NoChange)
    }
}

impl Sum for WeightChange {
    fn sum<I: Iterator<Item=Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a + b).unwrap_or(WeightChange::ZeroWeight)
    }
}
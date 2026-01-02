use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Mul, MulAssign};

/// Cluster updates, including loop updates.
pub mod cluster_update;
/// Diagonal updates, adding and removing operators.
pub mod diagonal_update;
/// Navigating the graph, moving between nodes.
pub mod graph_traits;
/// Assigning weights to nodes.
pub mod graph_weights;
/// Simple offdiagonal updates between pairs of the operators.
pub mod naive_flip_update;
/// Degrees of freedom at each site implementing associated traits.
pub mod spin_systems;
/// A simple thermal update on wordlines without any operators.
pub mod thermal_update;

/// A weight change using 64-bit floats, with special enum variants for 0 or 1.0.
#[derive(Clone, Copy, Debug)]
pub enum WeightChange {
    /// No weight change, or a factor of 1.0.
    NoChange,
    /// Zero weight, or a factor of 0.0.
    ZeroWeight,
    /// A general factor.
    Factor(f64),
}

impl WeightChange {
    /// Get the factor, if no change or zero return None.
    pub fn get_factor(&self) -> Option<&f64> {
        match self {
            WeightChange::NoChange | WeightChange::ZeroWeight => None,
            WeightChange::Factor(x) => Some(x),
        }
    }

    /// Get the weight change, if result is zero then return None.
    pub fn get_weight(&self) -> Option<f64> {
        match self {
            WeightChange::ZeroWeight => None,
            WeightChange::NoChange => Some(1.0),
            WeightChange::Factor(x) => Some(*x),
        }
    }

    /// Get a mutable reference to the factor if not `NoChange` or `ZeroWeight`.
    pub fn get_factor_mut(&mut self) -> Option<&mut f64> {
        match self {
            WeightChange::NoChange | WeightChange::ZeroWeight => None,
            WeightChange::Factor(x) => Some(x),
        }
    }

    /// Return true if the weight is zero exactly
    /// Possible complications if floating point errors occur.
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
            (x @ WeightChange::ZeroWeight, y) => {
                *x = y;
            }
            (x, y) => {
                let fx = x.get_factor().unwrap_or(&0.0);
                let fy = y.get_factor().unwrap_or(&0.0);
                *x = WeightChange::Factor(fx + fy);
            }
        }
    }
}

impl Product for WeightChange {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a * b).unwrap_or(WeightChange::NoChange)
    }
}

impl Sum for WeightChange {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a + b)
            .unwrap_or(WeightChange::ZeroWeight)
    }
}

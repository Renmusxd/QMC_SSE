use std::iter::Sum;
use std::ops::{Add, AddAssign};
use rand::Rng;


/// An updater which resums out the diagonal operators, flipping offdiagonal single site operators.
pub trait ResummedFlipUpdater {
    /// An index for a flippable degree of freedom, such as the region between two timeslices.
    type FlippableLocation;

    /// Perform a single update step.
    fn resummed_flip_update<R: Rng>(&mut self, rng: &mut R) {
        if let Some(FlipWeight { weight_for_flip, weight_for_remain }) = self.get_total_flip_weight() {
            let total_weight = weight_for_flip + weight_for_remain;
            let weight_choice = rng.random::<f64>() * total_weight;
            if weight_choice <= weight_for_flip {
                let flip_loc = self.get_flip_location_by_weight(weight_choice).expect("Must have submitted a flip location less than total weight.");
                self.flip_location(flip_loc)
            }
        }
    }

    /// Flip the location indicated by `loc`.
    fn flip_location(&mut self, loc: Self::FlippableLocation);

    /// Get a flip location from a randomly generated weight.
    fn get_flip_location_by_weight(&self, mut weight: f64) -> Result<Self::FlippableLocation, String> {
        let res = self.get_flip_weights().into_iter().try_for_each(|(loc, FlipWeight { weight_for_flip, ..})| {
            weight -= weight_for_flip;
            if weight <= 0.0 {
                Err(loc)
            } else {
                Ok(())
            }
        });

        match res {
            Err(loc) => Ok(loc),
            Ok(()) => Err("Asked for weight larger than sum of flip weights".to_string())
        }
    }

    /// Get the number of flippable locations.
    fn get_n_flip_locations(&self) -> usize;

    /// Get the total weight associated with all possible flips.
    fn get_total_flip_weight(&self) -> Option<FlipWeight> {
        if self.get_n_flip_locations() == 0 {
            None
        } else {
            let iter = self.get_flip_weights().into_iter();
            Some(iter.map(|(_,x)| x).sum())
        }
    }

    /// Get the weight associated with each flip, and the associated handle for the flip location.
    fn get_flip_weights(&self) -> impl IntoIterator<Item=(Self::FlippableLocation, FlipWeight)>;
}

/// The weight of performing vs not performing a flip.
#[derive(Copy, Clone, PartialEq)]
pub struct FlipWeight {
    weight_for_flip: f64,
    weight_for_remain: f64
}

impl Add for FlipWeight {
    type Output = FlipWeight;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            weight_for_flip: self.weight_for_flip + rhs.weight_for_flip,
            weight_for_remain: self.weight_for_remain + rhs.weight_for_remain,
        }
    }
}

impl AddAssign for FlipWeight {
    fn add_assign(&mut self, rhs: Self) {
        self.weight_for_flip += rhs.weight_for_flip;
        self.weight_for_remain += rhs.weight_for_remain;
    }
}

impl Sum for FlipWeight {
    fn sum<I: Iterator<Item=Self>>(iter: I) -> Self {
        let mut acc = Self {weight_for_flip: 0.0, weight_for_remain: 0.0};
        iter.for_each(|Self {weight_for_flip, weight_for_remain}| {
            acc.weight_for_flip += weight_for_flip;
            acc.weight_for_remain += weight_for_remain;
        });
        acc
    }
}
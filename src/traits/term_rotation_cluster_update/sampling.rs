use std::iter::Sum;
use std::ops::SubAssign;

use crate::utils::logwrapper::LogWrapper;
use num_traits::{Float, Zero};
use rand::Rng;
use rand::distr::Uniform;
use rand::distr::uniform::SampleUniform;

pub trait SampleFraction<Rhs = Self> {
    fn sample<R: Rng>(self, normalization: Rhs, rng: &mut R) -> bool;
}

pub trait SampleFromWeights: Sized {
    fn sample_from_weights<R: Rng>(weights: &[Self], rng: &mut R) -> usize;
}

impl<T> SampleFraction for T
where
    T: SampleUniform + Zero + PartialOrd,
{
    fn sample<R: Rng>(self, normalization: Self, rng: &mut R) -> bool {
        rng.sample(
            Uniform::new(Self::zero(), normalization)
                .expect("Normalization must not be zero or negative."),
        ) <= self
    }
}

impl<'a, T> SampleFraction<&'a T> for T
where
    T: Copy + SampleFraction,
{
    fn sample<R: Rng>(self, normalization: &'a Self, rng: &mut R) -> bool {
        self.sample(*normalization, rng)
    }
}

impl<P> SampleFraction for LogWrapper<P>
where
    P: SampleFraction + Float,
{
    fn sample<R: Rng>(self, normalization: Self, rng: &mut R) -> bool {
        let s = self.dissolve();
        let n = normalization.dissolve();
        s.sample(n, rng)
    }
}

impl<P> SampleFraction for &LogWrapper<P>
where
    P: SampleFraction + Float,
{
    fn sample<R: Rng>(self, normalization: Self, rng: &mut R) -> bool {
        let s = self.dissolve();
        let n = normalization.dissolve();
        s.sample(n, rng)
    }
}

impl<T> SampleFromWeights for T
where
    T: SampleUniform + Zero + PartialOrd,
    for<'a> T: SubAssign<&'a T> + Sum<&'a T>,
{
    fn sample_from_weights<R: Rng>(weights: &[Self], rng: &mut R) -> usize {
        let total_weight = weights.iter().sum::<Self>();
        let mut choice = rng.sample(
            Uniform::new(Self::zero(), total_weight).expect("Total weight cannot be zero."),
        );
        for (i, w) in weights.iter().enumerate() {
            choice -= w;
            if choice <= Self::zero() {
                return i;
            }
        }
        unreachable!("Cannot reach each of loop without choosing an index.");
    }
}

impl<P> SampleFromWeights for LogWrapper<P>
where
    P: Float + SampleUniform + SubAssign + Sum,
{
    fn sample_from_weights<R: Rng>(weights: &[Self], rng: &mut R) -> usize {
        // Could do Gimbel sampling trick if desired.
        let total_weight = weights.iter().map(|x| x.dissolve()).sum::<P>();
        let mut choice = rng
            .sample(Uniform::new(P::zero(), total_weight).expect("Total weight cannot be zero."));
        for (i, w) in weights.iter().enumerate() {
            choice -= w.dissolve();
            if choice <= P::zero() {
                return i;
            }
        }
        unreachable!("Cannot reach end of loop without choosing an index.");
    }
}

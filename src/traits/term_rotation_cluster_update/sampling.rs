use crate::utils::logwrapper::LogWrapper;
use num_traits::Float;
use rand::Rng;
use rand::distr::Uniform;

pub trait SampleFraction<Rhs = Self> {
    fn sample<R: Rng>(self, normalization: Rhs, rng: R) -> bool;
}

impl SampleFraction for f32 {
    fn sample<R: Rng>(self, normalization: Self, mut r: R) -> bool {
        if self >= normalization {
            panic!("Normalization must not be smaller than weight.")
        }
        r.sample(
            Uniform::new(0.0, normalization).expect("Normalization must not be zero or negative."),
        ) <= self
    }
}

impl SampleFraction for &f32 {
    fn sample<R: Rng>(self, normalization: Self, r: R) -> bool {
        f32::sample(*self, *normalization, r)
    }
}

impl SampleFraction for f64 {
    fn sample<R: Rng>(self, normalization: Self, mut r: R) -> bool {
        if self >= normalization {
            panic!("Normalization must not be smaller than weight.")
        }
        r.sample(
            Uniform::new(0.0, normalization).expect("Normalization must not be zero or negative."),
        ) <= self
    }
}

impl SampleFraction for &f64 {
    fn sample<R: Rng>(self, normalization: Self, r: R) -> bool {
        f64::sample(*self, *normalization, r)
    }
}

impl SampleFraction for u32 {
    fn sample<R: Rng>(self, normalization: Self, mut r: R) -> bool {
        if self >= normalization {
            panic!("Normalization must not be smaller than weight.")
        }
        r.sample(
            Uniform::new(0, normalization).expect("Normalization must not be zero or negative."),
        ) <= self
    }
}

impl SampleFraction for &u32 {
    fn sample<R: Rng>(self, normalization: Self, r: R) -> bool {
        u32::sample(*self, *normalization, r)
    }
}

impl SampleFraction for u64 {
    fn sample<R: Rng>(self, normalization: Self, mut r: R) -> bool {
        if self >= normalization {
            panic!("Normalization must not be smaller than weight.")
        }
        r.sample(
            Uniform::new(0, normalization).expect("Normalization must not be zero or negative."),
        ) <= self
    }
}

impl SampleFraction for &u64 {
    fn sample<R: Rng>(self, normalization: Self, r: R) -> bool {
        u64::sample(*self, *normalization, r)
    }
}

impl SampleFraction for usize {
    fn sample<R: Rng>(self, normalization: Self, mut r: R) -> bool {
        if self > normalization {
            panic!("Normalization must not be smaller than weight.")
        }
        r.sample(
            Uniform::new(0, normalization).expect("Normalization must not be zero or negative."),
        ) <= self
    }
}

impl SampleFraction for &usize {
    fn sample<R: Rng>(self, normalization: Self, r: R) -> bool {
        usize::sample(*self, *normalization, r)
    }
}

impl<P> SampleFraction for LogWrapper<P>
where
    P: SampleFraction + Float,
{
    fn sample<R: Rng>(self, normalization: Self, r: R) -> bool {
        let s = self.dissolve();
        let n = normalization.dissolve();
        s.sample(n, r)
    }
}

impl<P> SampleFraction for &LogWrapper<P>
where
    P: SampleFraction + Float,
{
    fn sample<R: Rng>(self, normalization: Self, r: R) -> bool {
        let s = self.dissolve();
        let n = normalization.dissolve();
        s.sample(n, r)
    }
}

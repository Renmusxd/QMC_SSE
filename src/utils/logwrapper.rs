use num_traits::{Float, One, Zero};
use std::{
    iter::{Product, Sum},
    ops::{Add, Mul},
};

#[derive(Clone, PartialEq, PartialOrd, Debug, Copy)]
pub struct LogWrapper<P> {
    logit: P,
}

impl<P> LogWrapper<P>
where
    P: Float,
{
    pub fn new(p: P) -> Self {
        Self { logit: p.ln() }
    }

    pub fn dissolve(self) -> P {
        self.logit.exp()
    }

    pub fn ln_raw(self) -> P {
        self.logit
    }
}

impl<P> From<P> for LogWrapper<P>
where
    P: Float,
{
    fn from(p: P) -> Self {
        Self::new(p)
    }
}

impl<P> Sum for LogWrapper<P>
where
    P: Product + Float,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a + b).unwrap_or_else(Self::zero)
    }
}

impl<P> One for LogWrapper<P>
where
    P: Zero,
{
    fn one() -> Self {
        Self { logit: P::zero() }
    }
}

impl<P> Zero for LogWrapper<P>
where
    P: Float,
{
    fn zero() -> Self {
        Self {
            logit: P::neg_infinity(),
        }
    }

    fn is_zero(&self) -> bool {
        let l: &P = &self.logit;
        l.is_infinite() && l.is_sign_negative()
    }
}

impl<P> Mul for LogWrapper<P>
where
    P: Add<P, Output = P>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            logit: self.logit.add(rhs.logit),
        }
    }
}

impl<'a, 'b, P> Mul<&'b LogWrapper<P>> for &'a LogWrapper<P>
where
    for<'c> &'a P: Add<&'c P, Output = P>,
{
    type Output = LogWrapper<P>;

    fn mul(self, rhs: &'b LogWrapper<P>) -> Self::Output {
        Self::Output {
            logit: (&self.logit).add(&rhs.logit),
        }
    }
}

impl<P> Add for LogWrapper<P>
where
    P: Mul<P, Output = P> + Float,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let max_p = self.logit.max(rhs.logit);
        let new_logit = max_p + ((self.logit - max_p).exp() + (rhs.logit - max_p).exp()).ln();
        Self { logit: new_logit }
    }
}

impl<'a, P> Add<&'a LogWrapper<P>> for LogWrapper<P>
where
    P: Mul<&'a P, Output = P> + Float,
{
    type Output = Self;
    fn add(self, rhs: &'a Self) -> Self {
        let max_p = self.logit.max(rhs.logit);
        let new_logit = max_p + ((self.logit - max_p).exp() + (rhs.logit - max_p).exp()).ln();
        Self { logit: new_logit }
    }
}

#[cfg(test)]
mod cluster_flip_tests {
    use super::*;

    #[test]
    fn test_logit_math_mult() {
        let x = 1.2345;
        let y = 1.1234;
        let xy = x * y;
        let lx = LogWrapper::new(x);
        let ly = LogWrapper::new(y);
        let lxy = lx * ly;

        assert!((xy - lxy.dissolve()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_logit_math_add() {
        let x = 1.2345;
        let y = 1.1234;
        let xy = x + y;
        let lx = LogWrapper::new(x);
        let ly = LogWrapper::new(y);
        let lxy = lx + ly;

        assert!((xy - lxy.dissolve()).abs() < f64::EPSILON);
    }
}

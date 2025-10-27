use rand::distr::Uniform;
use rand::Rng;
use crate::traits::graph_traits::DOFTypeTrait;

impl DOFTypeTrait for bool {
    fn local_dimension() -> usize {
        2
    }

    fn to_index(&self) -> usize {
        if *self { 1 } else { 0 }
    }

    fn from_index(index: usize) -> Self {
        index != 0
    }

    fn iterate_through_values() -> impl Iterator<Item = Self> {
        [false, true].into_iter()
    }

    fn index_dimension<It>(it: It) -> usize
    where
        It: IntoIterator<Item = Self>,
    {
        it.into_iter()
            .enumerate()
            .map(|(i, v)| (if v { 1 } else { 0 }) << i)
            .sum()
    }

    fn get_random<R>(rng: &mut R) -> Self where R: Rng {
        rng.random()
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct Spin<const N: usize> {
    value: usize,
}

impl<const N: usize> Spin<N> {
    pub fn new(value: usize) -> Self {
        Self { value }
    }
}

impl<const N: usize> DOFTypeTrait for Spin<N> {
    fn local_dimension() -> usize {
        N
    }

    fn to_index(&self) -> usize {
        self.value
    }

    fn from_index(index: usize) -> Self {
        Self { value: index }
    }

    fn iterate_through_values() -> impl Iterator<Item = Self> {
        (0..N).map(Self::new)
    }

    fn get_random<R>(rng: &mut R) -> Self where R: Rng {
        let choice = rng.sample(Uniform::new(0, N).unwrap());
        Self {
            value: choice
        }
    }
}

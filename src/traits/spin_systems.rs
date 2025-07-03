use crate::traits::graph_traits::DOFTypeTrait;

impl DOFTypeTrait for bool {
    fn local_dimension() -> usize {
        2
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

    fn iterate_through_values() -> impl Iterator<Item = Self> {
        (0..N).map(|s| Self::new(s))
    }

    fn index_dimension<It>(it: It) -> usize
    where
        It: IntoIterator<Item = Self>,
    {
        it.into_iter()
            .fold((1, 0), |(mut mult, mut acc), v| {
                acc += mult * v.value;
                mult *= N;
                (mult, acc)
            })
            .1
    }
}

use rand::Rng;
use crate::traits::graph_traits::{DOFTypeTrait, GraphStateNavigator};

pub trait ThermalUpdate: GraphStateNavigator {
    fn thermal_update<R>(&mut self, rng: &mut R) where R: Rng {
        self.apply_function_to_empty_worldlines(rng, |rng, value| {
            *value = Self::DOFType::get_random(rng)
        });
    }

    fn apply_function_to_empty_worldlines<T, F>(&mut self, context: T, f: F) where F: Fn(&mut T, &mut Self::DOFType);
}
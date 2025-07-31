use crate::ring_exchange_term::RingExchangeData;
use QmcSSE::traits::graph_traits::DOFTypeTrait;
use std::collections::{HashMap, HashSet};

pub struct PyrochloreLatticeHelper {
    lx: usize,
    ly: usize,
    lz: usize,
    defects: HashSet<LatticeSite>,
}

#[derive(Eq, PartialEq, Hash, Clone)]
pub struct LatticeSite {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub n: usize,
}

impl From<(usize, usize, usize, usize)> for LatticeSite {
    fn from(value: (usize, usize, usize, usize)) -> Self {
        Self {
            x: value.0,
            y: value.1,
            z: value.2,
            n: value.3,
        }
    }
}

struct LatticeRing {
    coords: Vec<LatticeSite>,
}

impl<T> From<Vec<T>> for LatticeRing
where
    T: Into<LatticeSite>,
{
    fn from(value: Vec<T>) -> Self {
        Self {
            coords: value.into_iter().map(|t| t.into()).collect(),
        }
    }
}

impl LatticeRing {
    fn start_at_unit_cell(
        &self,
        x: usize,
        y: usize,
        z: usize,
        system_x: usize,
        system_y: usize,
        system_z: usize,
    ) -> Vec<LatticeSite> {
        self.coords
            .iter()
            .map(|site| {
                let new_x = (x + site.x) % system_x;
                let new_y = (y + site.y) % system_y;
                let new_z = (z + site.z) % system_z;
                let n = site.n;
                LatticeSite::from((new_x, new_y, new_z, n))
            })
            .collect()
    }
}

impl PyrochloreLatticeHelper {
    pub fn new(lx: usize, ly: usize, lz: usize) -> Self {
        assert_eq!(lx % 2, 0, "lx must be even");
        assert_eq!(ly % 2, 0, "ly must be even");
        assert_eq!(lz % 2, 0, "lz must be even");
        Self {
            lx,
            ly,
            lz,
            defects: Default::default(),
        }
    }

    fn get_unit_cells(&self) -> impl Iterator<Item = (usize, usize, usize)> {
        (0..self.lx)
            .flat_map(move |x| (0..self.ly).flat_map(move |y| (0..self.lz).map(move |z| (x, y, z))))
    }

    pub fn get_spin_list(&self) -> impl Iterator<Item = LatticeSite> {
        self.get_unit_cells()
            .flat_map(move |(x, y, z)| (0..4usize).map(move |n| (x, y, z, n).into()))
            .filter(|key| !self.defects.contains(key))
    }

    /// False/True means into/out-of the unit cell.
    pub fn get_initial_state(&self) -> Vec<bool> {
        let base_config = [false, true, false, true];

        let flip_to_advance_x = [true, true, true, true];
        let flip_to_advance_y = [false, true, true, false];
        let flip_to_advance_z = [false, false, false, false];

        self.get_spin_list()
            .map(|LatticeSite { x, y, z, n }| {
                let mut base = base_config[n];
                // If x is odd, allow flips based on flip_to_advance_x, same for y and z.
                for (j, fj) in [x, y, z].into_iter().zip([
                    flip_to_advance_x[n],
                    flip_to_advance_y[n],
                    flip_to_advance_z[n],
                ]) {
                    let j_odd = j % 2 == 1;
                    if j_odd && fj {
                        base = !base
                    }
                }
                base
            })
            .collect()
    }

    pub fn get_terms(&self) -> Vec<(RingExchangeData<f64>, Vec<usize>)> {
        let spin_list = self.get_spin_list().collect::<Vec<_>>();
        let spin_lookup = spin_list
            .iter()
            .enumerate()
            .map(|(i, x)| (x.clone(), i))
            .collect::<HashMap<_, _>>();

        let mut ring_exchange_terms = vec![];

        // The xy plane loop:
        let ring = LatticeRing::from(vec![
            (0, 0, 0, 1),
            (0, 0, 0, 2),
            (0, 1, 0, 0),
            (0, 1, 0, 1),
            (1, 0, 0, 2),
            (1, 0, 0, 0),
        ]);
        let ring_flip_state = vec![false, true, false, true, false, true];
        ring_exchange_terms.extend(self.make_rings(&ring, &ring_flip_state, &spin_lookup));

        // The xz plane loop:
        let ring = LatticeRing::from(vec![
            (0, 0, 0, 1),
            (0, 0, 0, 3),
            (0, 0, 1, 0),
            (0, 0, 1, 1),
            (1, 0, 0, 3),
            (1, 0, 0, 0),
        ]);
        let ring_flip_state = vec![false, true, false, true, false, true];
        ring_exchange_terms.extend(self.make_rings(&ring, &ring_flip_state, &spin_lookup));

        // The yz plane loop:
        let ring = LatticeRing::from(vec![
            (0, 0, 0, 2),
            (0, 0, 0, 3),
            (0, 0, 1, 0),
            (0, 0, 1, 2),
            (0, 1, 0, 3),
            (0, 1, 0, 0),
        ]);
        let ring_flip_state = vec![false, true, false, true, false, true];
        ring_exchange_terms.extend(self.make_rings(&ring, &ring_flip_state, &spin_lookup));

        // The opposite face
        let ring = LatticeRing::from(vec![
            (1, 0, 0, 2),
            (1, 0, 0, 3),
            (0, 0, 1, 1),
            (0, 0, 1, 2),
            (0, 1, 0, 3),
            (0, 1, 0, 1),
        ]);
        let ring_flip_state = vec![false, true, false, true, false, true];
        ring_exchange_terms.extend(self.make_rings(&ring, &ring_flip_state, &spin_lookup));

        ring_exchange_terms
    }

    fn make_rings(
        &self,
        ring: &LatticeRing,
        flip_state: &[bool],
        spin_lookup: &HashMap<LatticeSite, usize>,
    ) -> impl Iterator<Item = (RingExchangeData<f64>, Vec<usize>)> {
        let other_ring_flip_state = flip_state.iter().map(|b| !*b).collect::<Vec<_>>();
        let index_for_flip = bool::index_dimension_slice(flip_state);
        let other_index_for_flip = bool::index_dimension_slice(&other_ring_flip_state);

        self.get_unit_cells().filter_map(move |(x, y, z)| {
            let ring_from_site = ring.start_at_unit_cell(x, y, z, self.lx, self.ly, self.lz);
            let all_spins_are_present = ring_from_site
                .iter()
                .all(|site| spin_lookup.contains_key(site));
            if all_spins_are_present {
                let term = RingExchangeData::new(1.0, index_for_flip, other_index_for_flip, 1 << 6);
                let indices = ring_from_site
                    .into_iter()
                    .map(|site| spin_lookup.get(&site).copied().expect("Should be present."))
                    .collect();
                Some((term, indices))
            } else {
                None
            }
        })
    }
}

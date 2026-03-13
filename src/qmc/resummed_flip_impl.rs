use crate::qmc::{GenericQMC, MatrixTerm, MatrixTermData, MatrixTermHandle};
use crate::traits::graph_traits::{DOFTypeTrait, GraphNode, GraphStateNavigator};
use crate::traits::graph_weights::GraphWeight;
use crate::traits::resummed_flip_update::{FlipWeight, ResummedFlipUpdater};

trait IntoResummationFlipper: GraphStateNavigator {
    type ResummationFlipper: ResummedFlipUpdater;
    type FlippableMatrixTerm;

    fn into_resummation_flipper(self) -> Self::ResummationFlipper;

    fn get_term_type(&self, term: &Self::FlippableMatrixTerm) -> MatrixTermType;
}

/// Terms for resummation are either offdiagonal acting on a single site, or diagonal acting on
/// one or more sites.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MatrixTermType {
    /// A term with offdiagonal components, acting on a single site
    SingleSiteFlippable,
    /// A term with diagonal components only, possible acting on multiple sites.
    MultiSiteDiagonal
}

impl<DOF, TermData>  IntoResummationFlipper for GenericQMC<DOF, TermData>
where DOF: DOFTypeTrait + DOFFlippable, TermData: MatrixTermData<f64> + MatrixTermFlippableOrDiagonal {
    type ResummationFlipper = GenericResummationFlipper<DOF, TermData>;
    type FlippableMatrixTerm = TermData;

    fn into_resummation_flipper(self) -> Self::ResummationFlipper {
        // TODO: replace substate Vec allocations with single buffer.
        let n_indices = self.get_all_indices().len();
        let mut indices_in_context_per_index = vec![vec![]; n_indices];
        let mut index_to_index_into_all_terms = vec![vec![]; n_indices];
        for (index_into_all_terms, term) in self.all_terms.iter().enumerate() {
            let term_data = &self.all_term_data[term.matrix_data_entry];
            if term_data.get_term_type() == MatrixTermType::MultiSiteDiagonal {
                for index in &term.act_on_indices {
                    index_to_index_into_all_terms[*index].push(index_into_all_terms);
                    indices_in_context_per_index[*index].extend_from_slice(&term.act_on_indices);
                }
            }
        }
        for context_indices in indices_in_context_per_index.iter_mut() {
            context_indices.sort();
            context_indices.dedup();
        }

        let mut state = self.initial_state.clone();
        let mut current_total_weight = 0.0;
        for term in &self.all_terms {
            let term_data = &self.all_term_data[term.matrix_data_entry];
            if term_data.get_term_type() == MatrixTermType::MultiSiteDiagonal {
                let substate = term.act_on_indices.iter().map(|index| state[*index]).collect::<Vec<_>>();
                current_total_weight += self.get_diagonal_matrix_element_from_term(term, &substate);
            }
        }

        let mut flippable_end_points = vec![vec![]; n_indices];
        let mut timeslices_and_weights_for_diagonal_terms = vec![];
        let mut accumulated_probs_for_last_flips = vec![1.0; n_indices];
        for (timeslice, node) in self.time_slices.iter().enumerate() {
            if let Some(node) = node {
                let term = self.get_matrix_term_for_node(node);
                let data = &self.all_term_data[term.matrix_data_entry];
                let term_type = self.get_term_type(data);
                match term_type {
                    MatrixTermType::SingleSiteFlippable => {
                        debug_assert_eq!(term.act_on_indices.len(), 1,
                                         "SingleSiteFlippable terms must act on a single site.");
                        let index = term.act_on_indices[0];
                        let context_indices = &indices_in_context_per_index[index];
                        let last_flippable_indices_for_neighbors = context_indices.iter().map(|context_index| {
                            let l = flippable_end_points[*context_index].len();
                            if l == 0 {
                                None
                            } else {
                                Some(l - 1)
                            }
                        }).collect();
                        flippable_end_points[index].push(FlippableRegionStartingPoint {
                            timeslice,
                            index_of_diagonal_array: timeslices_and_weights_for_diagonal_terms.len(),
                            flip_prob: 1.0,
                            last_flippable_indices_for_neighbors
                        });

                        // Calculate the change in current_total_weight by subtracting off the
                        // operators which connect to this spin, flipping it, then adding those back
                        let affected_terms = &index_to_index_into_all_terms[index];
                        for index_into_all_terms in affected_terms {
                            let term = &self.all_terms[*index_into_all_terms];
                            let substate = term.act_on_indices.iter().map(|index| state[*index]).collect::<Vec<_>>();
                            current_total_weight -= self.get_diagonal_matrix_element_from_term(term, &substate);
                        }
                        state[index] = node.get_output_state()[0];
                        for index_into_all_terms in affected_terms {
                            let term = &self.all_terms[*index_into_all_terms];
                            let substate = term.act_on_indices.iter().map(|index| state[*index]).collect::<Vec<_>>();
                            current_total_weight += self.get_diagonal_matrix_element_from_term(term, &substate);
                        }
                    }
                    MatrixTermType::MultiSiteDiagonal => {
                        debug_assert!(node.is_diagonal(), "MultiSiteDiagonal must be diagonal");

                        // Log that
                        timeslices_and_weights_for_diagonal_terms.push(DiagonalOperatorSum { timeslice, total_weight: current_total_weight });

                        #[cfg(debug_assertions)]
                        let total_weight = {
                            let mut acc_weight = 0.0;
                            for term in &self.all_terms {
                                let term_data = &self.all_term_data[term.matrix_data_entry];
                                if term_data.get_term_type() == MatrixTermType::MultiSiteDiagonal {
                                    let substate = term.act_on_indices.iter().map(|index| state[*index]).collect::<Vec<_>>();
                                    acc_weight += self.get_diagonal_matrix_element_from_term(term, &substate);
                                }
                            }
                            acc_weight
                        };

                        // Now lets update flip probabilities by checking how much the diagonal ops
                        // change when we flip a spin.
                        for index in 0 .. n_indices {
                            let old_state = state[index];

                            let affected_terms = &index_to_index_into_all_terms[index];
                            let mut weight_change = 0.0;
                            for index_into_all_terms in affected_terms {
                                let term = &self.all_terms[*index_into_all_terms];
                                let substate = term.act_on_indices.iter().map(|index| state[*index]).collect::<Vec<_>>();
                                weight_change -= self.get_diagonal_matrix_element_from_term(term, &substate);
                            }
                            let new_state = old_state.flip();
                            state[index] = new_state;

                            #[cfg(debug_assertions)]
                            let spin_flip_total_weight = {
                                let mut acc_weight = 0.0;
                                for term in &self.all_terms {
                                    let term_data = &self.all_term_data[term.matrix_data_entry];
                                    if term_data.get_term_type() == MatrixTermType::MultiSiteDiagonal {
                                        let substate = term.act_on_indices.iter().map(|index| state[*index]).collect::<Vec<_>>();
                                        acc_weight += self.get_diagonal_matrix_element_from_term(term, &substate);
                                    }
                                }
                                acc_weight
                            };

                            for index_into_all_terms in affected_terms {
                                let term = &self.all_terms[*index_into_all_terms];
                                let substate = term.act_on_indices.iter().map(|index| state[*index]).collect::<Vec<_>>();
                                weight_change += self.get_diagonal_matrix_element_from_term(term, &substate);
                            }

                            #[cfg(debug_assertions)]
                            let gt_weight_change = spin_flip_total_weight - total_weight;
                            debug_assert!((weight_change - gt_weight_change).abs() < f64::EPSILON);

                            state[index] = old_state;

                            // Now use the change in term weight to determine flip probabilities.
                            // If there exists a flippable end point, log it there. Otherwise
                            // just put it into `flippable_end_points` for later.
                            let modify_flip_prob = match flippable_end_points[index].last_mut() {
                                Some(fep) => &mut fep.flip_prob,
                                None => &mut accumulated_probs_for_last_flips[index]
                            };
                            *modify_flip_prob *= 1.0 + weight_change / current_total_weight;
                        }
                    }
                }
            }
        }

        for (index, flippables) in flippable_end_points.iter_mut().enumerate() {
            let last_flippable_for_index = flippables.last_mut();
            if let Some(last_flippable_for_index) = last_flippable_for_index {
                last_flippable_for_index.flip_prob *= accumulated_probs_for_last_flips[index];
            } else {
                // The entire worldline has no flippable end points. It can still be flipped but
                // needs to be treated differently.
                todo!()
            }
        }

        // Make a tree of weights for fast lookup.
        let mut weight_tree = fenwick_tree::FenwickTree::with_len(flippable_end_points.len());
        let total_index_to_dof_index_and_subindex = flippable_end_points.iter().enumerate().flat_map(|(index, flippables_for_index)| flippables_for_index.iter().enumerate().map(|(index_for_index, _)| {
            (index, index_for_index)
        })).collect::<Vec<_>>();
        for (total_index, f) in flippable_end_points.iter().flatten().enumerate() {
            weight_tree.add(total_index, f.flip_prob).expect("Adding to tree should always succeed.");
        }

        // TODO finish

        Self::ResummationFlipper {
            initial_state: self.initial_state,
            all_term_data: self.all_term_data,
            all_terms: self.all_terms
        }
    }

    fn get_term_type(&self, term: &Self::FlippableMatrixTerm) -> MatrixTermType {
        term.get_term_type()
    }
}

/// Flip a DOF.
pub trait DOFFlippable {
    /// Flip a DOF.
    fn flip(&self) -> Self;
}

#[derive(Debug, PartialEq, Clone)]
struct FlippableRegionStartingPoint {
    timeslice: usize,
    index_of_diagonal_array: usize,
    flip_prob: f64,
    last_flippable_indices_for_neighbors: Vec<Option<usize>>
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct DiagonalOperatorSum {
    timeslice: usize,
    total_weight: f64
}

/// Returns whether the term is a single site flippable or a multi site diagonal.
pub trait MatrixTermFlippableOrDiagonal {
    /// Returns whether the term is a single site flippable or a multi site diagonal.
    fn get_term_type(&self) -> MatrixTermType;
}

struct GenericResummationFlipper<DOF: DOFTypeTrait, TermData: MatrixTermData<f64>> {
    initial_state: Vec<DOF>,
    all_terms: Vec<MatrixTerm>,
    all_term_data: Vec<TermData>
}

impl<DOF: DOFTypeTrait, TermData: MatrixTermData<f64>> ResummedFlipUpdater for GenericResummationFlipper<DOF, TermData> {
    type FlippableLocation = FlipLocationIndex;

    fn flip_location(&mut self, loc: Self::FlippableLocation) {
        todo!()
    }

    fn get_n_flip_locations(&self) -> usize {
        todo!()
    }

    fn get_flip_weights(&self) -> impl IntoIterator<Item=(Self::FlippableLocation, FlipWeight)> {
        vec![]
        // todo!()
    }
}

#[derive(Clone,Copy,Eq,PartialEq,Ord,PartialOrd)]
struct FlipLocationIndex(usize);

#[cfg(test)]
mod resummed_tests {
    use crate::terms::multibody_tfim::MultibodyTFIMTerm;
    use super::*;

    #[test]
    fn test_construction()-> Result<(), String> {
        let n = 3;
        let mut qmc = GenericQMC::<bool, _>::new(n);
        let zzz = qmc.add_term(MultibodyTFIMTerm::ZN(3, -1.0), [0,1,2]);
        let xa = qmc.add_term(MultibodyTFIMTerm::X(1.0), [0]);
        let xb = qmc.add_term(MultibodyTFIMTerm::X(1.0), [1]);
        let xc = qmc.add_term(MultibodyTFIMTerm::X(1.0), [2]);
        qmc.add_node(0, xa);
        qmc.add_node(1, xb);
        qmc.add_node(2, xc);
        qmc.add_node(3, zzz);
        qmc.add_node(4, xa);
        qmc.add_node(5, xb);
        qmc.add_node(6, xc);

        let flipper = qmc.into_resummation_flipper();

        Ok(())
    }

}
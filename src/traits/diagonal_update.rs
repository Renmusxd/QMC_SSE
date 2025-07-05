use crate::traits::graph_traits::{
    GraphContext, GraphNode, Link, LinkedGraphNode, TimeSlicedGraph,
};
use crate::traits::graph_weights::{GraphWeight, MatrixTermTrait};
use log::debug;
use rand::Rng;

pub trait DiagonalUpdate: TimeSlicedGraph + GraphWeight
where
    Self::Node: LinkedGraphNode,
{
    // Corresponds to M in Sandvik 2019 Eq (18)
    // https://arxiv.org/pdf/1909.10591
    fn get_number_of_time_slices(&self) -> usize;

    // Corresponds to n in Sandvik 2019 Eq (18)
    fn get_number_of_non_identity_operators(&self) -> usize;

    fn construct_node(
        timeslice: &Self::TimesliceIndex,
        context: GraphContext<Self::DOFType, Link<Self::TimesliceIndex>>,
        term: Self::MatrixTerm,
    ) -> Self::Node;

    fn diagonal_update<R>(&mut self, beta: f64, mut rng: R)
    where
        R: Rng,
    {
        debug_assert!(self.check_graph_consistency());

        let mut incoming_state = self.get_initial_state().to_vec();
        let mut last_nodes = vec![None; self.get_num_dof()];
        let nd = self.get_possible_terms().len();
        let prob_sampler = rand::distr::Uniform::new(0.0, 1.0).unwrap();
        let term_sampler = rand::distr::Uniform::new(0, nd).unwrap();

        // Go through timeslices.
        let mut timeslice = self.get_first_timeslice();
        while let Some(t) = timeslice {
            let node = self.get_node(&t);
            let new_node = match node {
                None => {
                    let terms = self.get_possible_terms();
                    let term_to_try = &terms[rng.sample(term_sampler)];
                    let variables_for_term = term_to_try.get_indices_acted_on();
                    let local_state = variables_for_term
                        .iter()
                        .map(|i| {
                            let i = i.clone().into();
                            incoming_state[i].clone()
                        })
                        .collect::<Vec<_>>();

                    // Matrix element <s|H|s> if term H were placed here.
                    let matrix_elem =
                        self.get_diagonal_matrix_element_from_term(term_to_try, &local_state);

                    let numerator = beta * matrix_elem * nd as f64;
                    let denominator = self.get_number_of_time_slices()
                        - self.get_number_of_non_identity_operators();
                    let p_accept = numerator / denominator as f64;

                    if rng.sample(prob_sampler) < p_accept {
                        debug!("Accepted node addition from t={:?} with p={p_accept:?}", &t);
                        let term_to_try = term_to_try.clone();
                        let variables_for_term = term_to_try.get_indices_acted_on().to_vec();

                        let node = self.insert_node_with_hint(
                            &t,
                            &variables_for_term,
                            &last_nodes,
                            |context| Self::construct_node(&t, context, term_to_try),
                        );
                        Some(node)
                    } else {
                        debug!("Rejected node addition from t={:?} with p={p_accept:?}", &t);
                        None
                    }
                }
                Some(node) => {
                    let node_variables = node.get_indices();
                    debug_assert!({
                        let node_input = node.get_input_state();
                        node_variables
                            .iter()
                            .zip(node_input.iter())
                            .all(|(v, node_input)| {
                                let v = v.clone().into();
                                incoming_state[v].eq(node_input)
                            })
                    });

                    // Maybe remove
                    if node.is_diagonal() {
                        let matrix_element = self.get_matrix_element_from_node(node);

                        let numerator = self.get_number_of_time_slices()
                            - self.get_number_of_non_identity_operators()
                            + 1;
                        let denominator = (beta * matrix_element) * nd as f64;
                        let p_accept = numerator as f64 / denominator;

                        if rng.sample(prob_sampler) < p_accept {
                            debug!("Accepted node addition from t={:?} with p={p_accept:?}", &t);
                            self.remove_node(&t);
                            None
                        } else {
                            debug!("Rejected node addition from t={:?} with p={p_accept:?}", &t);
                            Some(node)
                        }
                    } else {
                        Some(node)
                    }
                }
            };

            if let Some(node) = new_node {
                node.iterate_over_outputs().enumerate().for_each(
                    |(rel_index, (global_index, dof_state, _))| {
                        let global_index = global_index.clone().into();
                        incoming_state[global_index] = dof_state.clone();
                        last_nodes[global_index] = Some(Link {
                            timeslice: t.clone(),
                            relative_index: rel_index,
                        });
                    },
                );
            };

            timeslice = self.get_next_timeslice(t);
        }

        debug_assert!(self.check_graph_consistency());
    }
}

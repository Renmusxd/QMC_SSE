use crate::qmc::{GenericQMC, MatrixTermData};
use crate::traits::diagonal_update::DiagonalUpdate;
use crate::traits::graph_traits::DOFTypeTrait;
use rustfft;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use std::ops::DivAssign;

impl<DOF: DOFTypeTrait, Data: MatrixTermData<f64>> GenericQMC<DOF, Data> {
    pub fn autocorr_for_terms<F>(&mut self, beta: f64, n_steps: usize, mut step: F) -> Vec<f64>
    where
        F: FnMut(&mut Self),
    {
        let all_data = (0..n_steps)
            .map(|_| {
                step(self);
                self.get_each_expectation_value(beta)
            })
            .collect::<Vec<_>>();

        fft_autocorrelation(&all_data)
    }
}

pub(crate) fn fft_autocorrelation(samples: &[Vec<f64>]) -> Vec<f64> {
    let tmax = samples.len();
    let n = samples[0].len();

    let means = (0..n)
        .map(|i| (0..tmax).map(|t| samples[t][i]).sum::<f64>() / tmax as f64)
        .collect::<Vec<_>>();

    let mut input = (0..n)
        .map(|i| {
            let mut v = (0..tmax)
                .map(|t| Complex::<f64>::new(samples[t][i] - means[i], 0.0))
                .collect::<Vec<Complex<f64>>>();
            let norm = v.iter().map(|v| (v.conj() * v).re).sum::<f64>().sqrt();
            v.iter_mut().for_each(|c| c.div_assign(norm));
            v
        })
        .collect::<Vec<_>>();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(tmax);
    let mut iplanner = FftPlanner::new();
    let ifft = iplanner.plan_fft_inverse(tmax);

    input.iter_mut().for_each(|input| {
        fft.process(input);
        input
            .iter_mut()
            .for_each(|c| *c = Complex::new(c.norm_sqr(), 0.0));
        ifft.process(input);
    });

    (0..tmax)
        .map(|t| (0..n).map(|i| input[i][t].re).sum::<f64>() / ((n * tmax) as f64))
        .collect()
}

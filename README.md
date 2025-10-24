A rust library for Quantum Monte Carlo.
A continuation and rewrite of previous work at [IsingMonteCarlo](https://github.com/Renmusxd/IsingMonteCarlo)

# Stochastic Series Expansion Quantum Monte Carlo
Quantum Monte Carlo (QMC) is a family of techniques used to study the thermal properties of quantum systems.
Stochastic Series Expansion (SSE) is a specific implementation constructed for discrete spin-like systems.
For more information about the technique, see this [review article](https://arxiv.org/pdf/1909.10591).
We provide a brief overview here.

The goal of this library is to extract thermal expectation values from a Hamiltonian $H$.
The expectation values may be written as a trace 
$$\langle \hat{Q} \rangle = \frac{1}{Z} \text{Tr}\left[ \hat{Q} e^{- \beta H} \right]$$
Where we take $H$ to be a sum over terms
$$ H = -\sum_{i=0}^m H_i$$

The key insight of SSE is that the exponential may be expanded in terms of the series order $n$,
$$ Z \equiv \text{Tr}\left[e^{\beta H}\right] = \sum_{n=0}^\infty \frac{\beta^n}{n!} \sum_{\{h_i\}} \sum_{\alpha_0} \langle \alpha_0| h_n h_{n-1} \ldots h_1 h_0 | \alpha_0 \rangle$$
where each of $h_i \in \{H_0, \ldots H_m\}$ is a term in the Hamiltonian, and the sum over $\{h_i\}$ represents a sum over operator strings of length $n$.
We now treat $n$ and $\{h_i\}$ as MCMC variables which are sampled using standard methods like Metropolis-Hastings or Heat Bath.
Diagonal observables may be estimated by measuring $\langle \alpha_0 \vert \hat{Q} \vert \alpha_0 \rangle$ with $\alpha_0$
sampled from the thermal distribution (See Sec. 3.1 in the [review](https://arxiv.org/pdf/1909.10591)).
The expected total energy may be estimated by evaluating $\langle H \rangle = -\partial_\beta \ln Z$, giving
$$ \langle H \rangle = - \frac{1}{\beta} \langle n \rangle$$
which is to say the expected energy is proportional to the expected operator string length.
Similarly, the expectation value of any term in the Hamiltonian may be estimated by counting the number of occurences of
that specific term in the operator string,
$$ \langle H_i \rangle = - \frac{1}{\beta} \langle n_i \rangle$$
making those off-diagonal terms simple to estimate. Other off-diagonal observables are not trivial to estimate, (See Sec. 3.2 in the [review](https://arxiv.org/pdf/1909.10591)).

# Library API
TODO

# Examples
TODO
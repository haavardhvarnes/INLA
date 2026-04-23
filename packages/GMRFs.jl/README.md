# GMRFs.jl

Sparse Gaussian Markov Random Fields for Julia.

A standalone, dependency-light package providing:

- `AbstractGMRF` and concrete types for IID, random-walk, AR1, seasonal,
  Besag, generic sparse-precision models.
- Sampling from `N(μ, Q⁻¹)` with and without linear constraints.
- Log-density evaluation and log-determinant.
- Marginal variances via selected inversion.
- Graphs.jl-compatible graph representation.
- Multiple sparse factorization backends via LinearSolve.jl.

This package is the numerical core of the Julia INLA ecosystem but is
usable independently for any application involving sparse Gaussian models:
spatial smoothing, disease mapping with external MCMC, image restoration,
4D-Var data assimilation.

## Status

Planning. See [`plans/plan.md`](plans/plan.md).

## Quick example (target API)

```julia
using GMRFs, Graphs

# Besag prior on a Germany-like adjacency graph
g = load_graph("germany.graph")
prior = BesagGMRF(graph = g, τ = 1.0, scale = true)

# Sample from the prior
using Random
x = rand(MersenneTwister(1), prior)

# Evaluate log-density
logpdf(prior, x)

# Marginal variances diag(Q⁻¹)
σ² = marginal_variances(prior)
```

## Installation

Not yet registered. When ready:
```julia
using Pkg
Pkg.add("GMRFs")
```

## See also

- [`LatentGaussianModels.jl`](../LatentGaussianModels.jl/) — builds LGMs on
  top of GMRFs.
- [`INLASPDE.jl`](../INLASPDE.jl/) — SPDE–Matérn on triangulated meshes.

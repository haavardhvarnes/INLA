# GMRFs.jl

Sparse Gaussian Markov Random Fields for Julia.

A standalone, dependency-light package providing:

- `AbstractGMRF` and concrete types for IID, random-walk, AR(1),
  seasonal, Besag, and `Generic0` sparse-precision models.
- `GMRFGraph` — `Graphs.jl`-compatible graph wrapping any sparse
  adjacency.
- Sampling from `N(μ, Q⁻¹)` with and without linear constraints.
- Log-density evaluation and log-determinant.
- Marginal variances via selected inversion.
- Sparse factorization caching (`FactorCache`) — symbolic-once,
  refactor-on-θ-change — exposed through `LinearSolve.jl`.

This package is the numerical core of the Julia INLA ecosystem but is
usable independently for any application involving sparse Gaussian
models: spatial smoothing, disease mapping with external MCMC, image
restoration, 4D-Var data assimilation.

## Status

`v0.1.0-rc1`. See the ecosystem [`CHANGELOG.md`](../../CHANGELOG.md)
for what landed and where R-INLA parity is known to be loose.

Shipped concrete types: `IIDGMRF`, `RW1GMRF`, `RW2GMRF`, `AR1GMRF`,
`SeasonalGMRF`, `BesagGMRF`, `Generic0GMRF`.

## Quick example

```julia
using GMRFs, Graphs, SparseArrays, Random

# Besag prior on a small adjacency. `GMRFGraph` accepts a
# `Graphs.AbstractGraph` or a sparse adjacency matrix.
g = GMRFGraph(cycle_graph(6))
prior = BesagGMRF(g; τ = 1.0, scale_model = true)

# Sample from the prior (linear sum-to-zero constraint applied).
x = rand(MersenneTwister(1), prior)

# Log-density
ℓ = logpdf(prior, x)

# Marginal variances diag(Q⁻¹) via selected inversion
σ² = marginal_variances(prior)

# Reuse the symbolic factor when only τ changes
cache = FactorCache(prior)
update!(cache, BesagGMRF(g; τ = 4.0, scale_model = true))
```

## Installation

Registered on a personal Julia registry at
[`haavardhvarnes/JuliaRegistry`](https://github.com/haavardhvarnes/JuliaRegistry) —
add it once, then `Pkg.add` as usual:

```julia
using Pkg
Pkg.Registry.add(RegistrySpec(url = "https://github.com/haavardhvarnes/JuliaRegistry"))
Pkg.Registry.add("General")  # idempotent if already added
Pkg.add("GMRFs")
```

## See also

- [`LatentGaussianModels.jl`](../LatentGaussianModels.jl/) — builds
  LGMs on top of GMRFs.
- [`INLASPDE.jl`](../INLASPDE.jl/) — SPDE–Matérn on triangulated
  meshes.
- [`GMRFsPardiso.jl`](../GMRFsPardiso.jl/) — license-gated MKL/Panua
  Pardiso backend (separate package).

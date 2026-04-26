# GMRFs.jl

Sparse Gaussian Markov random fields: graph, precision, sampling,
conditioning, log-density. The numerical core of the ecosystem;
useful standalone for anyone needing GMRF primitives without the
INLA machinery on top.

## What's here

- An abstract type `AbstractGMRF` plus concrete leaves: `IIDGMRF`,
  `RW1GMRF`, `RW2GMRF`, `AR1GMRF`, `SeasonalGMRF`, `BesagGMRF`.
- `GMRFGraph` — the topological substrate (a thin wrapper around
  Graphs.jl's `SimpleGraph` plus the adjacency `SparseMatrixCSC`).
- Sparse Cholesky-based sampling, log-density evaluation, marginal
  variances via [`SelectedInversion.jl`](https://github.com/SciML/SelectedInversion.jl).
- A `FactorCache` that exposes LinearSolve.jl's symbolic-factorisation
  reuse so callers can repeatedly solve `Q x = b` for varying values
  of `Q` with the same sparsity pattern. Hot-path callers (the inner
  Newton in `LatentGaussianModels.fit_laplace!`) lean on this.
- Linear-constraint types (`NoConstraint`, `LinearConstraint`) and
  Freni-Sterrantino-style sum-to-zero constraints for intrinsic CARs
  on disconnected graphs.

## Required methods for new GMRF subtypes

```julia
num_nodes(g)           -> Int
precision_matrix(g)    -> AbstractMatrix
constraints(g)         -> AbstractConstraint   # default: NoConstraint()
```

Sampling and log-density fall out of the precision matrix and
constraint set automatically.

## API

```@autodocs
Modules = [GMRFs]
```

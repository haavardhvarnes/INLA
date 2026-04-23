# Guidance for Claude Code in GMRFs.jl

Extends [`/CLAUDE.md`](../../CLAUDE.md). This file narrows scope for work
inside this package.

## Scope

This package owns:
- Abstract and concrete GMRF types.
- Sparse precision matrix construction and manipulation.
- Sampling (conditional and unconditional, with linear constraints).
- Log-density, log-determinant, selected inversion.
- Graph abstraction and Graphs.jl interop.

Out of scope:
- Observation likelihoods → `LatentGaussianModels.jl`.
- Hyperparameter priors → `LatentGaussianModels.jl`.
- Inference algorithms → `LatentGaussianModels.jl`.
- SPDE finite-element assembly → `INLASPDE.jl`.
- Plotting → `GMRFsMakieExt` weakdep.

If a request falls outside this scope, say so and point to the right
package.

## Dependencies allowed

Core only:
- SparseArrays, LinearAlgebra, Random, Statistics (stdlib)
- Graphs (JuliaGraphs)
- LinearSolve (SciML)
- Distributions (standard)
- ChainRulesCore (standard, ~zero mass — promoted from weakdep per
  ADR / plans/dependencies.md)

Weakdeps:
- MakieCore (plotting recipes)

Pardiso is **not** a weakdep of this package — it lives in the separate
`packages/GMRFsPardiso.jl/` sub-package because it is license-gated and
we want failure modes to be explicit.

Nothing else may enter `[deps]` without an ADR.

## Performance-critical paths

These are the hot paths. Benchmark before changing, benchmark after:
- Sparse Cholesky factorization (via LinearSolve).
- Sampling from `N(0, Q⁻¹)` — called repeatedly inside inference.
- Log-density evaluation — called many times per θ in INLA.
- Selected inversion for marginal variances — known hotspot.

The inner Newton loop in `LatentGaussianModels.jl` calls into this package
thousands of times per fit. Allocations in these paths multiply. Use
`@benchmark` from BenchmarkTools and check with `--track-allocation`.

## Symbolic factorization reuse

Q's sparsity pattern is fixed while its values change with θ. LinearSolve's
`init(prob)` + `solve!(cache)` pattern exposes this — use it. The
`FactorCache` type in this package wraps that pattern; any new sampling or
solve routine goes through it.

## Disconnected graphs

ICAR and Besag on disconnected graphs require one sum-to-zero constraint
per connected component (Freni-Sterrantino et al. 2018). This is the
single most common silent bug in this space. Any constraint-generation
code must use `Graphs.connected_components`, not assume connectivity.

## Testing conventions

- `test/regression/` — closed-form tests (dense Cholesky, known
  precisions).
- `test/oracle/` — R-INLA `inla.qsample` and `inla.qinv` outputs as JLD2
  fixtures.
- Every new GMRF subtype needs tests in both tiers.

## Style

- Type parameters on all struct fields that appear in hot paths.
- Abstract type docstrings list required methods explicitly.
- No `Any` in hot-path signatures.
- Prefer `rand!(rng, dest, gmrf)` patterns (in-place, explicit RNG) over
  allocating variants; define the allocating variant as a one-liner.

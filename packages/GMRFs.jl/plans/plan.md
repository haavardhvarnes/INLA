# GMRFs.jl — package plan

## Goal

A dependency-light, standalone-useful sparse GMRF library that is also the
numerical core of the Julia INLA ecosystem. Scoped to what the R-INLA C
library `gmrflib/` provides, reimplemented idiomatically in Julia.

## Deliverables — v0.1

Module layout:

```
src/
├── GMRFs.jl                 # main module, exports
├── graph.jl                 # AbstractGMRFGraph, GMRFGraph, interop
├── precision.jl             # SymmetricQ, tabulated Q from callbacks
├── gmrf.jl                  # AbstractGMRF, IID, RW1, RW2, AR1, Seasonal, Besag, Generic0
├── factorization.jl         # FactorCache, symbolic/numeric reuse via LinearSolve
├── sampling.jl              # rand, rand!, conditional sampling
├── logdensity.jl            # logpdf, log-determinant
├── constraints.jl           # LinearConstraint, constraint correction
├── marginals.jl             # diag(Q⁻¹) via selected inversion
└── utils.jl                 # internal helpers (no exports)

ext/
├── GMRFsChainRulesExt.jl
├── GMRFsMakieExt.jl
└── GMRFsPardisoExt.jl

test/
├── runtests.jl
├── regression/
│   ├── test_graph.jl
│   ├── test_iid.jl
│   ├── test_rw.jl
│   ├── test_ar1.jl
│   ├── test_besag.jl
│   ├── test_sampling_covariance.jl     # MC check Cov(samples) ≈ Q⁻¹
│   ├── test_logdet_dense.jl            # sparse vs dense Cholesky logdet
│   ├── test_constraints.jl
│   └── test_marginals_vs_inv.jl        # selected inv vs dense inv
└── oracle/
    ├── fixtures/                        # JLD2 files (R-INLA qsample/qinv)
    └── test_against_inla.jl
```

## Milestones

### M1 — Graph + precision (2 weeks)

- [ ] `AbstractGMRFGraph`, `GMRFGraph` (holds `SimpleGraph` + sparsity).
- [ ] Constructors from `SimpleGraph`, from `SparseMatrixCSC`, from adjacency list.
- [ ] Graphs.jl interface methods pass through.
- [ ] `connected_components` + per-component constraint helpers.
- [ ] `SymmetricQ` wrapping `SparseMatrixCSC`, lower-triangle-only storage.
- [ ] Tabulated Q from `(i, j) -> value` callback (the gmrflib Qfunc pattern).

### M2 — Concrete GMRFs (3 weeks)

- [ ] `IIDGMRF(n; τ)`.
- [ ] `RW1GMRF(n; τ, cyclic=false)`, `RW2GMRF(n; τ, cyclic=false)`.
- [ ] `AR1GMRF(n; ρ, τ)`.
- [ ] `SeasonalGMRF(n; period, τ)`.
- [ ] `BesagGMRF(graph; τ, scale_model=true)`.
- [ ] `Generic0GMRF(R; τ, rankdef, scale_model)` — user-supplied structure
      matrix `R`, precision `Q = τ R`.
- [ ] Each has closed-form log-determinant where available (RW, AR1).

Note: R-INLA's `generic1` differs from `generic0` by a single-scalar
rescaling of the eigenvalues of `R` (so that the largest eigenvalue of the
rescaled structure is 1). This is **not** a new GMRF type — it is a
constructor option. Exposed at the `LatentGaussianModels.jl` component
layer as `Generic1(...)` which calls `Generic0GMRF` after eigen-rescaling.
Keeping the split here avoids type proliferation in the numeric core.

### M3 — Sampling, log-density, constraints (3 weeks)

- [ ] `rand!(rng, x, gmrf)` — unconditional sample via `L⁻ᵀ z`.
- [ ] `rand(rng, gmrf)` — allocating version.
- [ ] Conditional sampling given `x_A = a`.
- [ ] `logpdf(gmrf, x)` using Cholesky factor.
- [ ] `LinearConstraint(A, e)` type, `constrain(gmrf, constraint)` correction.
- [ ] Multi-component constraint handling (for disconnected Besag).
- [ ] All sampling routines take explicit `AbstractRNG`; no global state.

### M4 — Marginal variances interface (1 week — reference impl only)

v0.1 ships only the **interface** and a correctness-only reference:

- [ ] `marginal_variances(gmrf)` public signature.
- [ ] Dense reference implementation via `diag(inv(Matrix(Q)))` — slow but
      correct, gated on `n < n_dense_limit` (default 1000); otherwise throw
      with a pointer to `LatentGaussianModels.jl` Phase 3.
- [ ] Correctness test against analytic `Q⁻¹` on IID, RW1, AR1.

The production sparse implementation (Takahashi recursion / `SelectedInversion.jl`
evaluation) moves to **Phase 3** in `LatentGaussianModels.jl`, where it is
actually needed for the Laplace step. See ADR-004. This avoids blocking
GMRFs.jl v0.1 on the single biggest numerical risk item.

### M5 — Factorization backends (1 week)

- [ ] `FactorCache` wrapping LinearSolve `init(prob)` pattern.
- [ ] Reuse symbolic factorization across θ updates.
- [ ] Pardiso extension with backend-specific optimizations.

### M6 — Oracle fixtures (1 week)

- [ ] `scripts/generate-fixtures/gmrf/` R scripts producing `inla.qsample`
      and `inla.qinv` outputs for each concrete GMRF type.
- [ ] Loader + comparator in `test/oracle/`.
- [ ] CI runs oracle tier on tagged releases only (slow).

## Out of scope for v0.1

- Non-Markov GMRFs (dense Matérn fields not expressible via sparse Q).
- SPDE FEM matrices — those live in INLASPDE.jl.
- Any hyperparameter inference.
- Plotting beyond minimal Makie recipes (in ext).

## Interfaces exported

Committed public surface (stable from v0.1 onwards):

```julia
# Types
AbstractGMRF, AbstractGMRFGraph
GMRFGraph, SymmetricQ
IIDGMRF, RW1GMRF, RW2GMRF, AR1GMRF, SeasonalGMRF, BesagGMRF, Generic0GMRF
LinearConstraint, FactorCache

# Functions (extended from Base/Distributions/Random)
rand, rand!, logpdf, length, eltype

# Package-owned
graph, precision_matrix, constraints, constrain
marginal_variances, log_determinant
factor, factorize!
scale_model, scale_factor
```

## Benchmark targets (v0.1)

Benchmarks are **tracked, not gated** in v0.1. A `benchmarks/` suite
records numbers on each tagged release; numbers do not block merge. The
targets below are *aspirational reference points*, not acceptance criteria:

- Sample from `N(0, Q⁻¹)` on 10k-node Besag graph: aim within 2× R-INLA.
- `logpdf` on same: aim within 2× R-INLA.
- Marginal variances: not measured in v0.1 (production impl moves to
  Phase 3, see M4).

R-INLA comparison numbers are reproducible only against a pinned R-INLA
version; the `renv.lock` from `scripts/generate-fixtures/` is the
authoritative reference.

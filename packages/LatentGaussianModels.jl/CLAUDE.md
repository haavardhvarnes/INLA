# Guidance for Claude Code in LatentGaussianModels.jl

Extends [`/CLAUDE.md`](../../CLAUDE.md). Narrowed scope for this package.

## Scope

This package owns:
- `AbstractLatentComponent` and concrete components (IID, RW, AR1, Besag,
  BYM2, ICAR, Leroux, Seasonal, Generic0/1).
- `AbstractLikelihood` and exponential-family implementations.
- `AbstractLinkFunction`.
- `AbstractHyperPrior` (PC priors, Gamma, LogNormal, etc.).
- `AbstractInferenceStrategy` (EmpiricalBayes, Laplace, INLA, HMC bridge).
- `AbstractIntegrationScheme` (EB, CCD, Grid, GridMCMC, GaussHermite).
- The `LatentGaussianModel` struct and its fit/predict API.
- Diagnostics: DIC, WAIC, CPO, PIT, log marginal likelihood.

Out of scope:
- Sparse precision internals → `GMRFs.jl`.
- SPDE FEM assembly → `INLASPDE.jl`.
- Plotting → `LGMMakieExt` weakdep.
- `@lgm` formula sugar → separate `LGMFormula.jl` sub-package (ADR-008).
- Turing/NUTS bridge → separate `LGMTuring.jl` sub-package (ADR-009).
  Core LGM commits only to `LogDensityProblems.jl` conformance.

## Key patterns

### Component contract

Every `AbstractLatentComponent` must implement:

```julia
graph(c)                           # -> AbstractGMRFGraph
precision_matrix(c, θ)             # -> SparseMatrixCSC
log_hyperprior(c, θ)               # -> Real
initial_hyperparameters(c)         # -> Vector
length(c)                          # -> Int
```

Optional:

```julia
prior_mean(c, θ)                   # default: zeros
constraints(c)                     # default: NoConstraint()
gmrf(c, θ)                         # default: constructed from precision_matrix
```

If you're adding a new component, start with the contract and write its
tests before implementing `precision_matrix`. The test "reconstructed
precision matches an analytical reference" is the primary correctness
check.

### Likelihood contract

```julia
log_density(ℓ, y, η, θ)            # -> Real, scalar log p(y | η, θ)
∇_η_log_density(ℓ, y, η, θ)        # -> AbstractVector, same length as y
∇²_η_log_density(ℓ, y, η, θ)       # -> AbstractVector, same length as y
                                   #    (diagonal of Hessian wrt η; the
                                   #    likelihood factorizes over obs)
link(ℓ)                            # -> AbstractLinkFunction
```

Return-type conventions, load-bearing for the inner Newton hot path:

- `∇_η_log_density` and `∇²_η_log_density` return a plain
  `AbstractVector{<:Real}` — **not** `Diagonal`, **not** a
  `SparseMatrixCSC`. The caller wraps with `Diagonal(...)` only where a
  matrix is algebraically required. This avoids allocation of wrapper
  types inside Newton iterations.
- Length equals the number of observations `length(y)`. Assertions are
  acceptable at the public function boundary; inside the hot path they
  are `@assert`-free.
- For likelihoods where the Hessian is not strictly diagonal
  (e.g. Dirichlet-multinomial), this contract does not fit — such
  likelihoods must implement a different `AbstractLikelihood` subtype
  with a dedicated method (not part of v0.1).

AD fallbacks via ForwardDiff/Enzyme are acceptable for user-defined
likelihoods. Closed-form implementations for Gaussian, Poisson,
Binomial, NegativeBinomial, Gamma are required for performance.

`θ` carries likelihood hyperparameters (Gaussian τ, NegBin size, Gamma
shape); likelihoods without such hyperparameters ignore it.

### Inference strategy contract

The canonical entry point is `fit`, dispatched on
`AbstractInferenceStrategy`:

```julia
fit(model, y, strategy = INLA(); kwargs...) -> AbstractInferenceResult
```

Third-party strategies subtype `AbstractInferenceStrategy` and add a
`fit` method; no change to LGM's source. The default is `INLA()` with
`int_strategy = :auto` (→ `CCD()` for dim(θ) > 2, `Grid()` otherwise)
per ADR-010.

Convenience aliases are exported for the common strategies
(per ADR-011):

```julia
inla(model, y; kwargs...)              == fit(model, y, INLA(); kwargs...)
laplace(model, y; kwargs...)           == fit(model, y, Laplace(); kwargs...)
empirical_bayes(model, y; kwargs...)   == fit(model, y, EmpiricalBayes(); kwargs...)
```

Quickstart docs use the aliases. Reference docs use the dispatched
`fit` form so third-party strategies are visible in the same layout.

## Performance-critical paths

1. **Inner Newton loop** for mode of `x | θ, y`. Called many times per fit.
   - No allocation in the inner iteration.
   - `FactorCache` reused across Newton steps (same sparsity pattern).
   - AD of the likelihood gradient uses `DiffRules` where possible; fall
     back to ForwardDiff only for user-defined likelihoods.

2. **Log-marginal evaluation `log π(θ | y)`**. Called at every CCD/grid
   point. The expensive part is the Laplace step; parallelize across θ
   points via threading.

3. **θ-mode finding.** Optimization.jl with quasi-Newton; Hessian needed
   at the mode for the integration basis.

## Testing conventions

- `test/regression/` — closed-form tests for components, likelihoods, priors.
- `test/oracle/fixtures/` — R-INLA fits on canonical datasets.
- `test/triangulation/` — Stan/NIMBLE cross-checks for BYM2 and geostats.

For every concrete component: a dense-vs-sparse precision test, a sampling
covariance test, an R-INLA posterior-agreement test on at least one dataset.

## What not to do

- Do not introduce a formula DSL here. `@lgm` lives in `LGMFormula.jl`.
  See `plans/macro-policy.md` and ADR-008.
- Do not depend on Turing or AdvancedHMC. Core LGM conforms to
  `LogDensityProblems`; the Turing bridge lives in `LGMTuring.jl`. See
  ADR-009.
- Do not add StatsModels or DataFrames to any dep list (core, weakdep,
  test). Tables.jl is the data-input contract.
- Do not silently differ from R-INLA defaults. See
  `plans/defaults-parity.md` for the complete list, including the
  public kwargs table (ADR-010).
- Do not add new `[deps]` without an ADR.

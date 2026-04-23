# Architecture

The ecosystem is organized around seven abstract types and two main seams
(component ↔ inference; observation ↔ latent field). This document describes
the load-bearing abstractions. Concrete types and package-specific APIs live
in each package's `plans/plan.md`.

## Abstract type hierarchy

```
AbstractGMRF                       # in GMRFs.jl
    ↳ knows: graph, Q(θ), maybe μ(θ)
    ↳ supports: sample, logpdf, conditional, marginal variances

AbstractGMRFGraph                  # in GMRFs.jl
    ↳ wraps Graphs.jl + SparseMatrixCSC sparsity pattern

AbstractLatentComponent            # in LatentGaussianModels.jl
    ↳ the `f(...)` term of R-INLA's formula syntax
    ↳ required methods: graph, precision_matrix(c, θ), log_hyperprior,
      initial_hyperparameters, length
    ↳ optional: prior_mean(c, θ), constraints(c)

AbstractLikelihood                 # in LatentGaussianModels.jl
    ↳ wraps a Distributions.jl family + link function
    ↳ log_density(ℓ, y, η, θ)              — scalar log p(y | η, θ)
    ↳ ∇_η_log_density(ℓ, y, η, θ)          — gradient wrt η, as AbstractVector
    ↳ ∇²_η_log_density(ℓ, y, η, θ)         — diagonal of Hessian wrt η, as
                                             AbstractVector (likelihood
                                             factorizes over observations)
    ↳ θ carries likelihood hyperparameters: Gaussian τ, NegBinomial size,
      Gamma shape. Components without such hyperparameters ignore θ.
    ↳ AD fallbacks (ForwardDiff/Enzyme) are acceptable for user-defined
      likelihoods; closed forms required for Gaussian/Poisson/Binomial.

AbstractLinkFunction               # in LatentGaussianModels.jl
    ↳ IdentityLink, LogLink, LogitLink, ProbitLink, …

AbstractHyperPrior                 # in LatentGaussianModels.jl
    ↳ evaluates log π(θ) and transforms to internal real-valued scale
    ↳ PC priors, Gamma, LogNormal, WeakPrior, …

AbstractInferenceStrategy          # in LatentGaussianModels.jl
    ↳ dispatches the whole inference algorithm
    ↳ EmpiricalBayes, Laplace, INLA, HMC, VB

AbstractIntegrationScheme          # in LatentGaussianModels.jl
    ↳ strategy for the outer θ integration inside INLA
    ↳ EmpiricalBayes, CCD, Grid, GridMCMC, GaussHermite
```

## Component ↔ GMRF composition rule

Both `GMRFs.jl` and `LatentGaussianModels.jl` ship concrete types with
overlapping names (`BesagGMRF` / `Besag`, `RW1GMRF` / `RW1`, etc.). They
are **not** two parallel hierarchies — they are layered:

- **`GMRFs.jl`** owns the *numerical core*: a type like `BesagGMRF`
  holds a graph and a precision pattern, and supports `rand`, `logpdf`,
  `precision_matrix`, `constraints`. It knows nothing about
  hyperpriors, inference, or how it will be used in a linear predictor.

- **`LatentGaussianModels.jl`** owns the *Bayesian-component wrapper*:
  a type like `Besag <: AbstractLatentComponent` **holds a `BesagGMRF`
  by composition** plus an `AbstractHyperPrior`, optional constraint
  modifications, and any Sørbye-Rue scaling metadata. Its
  `precision_matrix(c::Besag, θ)` method delegates to the wrapped
  `BesagGMRF`, reparameterizing via `θ` first.

**Rules:**

1. An LGM component does not reimplement a precision formula that
   exists in GMRFs.jl. If the numerical core does not expose what the
   component needs, extend the numerical core — do not duplicate it.
2. A GMRF in GMRFs.jl is usable standalone (it is the library's
   raison d'être); the LGM component adds Bayesian plumbing on top.
3. Shared function names (`graph`, `precision_matrix`, `constraints`)
   are **extended** across packages, not redefined. LGM's
   `graph(::AbstractLatentComponent)` is `GMRFs.graph` with additional
   methods — tested by Aqua's method-ownership check.
4. `BYM2` and other composite components wrap **multiple** GMRFs
   (typically an `IIDGMRF` + a scaled `BesagGMRF`) and compose their
   precisions. The composite is still a single `AbstractLatentComponent`.

This layering is the reason the package split in ADR-001 works.
Violations — e.g. `Besag` in LGM reimplementing Q construction from
scratch — should be caught in review and refactored to delegate.

## The LGM object

```julia
struct LatentGaussianModel{L,C,P,A}
    likelihood::L                  # <: AbstractLikelihood
    components::C                  # Tuple of AbstractLatentComponent
    linear_predictor::P            # how components contribute to η
    projector::A                   # observation-to-latent mapping (see below)
end
```

Composition is explicit: a `LatentGaussianModel(...)` constructor takes the
components as a tuple. No macro required. Optional macro sugar
(`@lgm y ~ intercept + f(region, Besag(W)) + ...`) expands to this call.

## The projector question — open issue

In SPDE models, the observation vector `y` relates to the latent field `x`
through a projector matrix `A` (observation points → mesh vertices). In
areal models, `A` is identity or an aggregation. In misaligned multi-response
models, there are multiple `A`'s per likelihood.

R-INLA handles this with `inla.stack`, which conflates observation mapping,
data stacking, and effect indexing into one object. The initial design held
`A` as a field on `LatentGaussianModel`; experience in Phase 3/4 will likely
promote this to its own abstract type `AbstractObservationMapping` with
dispatch on the likelihood and component kinds. Tracked as ADR candidate.

## Module / package boundaries

```
GMRFs.jl
├── Graph       (AbstractGMRFGraph, Graphs.jl interop)
├── Precision   (SymmetricQ, tabulated Q functions)
├── GMRF        (AbstractGMRF, concrete types)
├── Sampling    (rand, rand!, conditional)
├── LogDensity  (logpdf, log-determinant via Cholesky)
├── Constraints (linear hard/soft, constraint correction)
└── Marginals   (diag(Q⁻¹) via selected inversion)

LatentGaussianModels.jl
├── Components   (AbstractLatentComponent + IID, RW, AR1, Besag, BYM2, …)
├── Likelihoods  (AbstractLikelihood + Gaussian, Poisson, Binomial, …)
├── Priors       (AbstractHyperPrior + PC priors, Gamma, …)
├── Model        (LatentGaussianModel, linear predictor, projector)
├── Laplace      (inner Newton for mode of x | θ, y)
├── INLA         (outer loop, hyperparameter integration)
└── Diagnostics  (DIC, WAIC, CPO, PIT, marginal likelihood)

INLASPDE.jl
├── Assembly     (FEM matrices C, G₁, G₂ from Meshes triangulation)
├── Matern       (SPDE–Matérn link, α ∈ {1, 2} + rational for fractional)
├── Projector    (mesh vertex → observation point mapping)
└── Priors       (PC priors on range and σ, Fuglstad et al. 2019)
```

## Composability tests

The architecture should pass these sanity checks by construction:

1. A user who writes a new `AbstractLatentComponent` subtype with the five
   required methods gets full integration with INLA, Laplace, and MCMC
   inference strategies, with no source changes to our packages.
2. Swapping a `LinearSolve.KLUFactorization()` for
   `LinearSolve.MKLPardisoFactorize()` inside the inference loop requires
   one kwarg, not a code change.
3. Adding a new likelihood (say, Tweedie) is a struct plus methods, with no
   touches to the components or the INLA loop.
4. A Turing user can wrap a `LatentGaussianModel` as a
   `LogDensityProblems.jl` target and use NUTS, without any INLA machinery
   loading.

If any of these require source changes in our packages at the time of
writing, the architecture has leaked and needs fixing.

## Planned traits

Traits let us specialize without type explosion:

- `IsMarkov(::Type{<:AbstractLatentComponent})` — Markov precision structure
  allows sparse Q; otherwise we need different machinery.
- `IsStationary(::Type{<:AbstractLatentComponent})` — enables FFT-based
  simulation on regular grids.
- `SupportsLaplace(::Type{<:AbstractLikelihood})` — some likelihoods violate
  the Laplace assumption (heavy-tailed / multimodal); this trait lets us
  warn at fit time.
- `HasClosedFormLogdet(::Type{<:AbstractGMRF})` — RW1, AR1 have known
  log-determinants; use them instead of Cholesky when available.

## References

- Rue, Martino, Chopin 2009. The INLA paper.
- Lindgren, Rue, Lindström 2011. SPDE.
- See `references/papers.md` for the full annotated list.

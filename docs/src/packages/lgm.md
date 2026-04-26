# LatentGaussianModels.jl

Latent Gaussian models — likelihoods, latent components, hyperpriors,
inference. The R-INLA-equivalent layer on top of `GMRFs.jl`.

## What's here

- **`AbstractLikelihood`** with closed-form implementations:
  `GaussianLikelihood`, `PoissonLikelihood`, `BinomialLikelihood`,
  `NegativeBinomialLikelihood`, `GammaLikelihood`. AD-based fallbacks
  for user-defined likelihoods.
- **`AbstractLatentComponent`**: `Intercept`, `FixedEffects`, `IID`,
  `RW1`, `RW2`, `AR1`, `Seasonal`, `Besag`, `BYM`, `BYM2`, `Leroux`,
  `Generic0`, `Generic1`. Adding a new component is "subtype the
  abstract type and implement five methods".
- **`AbstractHyperPrior`**: `PCPrecision`, `GammaPrecision`,
  `LogNormalPrecision`, `WeakPrior`, `PCBYM2Phi`, `LogitBeta`.
- **Inference strategies**: `EmpiricalBayes`, `Laplace`, `INLA`, with
  the convenience aliases `empirical_bayes`, `laplace`, `inla`. The
  default integration scheme is `:auto` (CCD for `dim(θ) > 2`,
  `Grid` otherwise — see ADR-010 in
  [`plans/decisions.md`](https://github.com/HaavardHvarnes/INLA/blob/main/plans/decisions.md)).
- **Diagnostics**: `dic`, `waic`, `cpo`, `pit`,
  `log_marginal_likelihood`.
- **`INLALogDensity`** — a `LogDensityProblems`-conformant view of the
  joint posterior, with a `LogDensityOrder{1}` gradient. Downstream
  samplers (Turing via [`LGMTuring.jl`](https://github.com/HaavardHvarnes/INLA/tree/main/packages/LGMTuring.jl),
  AdvancedHMC, custom) plug in here.

## Building a model

```julia
model = LatentGaussianModel(likelihood, (component₁, component₂, ...), A)
res   = inla(model, y)
```

`A` is the sparse projector mapping the stacked latent vector `x` to
the linear predictor `η = A x`. For per-observation effects, `A`
includes an identity block; for areal models it includes the
indicator-by-region block. A `build_projector` helper is on the
roadmap (Phase E2 follow-up).

## Component contract

```julia
graph(c)                    -> AbstractGMRFGraph
precision_matrix(c, θ)      -> SparseMatrixCSC
log_hyperprior(c, θ)        -> Real
initial_hyperparameters(c)  -> Vector
length(c)                   -> Int
```

Optional:

```julia
prior_mean(c, θ)            # default: zeros
constraints(c)              # default: NoConstraint()
```

## API

```@autodocs
Modules = [LatentGaussianModels]
```

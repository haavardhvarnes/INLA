# Getting started

This page builds the smallest end-to-end fit: a Poisson regression with
an intercept and an iid per-observation random effect, fit by INLA. It
exists to show the moving parts, not the modelling — for canonical
datasets, see the [vignettes](vignettes/scotland-bym2.md).

## A toy Poisson + IID model

`n = 200` observations with intercept `α` and an iid random effect
`u_i ∼ N(0, τ⁻¹)`:

```math
y_i \sim \mathrm{Poisson}(\exp(\alpha + u_i)),
\quad u \sim \mathcal{N}(0, \tau^{-1} I).
```

```@example getting-started
using Random, Distributions, SparseArrays, LinearAlgebra
using GMRFs, LatentGaussianModels

rng = Random.Xoshiro(20260423)
n   = 200

α_true = 0.5
τ_true = 4.0
u_true = (1 / sqrt(τ_true)) .* randn(rng, n)
η_true = α_true .+ u_true
y      = [rand(rng, Poisson(exp(η_true[i]))) for i in 1:n]
nothing # hide
```

## Build the model

A `LatentGaussianModel` has three pieces: the likelihood, a tuple of
latent components, and a sparse projector `A` mapping the stacked
latent vector `x = [α; u]` to the linear predictor `η = A x`.

```@example getting-started
c_int = Intercept()
c_iid = IID(n; hyperprior = PCPrecision(1.0, 0.01))

A = sparse([ones(n) Matrix{Float64}(I, n, n)])

ℓ     = PoissonLikelihood()
model = LatentGaussianModel(ℓ, (c_int, c_iid), A)
nothing # hide
```

The hyperprior `PCPrecision(U = 1.0, α = 0.01)` reads "1% prior
probability that the standard deviation exceeds 1". It is the
[Penalised Complexity prior](references.md) of Simpson et al. (2017)
and is the v0.1 default for precisions.

## Fit by INLA

```@example getting-started
res = inla(model, y; int_strategy = :grid)
nothing # hide
```

`inla(...)` is shorthand for `fit(model, y, INLA(); int_strategy =
:grid)`. For higher-dimensional `θ` the default `:auto` switches to
CCD; here `dim(θ) = 1` so `:grid` is fine.

## What's in the result

```@example getting-started
fixed_effects(model, res)        # posterior summary of α
```

```@example getting-started
hyperparameters(model, res)      # posterior summary of τ
```

```@example getting-started
log_marginal_likelihood(res)     # for model comparison
```

That's the loop. Real models add covariates (`FixedEffects(X)`), swap
`IID` for `BYM2(graph; …)` or `RW1(n)`, pick a richer likelihood
(`NegativeBinomialLikelihood`, `GammaLikelihood`, …), and turn on
diagnostics (`dic`, `waic`, `cpo`, `pit`). The
[Scotland BYM2 vignette](vignettes/scotland-bym2.md) walks through
that on a real dataset.

# LatentGaussianModels.jl

Bayesian inference for latent Gaussian models (LGMs) in Julia.

Implements the INLA algorithm — integrated nested Laplace
approximations — for the class of models

```
y_i | x, θ ~ likelihood(η_i(x), θ)
η        = A · x + offset
x | θ    ~ N(μ(θ), Q⁻¹(θ))        # latent Gaussian field
θ        ~ π(θ)                     # hyperpriors
```

with the likelihood being Gaussian, Poisson, Binomial, NegativeBinomial,
Gamma, or other exponential-family members, and the latent field being
a combination of `AbstractLatentComponent`s — IID, random walks, AR(1),
seasonal, Besag, BYM/BYM2, Leroux, Generic0/1, SPDE, and user-defined.

Built on top of [`GMRFs.jl`](../GMRFs.jl/). Inference options include
empirical Bayes (Laplace at θ̂), full INLA, and HMC (via a
`LogDensityProblems` bridge — see `LGMTuring.jl`).

## Status

`v0.1.0-rc1`. See the ecosystem [`CHANGELOG.md`](../../CHANGELOG.md)
for what landed in this line and the documented heavy-tail / weakly-ID
cases where R-INLA parity is loose.

Shipped:
- **Components**: `Intercept`, `FixedEffects`, `IID`, `RW1`, `RW2`,
  `AR1`, `Seasonal`, `Besag`, `BYM`, `BYM2`, `Leroux`, `Generic0`,
  `Generic1`. SPDE2 ships in [`INLASPDE.jl`](../INLASPDE.jl/).
- **Likelihoods**: `GaussianLikelihood`, `PoissonLikelihood`,
  `BinomialLikelihood`, `NegativeBinomialLikelihood`,
  `GammaLikelihood`. Closed-form gradients/Hessians on the inner
  Newton hot path; ForwardDiff fallback for user-defined cases.
- **Hyperpriors**: `PCPrecision`, `GammaPrecision`,
  `LogNormalPrecision`, `WeakPrior`, `PCBYM2Phi`, `LogitBeta`.
- **Inference**: `EmpiricalBayes`, `Laplace`, `INLA`. Aliases
  `empirical_bayes(model, y)`, `laplace(...)`, `inla(...)` over the
  dispatched `fit(model, y, strategy)`.
- **θ-integration**: `Grid`, `GaussHermite`, `CCD`. Default
  `int_strategy = :auto` chooses CCD for dim θ > 2, Grid otherwise.
- **Diagnostics**: `dic`, `waic`, `cpo`, `pit`.

## Quick example — Scotland lip-cancer BYM2

This is the actual API exercised by
[`test/oracle/test_scotland_bym2.jl`](test/oracle/test_scotland_bym2.jl):

```julia
using LatentGaussianModels, GMRFs, SparseArrays, LinearAlgebra

# Inputs: y (counts), E (expected), x (covariate), W (adjacency).
n = length(y)
ℓ      = PoissonLikelihood(; E = E)
c_int  = Intercept()                          # improper by default
c_beta = FixedEffects(1)
c_bym2 = BYM2(GMRFGraph(W); hyperprior_prec = PCPrecision(1.0, 0.01))

# Latent layout is [α; β; b; u]; only `b` enters η.
A = sparse(hcat(
    ones(n),                                  # intercept → α
    reshape(x, n, 1),                         # covariate → β
    Matrix{Float64}(I, n, n),                 # per-obs pick of b_i
    zeros(n, n),                              # u is constrained, off η
))

model = LatentGaussianModel(ℓ, (c_int, c_beta, c_bym2), A)
res   = inla(model, y; int_strategy = :grid)

fixed_effects(model, res)         # posterior summaries
hyperparameters(model, res)
log_marginal_likelihood(res)
```

For the smaller Gaussian / Gamma / Negative Binomial / Generic / Seasonal
patterns, see the matching files under
[`test/oracle/`](test/oracle/).

## Installation

Not yet on the General registry. Registered in a personal Julia
registry — add it once, then `Pkg.add` as usual:

```julia
using Pkg
Pkg.Registry.add(RegistrySpec(url = "https://github.com/haavardhvarnes/JuliaRegistry"))
Pkg.Registry.add("General")  # idempotent if already added
Pkg.add("LatentGaussianModels")
```

## See also

- [`GMRFs.jl`](../GMRFs.jl/) — sparse precision core.
- [`INLASPDE.jl`](../INLASPDE.jl/) — SPDE components on triangulated
  meshes.
- [`LGMTuring.jl`](../LGMTuring.jl/) — NUTS bridge for INLA-vs-MCMC
  triangulation.
- [`bench/oracle_compare.jl`](../../bench/oracle_compare.jl) —
  reproducible R-INLA parity benchmark over the full oracle suite.

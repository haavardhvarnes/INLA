# LatentGaussianModels.jl

Bayesian inference for latent Gaussian models (LGMs) in Julia.

Implements the INLA algorithm — integrated nested Laplace approximations —
for the class of models

```
y_i | x, θ ~ likelihood(η_i(x), θ)
η        = A · x + offset
x | θ    ~ N(μ(θ), Q⁻¹(θ))        # latent Gaussian field
θ        ~ π(θ)                     # hyperpriors
```

with the likelihood being Gaussian, Poisson, Binomial, NegativeBinomial,
Gamma, or other exponential-family members, and the latent field being a
combination of `AbstractLatentComponent`s — IID, random walks, AR(1),
Besag, BYM2, SPDE, and user-defined.

Built on top of [`GMRFs.jl`](../GMRFs.jl/). Inference options include
empirical Bayes, full INLA, and HMC (via a `LogDensityProblems` bridge to
Turing).

## Status

Planning. See [`plans/plan.md`](plans/plan.md).

## Quick example (target API)

```julia
using LatentGaussianModels, GMRFs, Distributions

# Scottish lip cancer BYM2
model = LatentGaussianModel(
    likelihood = Poisson(link = LogLink()),
    components = (
        Intercept(),
        FixedEffect(AFF),
        BYM2(graph = W, prior_prec = PCPrec(1.0, 0.01),
                        prior_phi  = PCPhi(0.5, 2/3)),
    ),
    offsets = log.(E),
)

fit = inla(model, observed_counts)

summary(fit)
marginal(fit, :AFF)
```

## See also

- [`GMRFs.jl`](../GMRFs.jl/) — sparse precision core.
- [`INLASPDE.jl`](../INLASPDE.jl/) — SPDE components on triangulated meshes.

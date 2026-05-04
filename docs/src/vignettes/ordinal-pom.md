# Ordinal regression — proportional-odds model (`POMLikelihood`)

The proportional-odds model (POM) is the workhorse of ordinal-response
analysis: a single set of regression coefficients `β` describes how
covariates shift the *cumulative* log-odds of being at or below each
class boundary. It is the cumulative-link cousin of the binary logit;
where binary logit has one cutoff at zero, POM has `K − 1` ordered
cutoffs `α_1 < α_2 < ⋯ < α_{K−1}` carving the latent scale into `K`
ordered classes.

The model dates back to McCullagh (1980) and is the default ordinal
treatment in standard texts (Agresti 2010, *Analysis of Ordinal
Categorical Data*; Tutz 2012, *Regression for Categorical Data*). This
vignette walks through a synthetic recovery, matching the R-INLA oracle
fixture under
[`test/oracle/test_synthetic_pom.jl`](https://github.com/HaavardHvarnes/INLA.jl/blob/main/packages/LatentGaussianModels.jl/test/oracle/test_synthetic_pom.jl).

## Model

For ordered response `y_i ∈ {1, …, K}` and linear predictor `η_i`
(without an intercept — the cut points absorb it):

```math
\Pr(y_i \le k \mid \eta_i, \alpha) \;=\; F(\alpha_k - \eta_i),
\qquad k = 1, \dots, K-1
```

with `α_0 = −∞`, `α_K = +∞`, and `F` the standard logistic CDF. Class
probabilities are differences:

```math
\Pr(y_i = k \mid \eta_i, \alpha)
  \;=\; F(\alpha_k - \eta_i) \;-\; F(\alpha_{k-1} - \eta_i).
```

The "proportional odds" name comes from the log-odds:

```math
\log \frac{\Pr(y_i \le k)}{\Pr(y_i > k)} \;=\; \alpha_k - \eta_i,
```

so a one-unit increase in any covariate shifts the cumulative log-odds
of being at or below *every* boundary by the same `β` — the
proportional-odds assumption.

### Internal-scale parameterisation

The `K − 1` cutpoints carry as likelihood hyperparameters on R-INLA's
internal scale:

```math
\theta_1 = \alpha_1, \qquad
\theta_k = \log(\alpha_k - \alpha_{k-1}),\;\; k = 2, \dots, K-1.
```

The increments `α_k − α_{k−1} = exp(θ_k)` are strictly positive,
guaranteeing the ordering for any `θ ∈ ℝ^{K−1}`. The
`POMLikelihood(K)` constructor exposes `K` as `n_classes`;
`nhyperparameters(ℓ)` returns `K − 1`.

### Prior on the cut points

R-INLA's `pom` family hard-wires a single Dirichlet prior on the class
probabilities implied by the cut points at `η = 0`:

```math
\pi_k(\alpha) \;=\; F(\alpha_k) - F(\alpha_{k-1}),
\qquad (\pi_1, \dots, \pi_K) \sim \text{Dirichlet}(\gamma, \dots, \gamma).
```

The single concentration `γ > 0` (`dirichlet_concentration`,
default 3 — matching R-INLA) is the only free prior parameter. All
`K − 1` `θ` slots inherit from this joint prior; you tune the shape
of the prior by tuning `γ` alone.

`γ → 1` is uniform on the simplex; large `γ` concentrates mass on the
balanced-class corner `π_k = 1/K`. The default `γ = 3` is a mild
preference for balanced cutoffs that still lets the data move them.

## Synthetic recovery

```@example pom
using Random, SparseArrays
using GMRFs, LatentGaussianModels
using LatentGaussianModels: POMLikelihood, FixedEffects,
                            LatentGaussianModel, inla,
                            fixed_effects, hyperparameters,
                            log_marginal_likelihood

rng        = MersenneTwister(20260505)
n          = 400
K          = 4
β_true     = 0.8
α_true     = (-1.0, 0.0, 1.2)            # cut points

x  = randn(rng, n)
η  = β_true .* x

# Sample y via the cumulative-logit construction:
# y_i = 1 + Σ_k 𝟙{α_k − η_i < logit(U_i)}, U_i ~ Uniform(0, 1).
function _sample_ordinal(rng, η, α, K)
    n = length(η)
    y = Vector{Int}(undef, n)
    for i in 1:n
        u = rand(rng); t = log(u / (1 - u))
        k = K
        for j in 1:(K - 1)
            if α[j] - η[i] >= t
                k = j; break
            end
        end
        y[i] = k
    end
    return y
end

y = _sample_ordinal(rng, η, α_true, K)
(class_counts = [count(==(k), y) for k in 1:K], β_true = β_true)
```

The class counts should be roughly balanced because the cut points are
symmetric around zero and the linear predictor has unit-variance support.

### Fit

The latent vector is the single `β` slot — POM has no intercept, so
the design matrix is just the `n × 1` covariate column.

```@example pom
ℓ      = POMLikelihood(K)                # default γ = 3, logit link
c_β    = FixedEffects(1)
A      = sparse(reshape(x, n, 1))
model  = LatentGaussianModel(ℓ, (c_β,), A)
res    = inla(model, y; int_strategy = :grid)

fe = fixed_effects(model, res)
hp = hyperparameters(model, res)
(β_julia      = fe[1].mean,
 β_sd         = fe[1].sd,
 β_true       = β_true,
 log_marginal = log_marginal_likelihood(res))
```

The recovered `β` should land within `±0.1` of the true `β = 0.8` at
this `n`. The `θ` posterior carries the cut-point information; you can
recover the cut points themselves by walking the parameterisation in
reverse.

### Recovering the cut points

```@example pom
θ̂  = [hp[k].mean for k in 1:(K - 1)]
α̂  = Vector{Float64}(undef, K - 1)
α̂[1] = θ̂[1]
for k in 2:(K - 1)
    α̂[k] = α̂[k - 1] + exp(θ̂[k])
end
(α̂ = α̂, α_true = collect(α_true))
```

`α̂` should track `α_true = (-1.0, 0.0, 1.2)` to within `~0.1` per
component. The mode and `0.025`/`0.975` quantiles for each `θ` slot
live on `hp[k]` if you want credible-interval bands on the cut points
directly.

## Choosing `dirichlet_concentration`

The default `γ = 3` is a mild preference for balanced classes and
matches R-INLA's default. Tune it via the keyword argument:

```@example pom
ℓ_uniform = POMLikelihood(K; dirichlet_concentration = 1.0)
ℓ_strong  = POMLikelihood(K; dirichlet_concentration = 10.0)
(uniform = ℓ_uniform.dirichlet_concentration,
 default = POMLikelihood(K).dirichlet_concentration,
 strong  = ℓ_strong.dirichlet_concentration)
```

- `γ = 1` is uniform on the simplex — every class-probability vector
  on `Δ^{K−1}` is equally likely a priori.
- `γ < 1` concentrates mass on the *boundary* of the simplex — i.e.
  prior favouring unbalanced classes (some classes near-empty). Use
  with care; the small-class data tends to be the limiting evidence
  for ordinal cut-point identifiability.
- `γ > 3` increasingly pulls toward the balanced corner. With `γ` very
  large, the prior dominates the data and cut points sit near
  `F^{-1}(k/K)` regardless of `y`.

In practice the default works well unless you have strong external
information about expected class balance.

## Marginal log-likelihood — note on R-INLA agreement

`POMLikelihood`'s `log_hyperprior` includes the full Jacobian of the
`θ → α → π` chain (`Σ log f(α_k) + Σ_{k≥2} θ_k`), making the prior
mathematically exact on the internal scale. R-INLA's documentation
(see `inla.doc("pom")`) records that R-INLA's internal Dirichlet
prior is *"correct only up to a multiplicative constant due to a
missing correction in the log-Jacobian for the sum-to-zero
constraint."*

The practical consequence: every posterior moment of `α`, `β`, and `θ`
matches between Julia and R-INLA, but the absolute marginal
log-likelihood reported by the two tools differs by a fixed
`θ`-independent additive constant. **Don't compare absolute mlik
values across the two engines** — use one engine throughout when
ranking models by mlik. Within either engine, mlik differences between
candidate POM models *are* directly comparable.

## Notes on extending the model

- **Random effects.** Add `IID`, `BYM2`, etc. components alongside
  `FixedEffects` — POM's projector is just the linear predictor, and
  any latent component composes. The cut points are likelihood
  hyperparameters and are independent of the latent structure.
- **Continuation-ratio / adjacent-category links.** Currently only
  the cumulative-logit link is supported, mirroring R-INLA's
  `family = "pom"`. Cumulative probit can be added later via
  dispatch on the `link` argument; the closed forms in `pom.jl` are
  specific to logistic-CDF arithmetic.
- **Boundaries.** When `y` is uniformly the lowest class, the
  gradient collapses to `∂η log F(α_1 − η) = −(1 − F(α_1 − η))` —
  identical to a binary logistic regression with intercept `α_1`
  and slope `−1` on `η`. Same applies symmetrically at the highest
  class.

The regression test
[`test/regression/test_likelihoods.jl`](https://github.com/HaavardHvarnes/INLA.jl/blob/main/packages/LatentGaussianModels.jl/test/regression/test_likelihoods.jl)
covers the closed-form derivative and Jacobian math; the R-INLA oracle
[`test/oracle/test_synthetic_pom.jl`](https://github.com/HaavardHvarnes/INLA.jl/blob/main/packages/LatentGaussianModels.jl/test/oracle/test_synthetic_pom.jl)
locks the posterior moments against R-INLA on a synthetic ordinal
fixture.

# Joint longitudinal + survival via `Copy`

Joint models pair a Gaussian longitudinal arm and a survival arm
through a shared random effect. Following the parameterisation of
Baghfalaki et al. (2024), the latent subject effect `b_i` enters both
arms — additively in the longitudinal mean and as a `Copy`-scaled term
in the survival linear predictor. INLA.jl's multi-likelihood
`LatentGaussianModel` plus the `Copy` component handle the
construction.

This vignette is a synthetic-recovery walkthrough. The data-generating
process matches the regression test
[`test/regression/test_inla_joint_baghfalaki.jl`](https://github.com/HaavardHvarnes/INLA.jl/blob/main/packages/LatentGaussianModels.jl/test/regression/test_inla_joint_baghfalaki.jl)
and the R-INLA oracle [`test/oracle/test_synthetic_baghfalaki.jl`](https://github.com/HaavardHvarnes/INLA.jl/blob/main/packages/LatentGaussianModels.jl/test/oracle/test_synthetic_baghfalaki.jl).
See [`docs/src/vignettes/coxph-weibull-survival.md`](coxph-weibull-survival.md)
for the standalone Weibull arm and
[`ADR-021`](../../../plans/decisions.md) for the design rationale that
puts the `Copy` β-scaling on the receiving likelihood.

## Model

For subjects `i = 1, …, N` with `K` longitudinal measurements per
subject and a single survival outcome:

```math
\begin{aligned}
b_i        &\sim \mathcal{N}(0,\, \sigma_b^2),                                                  \\
y_{i,j}    &\sim \mathcal{N}(\alpha_{\text{long}} + b_i,\, \sigma_g^2),  &j &= 1, \dots, K, \\
T_i        &\sim \text{Weibull}\big(\alpha_w,\, \exp(\alpha_{\text{surv}} + \varphi \cdot b_i)\big),
\end{aligned}
```

with right-censoring at an administrative cutoff `T_{\text{admin}}`.
The latent vector is `x = [α_long; α_surv; b_1, …, b_N]`, length `N + 2`.

| Parameter            | Symbol                  | Role                                                  |
|----------------------|-------------------------|-------------------------------------------------------|
| Longitudinal intercept | `α_long`             | Mean of `y_{i,j}` after subject-shift                 |
| Survival intercept     | `α_surv`             | Baseline log-hazard rate                              |
| Longitudinal noise SD  | `σ_g`                | Within-subject measurement noise                      |
| Random-effect SD       | `σ_b`                | Across-subject heterogeneity                          |
| Weibull shape          | `α_w`                | Survival hazard shape (PH parameterisation)           |
| **Copy scaling**       | **`φ`**              | How strongly `b_i` modulates the survival arm's hazard |

`φ` is the load-bearing parameter that ties the two arms together;
`φ = 0` recovers two independent fits.

## Simulate

```@example baghfalaki
using Random, SparseArrays, LinearAlgebra
using GMRFs, LatentGaussianModels
using LatentGaussianModels: Censoring, NONE, RIGHT,
                            CopyTargetLikelihood, Copy,
                            LinearProjector, StackedMapping,
                            log_marginal_likelihood,
                            n_likelihoods, n_hyperparameters

rng = MersenneTwister(20260502)

N           = 80    # subjects
K           = 5     # longitudinal observations per subject
α_long_true = 0.4
α_surv_true = -0.2
σ_b_true    = 0.7
σ_g_true    = 0.5
shape_true  = 1.5
φ_true      = 0.8
T_admin     = 3.5

b_true   = σ_b_true .* randn(rng, N)
y_long   = Float64[]
subj_idx = Int[]
for i in 1:N
    for _ in 1:K
        push!(y_long, α_long_true + b_true[i] + σ_g_true * randn(rng))
        push!(subj_idx, i)
    end
end
n_long = length(y_long)

T_event   = Vector{Float64}(undef, N)
censoring = Vector{Censoring}(undef, N)
for i in 1:N
    λ_i = exp(α_surv_true + φ_true * b_true[i])
    u   = rand(rng)
    t_i = (-log(u) / λ_i)^(1 / shape_true)
    if t_i > T_admin
        T_event[i]   = T_admin
        censoring[i] = RIGHT
    else
        T_event[i]   = t_i
        censoring[i] = NONE
    end
end
n_cens = count(==(RIGHT), censoring)

(N = N, K = K, n_long = n_long, n_cens = n_cens,
 censoring_pct = round(100 * n_cens / N; digits = 1))
```

## Build the model

The latent layout is `x = [α_long; α_surv; b_1, …, b_N]`. Two
`LinearProjector` blocks pick the active components for each arm; the
`StackedMapping` glues them together. The Weibull arm's `α_surv` is
selected through its `A_w`, while the subject effect `b_i` enters the
hazard through `Copy(3:(2+N); β_init = 1.0, fixed = false)` —
introducing a single new hyperparameter `φ` for the joint scaling.

```@example baghfalaki
A_g = spzeros(Float64, n_long, 2 + N)
for r in 1:n_long
    A_g[r, 1]               = 1.0  # α_long
    A_g[r, 2 + subj_idx[r]] = 1.0  # b_{subj_idx[r]}
end

A_w = spzeros(Float64, N, 2 + N)
for i in 1:N
    A_w[i, 2] = 1.0  # α_surv
end

mapping = StackedMapping(
    (LinearProjector(A_g), LinearProjector(A_w)),
    [1:n_long, (n_long + 1):(n_long + N)])

ℓ_g      = GaussianLikelihood()
ℓ_w_base = WeibullLikelihood(censoring = censoring)
ℓ_w      = CopyTargetLikelihood(
    ℓ_w_base,
    Copy(3:(2 + N); β_init = 1.0, fixed = false))

model = LatentGaussianModel(
    (ℓ_g, ℓ_w),
    (Intercept(), Intercept(), IID(N)),
    mapping)

(n_likelihoods = n_likelihoods(model),
 n_hyperparameters = n_hyperparameters(model))
```

The hyperparameter layout is
`θ = [log τ_g, log α_w, φ, log τ_b]` — Gaussian precision, Weibull
shape, Copy β, and IID precision in that order.

## Fit

```@example baghfalaki
y   = vcat(y_long, T_event)
res = inla(model, y; int_strategy = :grid)
nothing # hide
```

## Fixed effects

```@example baghfalaki
fe = fixed_effects(model, res)
(α_long_julia = fe[1].mean, α_long_true = α_long_true,
 α_surv_julia = fe[2].mean, α_surv_true = α_surv_true)
```

## Random-effect recovery

The cleanest summary of the joint model's identifiability is the
correlation between the recovered subject effects `b̂_i` and the
data-generating `b_true`:

```@example baghfalaki
using Statistics: cor

re_b = random_effects(model, res)["IID[3]"]
(cor_b̂_b_true = cor(re_b.mean, b_true),)
```

Values above 0.85 indicate the longitudinal arm is informing the
random-effects posterior much more strongly than its own marginal
structure would (`σ_b ≈ 0.7` vs. `σ_g ≈ 0.5` per single observation,
with `K = 5` repeats), and the survival arm is consistent with that
shape via the `Copy` link.

## Hyperparameter recovery

```@example baghfalaki
τ_g_J  = exp(res.θ̂[1])
α_w_J  = exp(res.θ̂[2])
φ_J    = res.θ̂[3]
τ_b_J  = exp(res.θ̂[4])
σ_g_J  = 1 / sqrt(τ_g_J)
σ_b_J  = 1 / sqrt(τ_b_J)

(σ_g_julia = σ_g_J, σ_g_true = σ_g_true,
 σ_b_julia = σ_b_J, σ_b_true = σ_b_true,
 α_w_julia = α_w_J, α_w_true = shape_true,
 φ_julia   = φ_J,   φ_true   = φ_true)
```

The Copy scaling `φ` is the load-bearing parameter — recovering it
within `±0.3` of the true `0.8` on `n = 80` subjects with ~25%
censoring confirms that the β-scaled latent share works end-to-end on
a multi-block joint model. The Weibull shape `α_w` is independent of
the Copy mechanism; its posterior band reflects the inherent spread on
this number of events rather than anything about the joint
construction.

## Notes on extending the model

- **Multiple longitudinal measurements per visit.** Stack additional
  Gaussian (or non-Gaussian) likelihood blocks via `StackedMapping`,
  each with its own `LinearProjector` selecting the relevant latent
  columns; one `Copy` per block tied to the same IID component reuses
  `b_i` everywhere.
- **Different survival likelihood.** Swap the `WeibullLikelihood` for
  any other censoring-aware survival likelihood
  (`LognormalSurvLikelihood`, `GammaSurvLikelihood`, the Cox PH
  augmentation, …); the `Copy` machinery is agnostic to the receiving
  likelihood.
- **Fixed `β`.** Pass `Copy(...; β_init = β₀, fixed = true)` to drop
  the Copy hyperparameter and pin the scaling. Useful when `φ` is
  identified externally (e.g. from a calibration study) or for
  sensitivity analysis around a focal value.
- **Non-Gaussian longitudinal arm.** Replace `GaussianLikelihood()`
  with `PoissonLikelihood`, `BinomialLikelihood`, etc. — the
  multi-likelihood pipeline handles arbitrary mixed families on each
  block.

R-INLA's `inla.surv` multi-likelihood machinery covers a similar slice
of this pattern; the oracle test
[`test/oracle/test_synthetic_baghfalaki.jl`](https://github.com/HaavardHvarnes/INLA.jl/blob/main/packages/LatentGaussianModels.jl/test/oracle/test_synthetic_baghfalaki.jl)
asserts agreement with R-INLA on the same dataset (fixed effects,
hyperparameters, random-effect posteriors). The marginal log-likelihood
inherits the polynomial-form Laplace gap from the standalone Weibull
arm — closure tracked separately as post-v0.1 work.

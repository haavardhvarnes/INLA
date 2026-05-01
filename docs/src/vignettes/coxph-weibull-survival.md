# Survival — Cox PH and Weibull on synthetic data

Two parameterisations of right-censored survival regression sit side
by side in `LatentGaussianModels.jl`:

- **Cox PH.** Hazard `λ_i(t) = λ_0(t) · exp(xᵀ β)` with a piecewise-
  constant baseline log-hazard given an RW1 smoothing prior. Implemented
  via the Holford / Laird-Olivier piecewise-exponential-as-Poisson
  augmentation: each subject is split across baseline intervals into
  augmented Poisson rows (`y_{ik}`, exposure `E_{ik}`).
- **Weibull AFT/PH.** Likelihood
  `f(t | η, α) = α · t^{α-1} · μ^α · exp(-(μ t)^α)` with `μ = exp(η)`
  and a single shape hyperparameter `α`. No augmentation; the censoring-
  aware log-density is closed-form.

Both inherit the latent-component vocabulary (`Intercept`,
`FixedEffects`, …) and run through the same `inla(...)` entry point.
This vignette walks each through end-to-end against R-INLA.

The fixtures driving this page also drive the regression tests
[`test/oracle/test_synthetic_coxph.jl`](https://github.com/HaavardHvarnes/INLA.jl/blob/main/packages/LatentGaussianModels.jl/test/oracle/test_synthetic_coxph.jl)
and
[`test/oracle/test_synthetic_weibull_survival.jl`](https://github.com/HaavardHvarnes/INLA.jl/blob/main/packages/LatentGaussianModels.jl/test/oracle/test_synthetic_weibull_survival.jl).

## Cox proportional hazards

The synthetic Cox PH fixture has `n = 400` subjects, two covariates
`x_1, x_2 ∼ N(0, 1)`, true slopes `β = (0.5, -0.3)`, and a
piecewise-exponential baseline with ten intervals on `[0, 5]`. Censoring
times are `Uniform(2.5, 6.5)`, yielding ~50–60% events.

```@example coxph
using JLD2, SparseArrays, LinearAlgebra
using GMRFs, LatentGaussianModels

const COXPH_FX = joinpath(@__DIR__, "..", "..", "..",
    "packages", "LatentGaussianModels.jl",
    "test", "oracle", "fixtures", "synthetic_coxph.jld2")

fx_cox = jldopen(COXPH_FX, "r") do f
    f["fixture"]
end

inp = fx_cox["input"]
time  = Float64.(inp["time"])
event = Int.(inp["event"])
X     = Float64.(inp["X"])
n     = length(time)
nothing # hide
```

`inla_coxph(time, event)` performs the augmentation: each subject
contributes one Poisson row per baseline interval they enter, with
exposure equal to the time spent in that interval. The breakpoints are
quantile-based by default.

```@example coxph
aug = inla_coxph(time, event)
(n_subjects = aug.n_subjects, n_intervals = aug.n_intervals,
 n_aug_rows = length(aug.y), total_events = sum(aug.y))
```

The latent layout is `[γ; β]` — `γ` is the baseline log-hazard step
heights (one per interval, RW1-smoothed) and `β` is the covariate slope
block. `coxph_design(aug, X)` assembles the joint design.

```@example coxph
ℓ          = PoissonLikelihood(E = aug.E)
c_baseline = RW1(aug.n_intervals; hyperprior = PCPrecision(1.0, 0.01))
c_beta     = FixedEffects(2)
A          = coxph_design(aug, X)

model_cox = LatentGaussianModel(ℓ, (c_baseline, c_beta), A)
res_cox   = inla(model_cox, aug.y)
nothing # hide
```

The covariate posterior:

```@example coxph
random_effects(model_cox, res_cox)["FixedEffects[2]"]
```

Compared to R-INLA's `family = "coxph"` fit on the same data:

```@example coxph
sf      = fx_cox["summary_fixed"]
β_R     = Float64.(sf["mean"])
β_sd_R  = Float64.(sf["sd"])

re      = random_effects(model_cox, res_cox)
β_J     = re["FixedEffects[2]"].mean
β_sd_J  = re["FixedEffects[2]"].sd

(rownames = sf["rownames"],
 β_julia  = β_J, β_R = β_R,
 sd_julia = β_sd_J, sd_R = β_sd_R)
```

The covariate slopes match within ≈1.5 R-INLA posterior SDs on this
fixture. The marginal log-likelihood differs by an η-independent
augmentation constant `Σ_{events} log E_{k_last,i}` that cancels in
posterior inference; we don't compare it. The baseline-hazard component
is RW1-smoothed on each side's own cutpoint grid (R-INLA equispaced,
ours quantile-based), so the per-knot baseline values aren't directly
comparable either — both produce statistically equivalent fits to the
covariate effects.

## Weibull survival

The synthetic Weibull fixture uses `n = 200`, intercept + one
covariate, true intercept `α = -0.5`, slope `β = 0.7`, shape
`α_w = 1.5`, and ~25% right-censoring.

```@example weibull
using JLD2, SparseArrays, LinearAlgebra
using GMRFs, LatentGaussianModels
using LatentGaussianModels: NONE, RIGHT

const WB_FX = joinpath(@__DIR__, "..", "..", "..",
    "packages", "LatentGaussianModels.jl",
    "test", "oracle", "fixtures", "synthetic_weibull_survival.jld2")

fx_wb = jldopen(WB_FX, "r") do f
    f["fixture"]
end

inp   = fx_wb["input"]
time  = Float64.(inp["time"])
event = Int.(inp["event"])
xcov  = Float64.(inp["x"])
n     = length(time)
nothing # hide
```

`Censoring` is a per-row enum on the likelihood struct. `NONE` marks an
observed event, `RIGHT` marks right-censoring; `LEFT` and `INTERVAL` are
also supported.

```@example weibull
cens = Censoring[e == 1 ? NONE : RIGHT for e in event]
ℓ    = WeibullLikelihood(censoring = cens)
A    = sparse(hcat(ones(n), reshape(xcov, n, 1)))

model_wb = LatentGaussianModel(ℓ, (Intercept(), FixedEffects(1)), A)
res_wb   = inla(model_wb, time)
nothing # hide
```

Fixed effects:

```@example weibull
fixed_effects(model_wb, res_wb)
```

Shape hyperparameter on the user scale (`α_w = exp(θ̂)`):

```@example weibull
α_w_julia = exp(res_wb.θ_mean[1])
α_w_sd_julia = sqrt(res_wb.Σθ[1, 1]) * exp(res_wb.θ̂[1])
(α_w_julia = α_w_julia, α_w_sd_julia = α_w_sd_julia)
```

Compared to R-INLA's `family = "weibullsurv"` (variant 0, Weibull-PH
parameterisation):

```@example weibull
sf      = fx_wb["summary_fixed"]
sh      = fx_wb["summary_hyperpar"]
α_R     = Float64(sf["mean"][1])
β_R     = Float64(sf["mean"][2])
α_w_R   = Float64(sh["mean"])

fe      = fixed_effects(model_wb, res_wb)

(intercept_julia = fe[1].mean, intercept_R = α_R,
 slope_julia    = fe[2].mean, slope_R    = β_R,
 α_w_julia      = α_w_julia,  α_w_R      = α_w_R)
```

Fixed effects agree within 5% relative; the shape posterior matches
within 10%. The marginal log-likelihood reported by R-INLA's
`weibullsurv` family differs from our integrated `mlik` by ~10 nats on
this fixture — likely R-INLA's internal Gaussian normalising constant
for this family. Tracked as a v0.2 calibration item; the posterior
agreement above is what the user sees.

## Picking a parameterisation

| Use Cox PH when… | Use Weibull when… |
|---|---|
| Baseline hazard shape is unknown or non-monotone | Hazard is plausibly monotone (early-failure or aging populations) |
| Number of events is moderate-to-large (RW1 smoothing has informative segments) | `n` is small or the dataset has few events |
| You want a model-free baseline ("non-parametric in time") | You want a single shape hyperparameter and tighter prediction intervals |

Both inherit the rest of the LGM stack — random effects (RW1, AR1,
Besag, BYM2, …), constraints, multi-likelihood blocks, and the
`LogDensityProblems` seam for downstream samplers.

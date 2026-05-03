# Measurement-error regression — `MEB` and `MEC`

Errors-in-variables regression covers the case where a covariate `x` is
not observed cleanly: instead the analyst sees a noisy proxy `w`. The
two canonical noise structures are

- **Berkson** (`MEB`): the latent truth is `x = w + u`, the proxy `w` is
  fixed (e.g. a calibration-machine readout assigned to a person), and
  the noise `u` carries through into `x`.
- **Classical** (`MEC`): the latent truth is `x` itself, and the proxy
  `w = x + u` is the analyst's noisy observation.

Naïvely regressing `y` on `w` produces consistent estimates under
Berkson noise but **attenuated** slope estimates under classical noise
(Carroll, Ruppert, Stefanski, Crainiceanu 2006). This vignette walks
through both flavours; the focal example is a Carroll-style classical
fit recovering the unattenuated slope through `MEC`.

The components were locked by [ADR-023](../../../plans/decisions.md);
their R-INLA equivalents are `f(w, model = "meb", ...)` and
`f(w, model = "mec", ...)`. β-scaling is supplied by a `Copy` on the
receiving likelihood per [ADR-021](../../../plans/decisions.md), the
same pattern as
[`docs/src/vignettes/joint-longitudinal-survival.md`](joint-longitudinal-survival.md).

## Berkson (`MEB`) — `x = w + u`

The regression is `y_i = α + β · x_i + ε_i` with `x_i = w_i + u_i`,
`u_i ~ N(0, τ_u⁻¹)`. The Berkson tie pushes the noise *into* the
latent — the conditional `x | w` has prior mean `w` and precision
`τ_u`. `MEB` carries the prior; `Copy` on the receiving Gaussian
likelihood carries `β`.

### Simulate

```@example measurement_error
using Random, SparseArrays, LinearAlgebra
using GMRFs, LatentGaussianModels
using LatentGaussianModels: CopyTargetLikelihood, Copy, GaussianPrior,
                            LinearProjector, log_marginal_likelihood,
                            hyperparameters, fixed_effects

rng       = MersenneTwister(20260503)
n         = 80
α_true    = 0.2
β_true    = 0.7
σ_y_true  = 0.3
τ_u_true  = 4.0     # Berkson noise precision

w         = randn(rng, n)                                 # observed proxy
x_true    = w .+ randn(rng, n) ./ sqrt(τ_u_true)          # latent truth
y         = α_true .+ β_true .* x_true .+ σ_y_true .* randn(rng, n)

(n = n, β_true = β_true, τ_u_true = τ_u_true)
```

### Fit

The latent vector is `[α; x_1, …, x_n]`. The intercept enters `η`
through column 1 of the projector; `MEB` carries the per-slot prior
`x ~ N(w, τ_u⁻¹)` for slots `2:(n+1)`; `Copy(2:(n+1))` injects
`β · x_i` into the linear predictor through the receiving likelihood.

```@example measurement_error
α        = Intercept()
c_meb    = MEB(w; τ_u_init = log(τ_u_true))
A_meb    = sparse(hcat(ones(n), zeros(n, n)))   # only the intercept here
ℓ_meb    = CopyTargetLikelihood(
    GaussianLikelihood(),
    Copy(2:(n + 1); β_prior = GaussianPrior(1.0, 0.5),
        β_init = 1.0, fixed = false))
m_meb    = LatentGaussianModel(ℓ_meb, (α, c_meb), LinearProjector(A_meb))
res_meb  = inla(m_meb, y; int_strategy = :grid)

hp_meb   = hyperparameters(m_meb, res_meb)
fe_meb   = fixed_effects(m_meb, res_meb)
(α_julia = fe_meb[1].mean, α_true = α_true,
 β_julia = hp_meb[2].mean, β_true = β_true,
 log_marginal = log_marginal_likelihood(res_meb))
```

`hp_meb[2]` is the `Copy`'s β slot (label `"likelihood[2]"` in the
canonical layout). `MEB`'s only own hyperparameter `log τ_u` lives at
`hp_meb[3]` — left at the supplied initial value here since with
`n = 80` the data does not move it materially from the seed value.

## Classical (`MEC`) — `w = x + u` and slope attenuation

The naïve regression of `y` on `w` produces a biased slope:

```math
\widehat{\beta}_{\text{naïve}} \;\to\; \beta \cdot \frac{\tau_u}{\tau_u + \tau_x}
```

as `n → ∞`. The factor on the right is < 1 whenever `τ_u` is finite.
`MEC` undoes the attenuation by carrying the conjugate-Gaussian prior
`x ~ N(μ̂(θ), Q̂(θ)⁻¹)` with

```math
\hat{Q}(\theta) = \tau_x I + \tau_u D, \qquad
\hat{\mu}(\theta) = \hat{Q}(\theta)^{-1}\,(\tau_x \mu_x \mathbf{1} + \tau_u D \cdot w),
```

so that the latent `x` is the conjugate-Bayesian update of the
prior `x ~ N(μ_x, τ_x⁻¹)` by the noisy observation `w`. R-INLA's
default has all three of `(τ_u, μ_x, τ_x)` *fixed* — the analyst
supplies them from a calibration study or sensitivity range — and
the model degrades gracefully to plain regression unless one is
unfixed.

### Simulate (Carroll-style)

```@example measurement_error
n_c        = 120
τ_u_c_true = 25.0    # classical-error precision (large ⇒ small noise)
τ_x_c_true = 1.0     # truth-distribution precision
μ_x_c_true = 0.0
α_c_true   = -0.3
β_c_true   = 0.6
σ_y_c_true = 0.25

x_c_true   = μ_x_c_true .+ randn(rng, n_c) ./ sqrt(τ_x_c_true)
w_c        = x_c_true .+ randn(rng, n_c) ./ sqrt(τ_u_c_true)
y_c        = α_c_true .+ β_c_true .* x_c_true .+
             σ_y_c_true .* randn(rng, n_c)

(τ_u_c_true = τ_u_c_true, τ_x_c_true = τ_x_c_true,
 attenuation_factor = τ_u_c_true / (τ_u_c_true + τ_x_c_true))
```

### Naïve baseline — regress `y` on `w` directly

```@example measurement_error
X_naive          = hcat(ones(n_c), w_c)
β_naive          = X_naive \ y_c
attenuation_pred = β_c_true * τ_u_c_true / (τ_u_c_true + τ_x_c_true)

(β_naive_intercept = β_naive[1],
 β_naive_slope     = β_naive[2],
 β_true            = β_c_true,
 attenuation_pred  = attenuation_pred)
```

The naïve OLS slope sits near `attenuation_pred`, not `β_c_true` — the
classical signature.

### `MEC` fit — slope recovered

```@example measurement_error
c_mec    = MEC(w_c;
    τ_u_init = log(τ_u_c_true),
    μ_x_init = μ_x_c_true,
    τ_x_init = log(τ_x_c_true))
A_mec    = sparse(hcat(ones(n_c), zeros(n_c, n_c)))
ℓ_mec    = CopyTargetLikelihood(
    GaussianLikelihood(),
    Copy(2:(n_c + 1); β_prior = GaussianPrior(1.0, 0.5),
        β_init = 1.0, fixed = false))
m_mec    = LatentGaussianModel(ℓ_mec, (Intercept(), c_mec),
    LinearProjector(A_mec))
res_mec  = inla(m_mec, y_c; int_strategy = :grid)

hp_mec   = hyperparameters(m_mec, res_mec)
fe_mec   = fixed_effects(m_mec, res_mec)
(α_julia      = fe_mec[1].mean, α_true = α_c_true,
 β_julia      = hp_mec[2].mean, β_true = β_c_true,
 log_marginal = log_marginal_likelihood(res_mec))
```

The recovered β should land near `β_c_true = 0.6`, well above the
attenuated naïve estimate. With `n_c = 120` and noise calibrated as
above the recovery is within `±0.1` of the truth on most seeds.

## Choosing the `fix_*` toggles

`MEC`'s three slots `(τ_u, μ_x, τ_x)` are all default-fixed because in
practice each one needs an external information source to be
identifiable on a single regression slice:

| Slot     | Default | Unfix when                                                  |
|----------|---------|-------------------------------------------------------------|
| `τ_u`    | fixed   | A calibration / replicate-measurement study supplies a prior |
| `μ_x`    | fixed   | The analyst is unsure of the latent location and has weak prior |
| `τ_x`    | fixed   | The analyst is willing to model the latent's variability      |

Unfixing `τ_x` while leaving `τ_u` fixed corresponds to Muff et al.
2015's "structural" model variant. Unfixing `τ_u` makes the model
identification depend critically on either replicate measurements per
subject or the prior on `τ_u`.

### Example — unfix `τ_x` (estimate the truth's variability)

```@example measurement_error
c_mec_τx_free = MEC(w_c;
    τ_u_init = log(τ_u_c_true),
    μ_x_init = μ_x_c_true,
    τ_x_init = log(τ_x_c_true),
    fix_τ_x  = false)
m_mec2    = LatentGaussianModel(
    CopyTargetLikelihood(
        GaussianLikelihood(),
        Copy(2:(n_c + 1); β_prior = GaussianPrior(1.0, 0.5),
            β_init = 1.0, fixed = false)),
    (Intercept(), c_mec_τx_free),
    LinearProjector(A_mec))
res_mec2  = inla(m_mec2, y_c; int_strategy = :grid)

hp_mec2   = hyperparameters(m_mec2, res_mec2)
τ_x_row   = hp_mec2[end]    # last hyperparameter = `MEC[2]` log τ_x
(τ_x_julia = exp(τ_x_row.mean), τ_x_true = τ_x_c_true,
 β_julia = hp_mec2[2].mean, β_true = β_c_true)
```

`τ_x_julia` should land near `τ_x_c_true = 1.0`, and the slope should
remain close to `β_c_true`.

## Notes on extending the model

- **Multiple noisy covariates.** Stack one `MEC` (or `MEB`) per
  proxied covariate alongside an `Intercept()` and any clean
  covariates. Each measurement-error component contributes its own
  latent block; one `Copy` per receiving block injects its β-scaled
  share into the linear predictor.
- **Non-Gaussian outcomes.** Swap `GaussianLikelihood()` for
  `PoissonLikelihood`, `BinomialLikelihood`, or any other family —
  `Copy` is agnostic to the receiving likelihood (see
  [`vignettes/joint-longitudinal-survival.md`](joint-longitudinal-survival.md)
  for the Weibull-survival case).
- **Calibration-study informed priors.** Replace the default
  `GammaPrecision(1.0, 1e-4)` on `τ_u` with a tighter prior derived
  from replicate measurements. The resulting `MEC` fit is then a true
  errors-in-variables update rather than a sensitivity analysis at a
  single fixed `τ_u`.

The regression tests
[`test/regression/test_meb.jl`](https://github.com/HaavardHvarnes/INLA.jl/blob/main/packages/LatentGaussianModels.jl/test/regression/test_meb.jl)
and
[`test/regression/test_mec.jl`](https://github.com/HaavardHvarnes/INLA.jl/blob/main/packages/LatentGaussianModels.jl/test/regression/test_mec.jl)
cover the closed-form math behind both components, including the
conjugate-Gaussian prior-mean formula for `MEC`. R-INLA oracles for
both flavours are tracked separately under Phase I-B in
[`plans/phase-i-and-onwards-mighty-emerson.md`](../../../plans/phase-i-and-onwards-mighty-emerson.md).

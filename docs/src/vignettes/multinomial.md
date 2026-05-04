# Multinomial regression — independent-Poisson reformulation

Multinomial-logit regression with `K` classes does not appear as a
first-class likelihood in either R-INLA or this package. Both engines
fit it through the **Multinomial-Poisson trick** (Baker 1994; Chen
1985): for row `i` with `K`-vector of counts `Y_i ~ Multinomial(N_i,
π_i)` and per-class linear predictor `η_ik = x_i^⊤ β_k`, the
likelihood factorises as

```math
Y_{ik} \mid \alpha_i \;\stackrel{\text{ind.}}{\sim}\; \text{Poisson}(\lambda_{ik}),
\qquad \lambda_{ik} = \exp(\alpha_i + x_i^\top \beta_k),
```

where `α_i` is a per-row nuisance intercept that absorbs the row-sum
information `N_i`. Conditional on `(α_i)_{i=1}^{n}` set to a
fixed-precision IID prior (R-INLA's recipe:
`prec = list(initial = -10, fixed = TRUE)`), the posterior on the `β`
coefficients matches the original multinomial regression bit-for-bit.
Reference-class identifiability comes from dropping the
reference-class block (corner-point parameterisation, Agresti 2010
§8.5): `β_K = 0`.

This vignette walks through synthetic recovery of `β` against the
R-INLA oracle fixture under
[`test/oracle/test_synthetic_multinomial.jl`](https://github.com/HaavardHvarnes/INLA.jl/blob/main/packages/LatentGaussianModels.jl/test/oracle/test_synthetic_multinomial.jl).
The decision to use the independent-Poisson route as the default is
recorded in ADR-024.

## Building blocks

LatentGaussianModels.jl exposes two small helpers that turn an
`(n_rows, K)` count matrix into the long-format Poisson layout:

- [`multinomial_to_poisson`](@ref) — reshapes `Y` into the long-format
  triple `(y, row_id, class_id)` with row-major layout
  `idx = (i - 1) * K + k`.
- [`multinomial_design_matrix`](@ref) — builds the class-specific
  covariate block of the long-format design matrix; drops the
  reference-class block to identify the model.

The per-row nuisance `α_i` is attached as an `IID(n_rows; τ_init,
fix_τ)` component with `fix_τ = true`. The `IID` constructor's
`τ_init` and `fix_τ` keyword arguments mirror R-INLA's
`hyper = list(prec = list(initial = ..., fixed = TRUE))` directly.

## Synthetic recovery

```@example multinomial
using Random, SparseArrays, Distributions
using GMRFs, LatentGaussianModels
using LatentGaussianModels: PoissonLikelihood, FixedEffects, IID,
                            LatentGaussianModel, inla, component_range,
                            log_marginal_likelihood,
                            multinomial_to_poisson,
                            multinomial_design_matrix

rng       = MersenneTwister(20260504)
n_rows    = 200
K         = 3
p         = 1
N_trials  = 5
β_true    = (0.7, -0.4)         # β_1, β_2; β_3 = 0 (reference)

x = randn(rng, n_rows)

# Sample multinomial counts row-by-row.
Y = zeros(Int, n_rows, K)
for i in 1:n_rows
    η   = (β_true[1] * x[i], β_true[2] * x[i], 0.0)
    p_i = exp.(η .- maximum(η)); p_i ./= sum(p_i)
    Y[i, :] .= rand(rng, Multinomial(N_trials, collect(p_i)))
end
(class_totals = sum(Y, dims = 1), N_total = sum(Y))
```

### Long-format layout + design matrix

```@example multinomial
helper = multinomial_to_poisson(Y)
X      = reshape(x, n_rows, p)
A_β    = multinomial_design_matrix(helper, X)
A_α    = sparse(1:helper.n_long, helper.row_id, ones(helper.n_long),
                helper.n_long, n_rows)
A      = hcat(A_β, A_α)
size(A)
```

`A_β` carries the `(K - 1) * p = 2` `β` columns (corner-point: β_3 is
dropped); `A_α` is the indicator matrix that maps each long-format
observation to its row's nuisance intercept.

### Fit

```@example multinomial
ℓ      = PoissonLikelihood()
c_β    = FixedEffects((K - 1) * p)
c_α    = IID(n_rows; τ_init = -10.0, fix_τ = true)
model  = LatentGaussianModel(ℓ, (c_β, c_α), A)

res    = inla(model, helper.y)

β_rng  = component_range(model, 1)
β_mean = res.x_mean[β_rng]
β_sd   = sqrt.(res.x_var[β_rng])
(β_julia      = β_mean,
 β_sd         = β_sd,
 β_true       = collect(β_true),
 log_marginal = log_marginal_likelihood(res))
```

The recovered `β` should sit within ~5% of R-INLA's posterior mean and
its standard deviation within ~10% — the oracle test
[`test/oracle/test_synthetic_multinomial.jl`](https://github.com/HaavardHvarnes/INLA.jl/blob/main/packages/LatentGaussianModels.jl/test/oracle/test_synthetic_multinomial.jl)
locks those tolerances against the R-INLA fixture.

## Notes on the dim(θ) = 0 fast path

The model above has *zero* hyperparameters: `PoissonLikelihood` has
none, the `FixedEffects` block has none, and `IID(n_rows; τ_init =
-10.0, fix_τ = true)` has none (the precision is hard-coded). The INLA
fast path detects `n_hyperparameters(model) == 0` and skips both the
θ-mode optimisation and the θ-grid integration — a single Laplace
approximation at `θ = Float64[]` is the exact posterior, and the
returned `INLAResult` carries an empty `θ̂` and a single integration
point with weight one.

This fast path is automatic — `inla(model, y)` does the right thing
without any extra arguments.

## Why this layout, not a built-in `MultinomialLikelihood`

ADR-024 records the trade-off in detail. In short:

- **Independent-Poisson** is R-INLA's default and matches its posterior
  bit-for-bit. The latent dimension grows by `n_rows` (one nuisance
  intercept per row), but the per-row `α_i` is a fixed-precision IID
  with one global precision — no extra hyperparameters.
- **Stick-breaking** would parameterise `K - 1` independent
  binary-logistic sub-models conditional on each class boundary. The
  latent dimension is smaller (no nuisance intercepts), but the
  posterior diverges from R-INLA's, and the per-class `β_k` are not
  exchangeable across the boundary chain — a different statistical
  object.

The independent-Poisson route is the right default for users coming
from R-INLA and for ranking against R-INLA oracle fixtures.

## Extending the model

- **Random effects.** Add `IID`, `BYM2`, etc. components alongside the
  class-specific `FixedEffects` block. The `α` nuisance lives on its
  own component slot and composes with anything else.
- **Per-class covariates.** `multinomial_design_matrix` accepts a
  shared `(n_rows, p)` covariate matrix; if you want different
  covariates per class, build the long-format `A_β` block by hand using
  `helper.row_id` / `helper.class_id`.
- **Choosing the reference class.** Pass `reference_class = k` to
  `multinomial_design_matrix` to drop a non-default class. The fitted
  `β` then refers to log-odds against class `k`.
- **Class names.** `multinomial_to_poisson(Y; class_names = ...)`
  attaches names to the helper for downstream use; the design-matrix
  layout is independent of the names.

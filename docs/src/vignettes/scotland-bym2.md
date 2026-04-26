# Areal — Scotland lip-cancer BYM2

The Scotland lip-cancer dataset (Clayton & Kaldor 1987; Breslow & Clayton
1993) is the standard areal-disease-mapping example: 56 districts, observed
counts `y_i`, expected counts `E_i` from age-stratification, and a single
covariate `x_i` (proportion of the workforce in agriculture, fishing, or
forestry — "AFF"). The model is a Poisson regression with a BYM2 spatial
random effect:

```math
\begin{aligned}
y_i &\sim \mathrm{Poisson}(E_i \exp(\eta_i)), \\
\eta_i &= \alpha + \beta\,x_i + b_i, \\
b &\sim \text{BYM2}(\tau, \phi; W),
\end{aligned}
```

with the [Riebler et al. (2016)](../references.md) BYM2 parameterisation
and PC priors on `τ` and `φ`.

## Loading the fixture

The Scotland BYM2 fixture lives under
[`packages/LatentGaussianModels.jl/test/oracle/fixtures/scotland_bym2.jld2`](https://github.com/HaavardHvarnes/INLA/blob/main/packages/LatentGaussianModels.jl/test/oracle/fixtures/scotland_bym2.jld2).
It carries both the input data (`y`, `E`, `x`, `W`) and the R-INLA
posterior summaries used as the validation oracle.

```@example scotland
using JLD2, SparseArrays, LinearAlgebra
using GMRFs, LatentGaussianModels

const FIXTURE = joinpath(@__DIR__, "..", "..", "..",
    "packages", "LatentGaussianModels.jl",
    "test", "oracle", "fixtures", "scotland_bym2.jld2")

fx = jldopen(FIXTURE, "r") do f
    f["fixture"]
end

inp = fx["input"]
y   = Int.(inp["cases"])
E   = Float64.(inp["expected"])
x   = Float64.(inp["x"])
W   = inp["W"]
n   = length(y)
nothing # hide
```

## Building the model

Latent vector layout: `x = [α; β; b; u]`. The intercept and AFF slope
go into `η = A x` directly; `b` is the per-region BYM2 effect; `u` is
the BYM2 unstructured component, which is internally constrained and
does not contribute to `η` directly (it shows up via `b`'s scaled
mixture).

```@example scotland
ℓ      = PoissonLikelihood(; E = E)
c_int  = Intercept()
c_beta = FixedEffects(1)
c_bym2 = BYM2(GMRFGraph(W); hyperprior_prec = PCPrecision(1.0, 0.01))

A = sparse(hcat(
    ones(n),                        # intercept → α
    reshape(x, n, 1),               # AFF slope → β
    Matrix{Float64}(I, n, n),       # per-region pick of bᵢ
    zeros(n, n),                    # u does not enter η
))

model = LatentGaussianModel(ℓ, (c_int, c_beta, c_bym2), A)
nothing # hide
```

## Fitting

```@example scotland
res = inla(model, y; int_strategy = :grid)
nothing # hide
```

## Posterior summaries

```@example scotland
fixed_effects(model, res)
```

```@example scotland
hyperparameters(model, res)
```

```@example scotland
log_marginal_likelihood(res)
```

## Comparing to R-INLA

The fixture's `summary_fixed` and `summary_hyperpar` carry the R-INLA
posterior. Per
[`plans/testing-strategy.md`](https://github.com/HaavardHvarnes/INLA/blob/main/plans/testing-strategy.md),
the v0.1 oracle tolerances are 7% relative on fixed-effect means and
10% relative on `τ`. The full assertion suite lives in
[`test/oracle/test_scotland_bym2.jl`](https://github.com/HaavardHvarnes/INLA/blob/main/packages/LatentGaussianModels.jl/test/oracle/test_scotland_bym2.jl).

```@example scotland
sf = fx["summary_fixed"]
rn = String.(sf["rownames"])
α_R = Float64(sf["mean"][findfirst(==("(Intercept)"), rn)])
β_R = Float64(sf["mean"][findfirst(==("x"), rn)])

fe = fixed_effects(model, res)
(α_julia = fe[1].mean, α_R = α_R, β_julia = fe[2].mean, β_R = β_R)
```

The marginal log-likelihood gap on Scotland (`K = 4` connected
components) currently sits around 5 nats / 3.4% — likely the global
Sørbye–Rue scaling vs R-INLA's per-component scaling
([Freni-Sterrantino et al. 2018](../references.md)). Asserted via
`@test_broken` so the suite surfaces a future fix automatically;
Pennsylvania (single component) passes within the 1% tolerance.

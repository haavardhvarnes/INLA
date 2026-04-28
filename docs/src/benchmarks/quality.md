# Quality vs R-INLA

Side-by-side posterior summaries for the three R-INLA examples that
ship as oracle fixtures: Scotland lip cancer (BYM2 areal),
Pennsylvania lung cancer (BYM2 areal), and Meuse zinc (SPDE
geostatistics). Each section refits the model in Julia and compares
against the R-INLA posterior stored in the fixture.

The tolerance bands come from
[`plans/testing-strategy.md`](https://github.com/HaavardHvarnes/INLA.jl/blob/main/plans/testing-strategy.md):
roughly 1–7% relative on fixed-effect means, 5–10% on hyperparameters,
2% on the marginal log-likelihood for connected-graph BYM2 and the
SPDE oracle. Looser bands and known divergences are noted per dataset.
The full assertion suites live in each package's `test/oracle/`.

Performance / wall-clock comparison is deferred to a v0.1.x patch (see
[`plans/quality-and-perf-benchmarks.md`](https://github.com/HaavardHvarnes/INLA.jl/blob/main/plans/quality-and-perf-benchmarks.md)).

## Roll-up

| Dataset | Fixed effects | Hyperparameters | Marginal log-lik |
|---|---|---|---|
| Scotland BYM2 | within 7% | within 10% | gap ≈ 3.4% (multi-component scaling, `@test_broken`) |
| Pennsylvania BYM2 | within 5% | within 10% | within 2% |
| Meuse SPDE | within 1% | within 5% | within 2 nats |

## Helpers (hidden)

```@setup bench
using JLD2, SparseArrays, LinearAlgebra, Markdown
using GMRFs, LatentGaussianModels, INLASPDE

function _comp_md(rows)
    head = "| Quantity | Julia | R-INLA | Rel. error |\n"
    sep  = "|---|---|---|---|\n"
    body = join(["| $(r[1]) | $(round(r[2]; sigdigits = 5)) | $(round(r[3]; sigdigits = 5)) | $(round(abs(r[2] - r[3]) / max(abs(r[3]), 1.0) * 100; sigdigits = 3))% |"
                 for r in rows], "\n")
    return Markdown.parse(head * sep * body)
end

function _row_lookup(rows, name, col)
    rn = String.(rows["rownames"])
    idx = findfirst(==(name), rn)
    return Float64(rows[col][idx])
end

const REPO_ROOT = joinpath(@__DIR__, "..", "..", "..")
```

## Scotland BYM2

Areal Poisson disease mapping; 56 districts; intercept + AFF
covariate + BYM2 spatial random effect; PC priors throughout. See the
[Scotland vignette](../vignettes/scotland-bym2.md).

```@example bench
fx = jldopen(joinpath(REPO_ROOT, "packages", "LatentGaussianModels.jl",
    "test", "oracle", "fixtures", "scotland_bym2.jld2"), "r") do f
    f["fixture"]
end
inp = fx["input"]
y, E, x, W = Int.(inp["cases"]), Float64.(inp["expected"]),
             Float64.(inp["x"]), inp["W"]
n = length(y)

ℓ      = PoissonLikelihood(; E = E)
c_int  = Intercept()
c_beta = FixedEffects(1)
c_bym2 = BYM2(GMRFGraph(W); hyperprior_prec = PCPrecision(1.0, 0.01))
A = sparse(hcat(ones(n), reshape(x, n, 1),
                Matrix{Float64}(I, n, n), zeros(n, n)))
model = LatentGaussianModel(ℓ, (c_int, c_beta, c_bym2), A)
res   = inla(model, y; int_strategy = :grid)

fe   = fixed_effects(model, res)
α_R  = _row_lookup(fx["summary_fixed"], "(Intercept)", "mean")
β_R  = _row_lookup(fx["summary_fixed"], "x", "mean")
τ_R  = _row_lookup(fx["summary_hyperpar"], "Precision for region", "mean")
φ_R  = _row_lookup(fx["summary_hyperpar"], "Phi for region", "mean")
mlik_R = Float64(fx["mlik"][1])

_comp_md([
    ("α (intercept) mean",  fe[1].mean,                  α_R),
    ("β (AFF) mean",        fe[2].mean,                  β_R),
    ("τ_b mean",            exp(res.θ̂[1]),               τ_R),
    ("φ mean",              1 / (1 + exp(-res.θ̂[2])),    φ_R),
    ("log marginal lik",    log_marginal_likelihood(res), mlik_R),
])
```

**Status.** Fixed effects pass within 7% relative; `τ_b` within 10%;
`φ` is weakly identified (loose comparison). The marginal log-lik gap
of ~3.4% is asserted via `@test_broken`; root cause is the global
Sørbye–Rue scaling vs R-INLA's per-component scaling on this `K = 4`
disconnected graph (Freni-Sterrantino et al. 2018).

## Pennsylvania BYM2

Same model class as Scotland but with a single connected graph (`K =
1`), 67 counties, smoking-rate covariate. See
[`test/oracle/test_pennsylvania_bym2.jl`](https://github.com/HaavardHvarnes/INLA.jl/blob/main/packages/LatentGaussianModels.jl/test/oracle/test_pennsylvania_bym2.jl).

```@example bench
fx = jldopen(joinpath(REPO_ROOT, "packages", "LatentGaussianModels.jl",
    "test", "oracle", "fixtures", "pennsylvania_bym2.jld2"), "r") do f
    f["fixture"]
end
inp = fx["input"]
y, E, x, W = Int.(inp["cases"]), Float64.(inp["expected"]),
             Float64.(inp["x"]), inp["W"]
n = length(y)

ℓ      = PoissonLikelihood(; E = E)
c_int  = Intercept()
c_beta = FixedEffects(1)
c_bym2 = BYM2(GMRFGraph(W); hyperprior_prec = PCPrecision(1.0, 0.01))
A = sparse(hcat(ones(n), reshape(x, n, 1),
                Matrix{Float64}(I, n, n), zeros(n, n)))
model = LatentGaussianModel(ℓ, (c_int, c_beta, c_bym2), A)
res   = inla(model, y; int_strategy = :grid)

fe   = fixed_effects(model, res)
α_R  = _row_lookup(fx["summary_fixed"], "(Intercept)", "mean")
β_R  = _row_lookup(fx["summary_fixed"], "x", "mean")
τ_R  = _row_lookup(fx["summary_hyperpar"], "Precision for region", "mean")
mlik_R = Float64(fx["mlik"][1])

_comp_md([
    ("α (intercept) mean",  fe[1].mean,                   α_R),
    ("β (smoking) mean",    fe[2].mean,                   β_R),
    ("τ_b mean",            exp(res.θ̂[1]),                τ_R),
    ("log marginal lik",    log_marginal_likelihood(res), mlik_R),
])
```

**Status.** All headline summaries pass within 2% relative on the
marginal log-likelihood, 5% on fixed effects, 10% on `τ_b`. This is
the cleanest BYM2 oracle: the connected-graph case where the
Sørbye–Rue scaling is unambiguous.

## Meuse SPDE

Gaussian observation model on log-zinc; intercept + distance-to-river
covariate + SPDE-Matérn (α = 2) random field; PC priors on `(ρ, σ)`
and on noise precision. See the
[Meuse vignette](../vignettes/meuse-spde.md).

```@example bench
fxt = load(joinpath(REPO_ROOT, "packages", "INLASPDE.jl",
    "test", "oracle", "fixtures", "meuse_spde.jld2"))["fixture"]

y       = Float64.(fxt["input"]["y"])
dist    = Float64.(fxt["input"]["dist"])
points  = fxt["mesh"]["loc"]::Matrix{Float64}
tv      = fxt["mesh"]["tv"]::Matrix{Int}
A_field = SparseMatrixCSC{Float64, Int}(fxt["A_field"])
n_obs = length(y)

spde = SPDE2(points, tv; α = 2,
    pc = PCMatern(; range_U = 0.5, range_α = 0.5,
                    sigma_U = 1.0, sigma_α = 0.5))

c_int  = Intercept(prec = 1.0e-3)
c_dist = FixedEffects(1; prec = 1.0e-3)
A = hcat(ones(n_obs, 1), reshape(dist, n_obs, 1), A_field)

ℓ     = GaussianLikelihood(hyperprior = PCPrecision(1.0, 0.01))
model = LatentGaussianModel(ℓ, (c_int, c_dist, spde), A)
res   = inla(model, y)

fe = fixed_effects(model, res)
α_R = _row_lookup(fxt["summary_fixed"], "intercept", "mean")
β_R = _row_lookup(fxt["summary_fixed"], "dist", "mean")
τ_R = _row_lookup(fxt["summary_hyperpar"],
                  "Precision for the Gaussian observations", "mean")
ρ_R = _row_lookup(fxt["summary_hyperpar"], "Range for field", "mean")
σ_R = _row_lookup(fxt["summary_hyperpar"], "Stdev for field", "mean")
mlik_R = Float64(fxt["mlik"][1])

ρ̂, σ̂ = spde_user_scale(spde, res.θ̂[2:3])

_comp_md([
    ("α (intercept) mean",  fe[1].mean,                   α_R),
    ("β (dist) mean",       fe[2].mean,                   β_R),
    ("τ_ε (noise prec.)",   exp(res.θ̂[1]),                τ_R),
    ("ρ (range)",           ρ̂,                            ρ_R),
    ("σ (stdev)",           σ̂,                            σ_R),
    ("log marginal lik",    log_marginal_likelihood(res), mlik_R),
])
```

**Status.** Fixed effects within 1% relative; `(ρ, σ)` within ~3.5%;
noise precision within ~5%; marginal log-lik within ~1.3 nats. The
SPDE oracle is the tightest of the three — Matérn covariance
reproduction is well-conditioned on the Meuse mesh, and the joint
PC-Matérn / PC-precision priors agree closely with R-INLA's defaults.

## Source

This page is generated from
[`docs/src/benchmarks/quality.md`](https://github.com/HaavardHvarnes/INLA.jl/blob/main/docs/src/benchmarks/quality.md).
The same fits run as oracle tests in each package's `test/oracle/`
suite — the docs page and the test suite use identical model code
and fixtures, so any divergence between rendered numbers and CI
status would be a bug.

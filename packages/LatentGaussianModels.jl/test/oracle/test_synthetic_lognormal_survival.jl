# Oracle test: Lognormal-AFT survival regression on synthetic right-censored
# data vs R-INLA's `family = "lognormalsurv"`. Smallest oracle for the
# lognormal-survival pathway with a single hyperparameter (precision τ on
# log T) on top of the censoring-aware likelihood. Intercept + one
# covariate, n = 200, ~14% right-censored.
#
# This fixture exercises `dim(θ) = 1`, so we use full `inla(...)`
# integration over θ rather than the `dim(θ) = 0` direct-Laplace
# fallback used in `test_synthetic_exponential_survival.jl`.
#
# Hyperprior parity: PC prior on σ with `P(σ > 1) = 0.01`. Matches our
# `LognormalSurvLikelihood` default `PCPrecision(1.0, 0.01)` exactly,
# so this oracle does not need a hyperprior override.
#
# Fixture: scripts/generate-fixtures/lgm/synthetic_lognormal_survival.R.
# Skipped transparently if the JLD2 fixture has not been generated.

include("load_fixture.jl")

using Test
using SparseArrays
using LinearAlgebra
using LatentGaussianModels: LognormalSurvLikelihood, NONE, RIGHT, Censoring,
                            Intercept, FixedEffects, LatentGaussianModel, inla,
                            fixed_effects, hyperparameters

const LNS_SURV_FIXTURE = "synthetic_lognormal_survival"

const LNS_SURV_FE_MEAN_TOL = 0.05    # |Δmean| / max(|R|, 1)
const LNS_SURV_FE_SD_TOL = 0.10    # |Δsd|  / R-sd
const LNS_SURV_PREC_TOL = 0.10    # |Δτ|   / R-mean
const LNS_SURV_PREC_SD_TOL = 0.20    # |Δsd_τ| / R-sd

_rel_lns(a, b) = abs(a - b) / max(abs(b), 1.0)

function _lns_row(frame, name::AbstractString, col::AbstractString)
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $(join(rn, "; ")))")
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

@testset "synthetic_lognormal_survival vs R-INLA" begin
    if !has_oracle_fixture(LNS_SURV_FIXTURE)
        @test_skip "oracle fixture $LNS_SURV_FIXTURE not generated " *
                   "(see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(LNS_SURV_FIXTURE)
        @test fx["name"] == LNS_SURV_FIXTURE
        @test haskey(fx, "summary_fixed")
        @test haskey(fx, "summary_hyperpar")

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate"
        else
            inp = fx["input"]
            time = Float64.(inp["time"])
            event = Int.(inp["event"])
            xcov = Float64.(inp["x"])
            n = length(time)

            cens = Censoring[e == 1 ? NONE : RIGHT for e in event]
            ℓ = LognormalSurvLikelihood(censoring=cens)
            A = sparse(hcat(ones(n), reshape(xcov, n, 1)))
            model = LatentGaussianModel(ℓ, (Intercept(), FixedEffects(1)), A)

            res = inla(model, time)

            # --- Fixed effects: posterior mean + sd --------------------------
            sf = fx["summary_fixed"]
            α_R = _lns_row(sf, "(Intercept)", "mean")
            β_R = _lns_row(sf, "x", "mean")
            α_sd_R = _lns_row(sf, "(Intercept)", "sd")
            β_sd_R = _lns_row(sf, "x", "sd")

            fe = fixed_effects(model, res)
            @test length(fe) == 2
            @test _rel_lns(fe[1].mean, α_R) < LNS_SURV_FE_MEAN_TOL
            @test _rel_lns(fe[2].mean, β_R) < LNS_SURV_FE_MEAN_TOL
            @test abs(fe[1].sd - α_sd_R) / α_sd_R < LNS_SURV_FE_SD_TOL
            @test abs(fe[2].sd - β_sd_R) / β_sd_R < LNS_SURV_FE_SD_TOL

            # --- Precision τ: posterior mean + sd on user scale -------------
            sh = fx["summary_hyperpar"]
            τ_R = _lns_row(sh, "Precision for the lognormalsurv observations",
                "mean")
            τ_sd_R = _lns_row(sh, "Precision for the lognormalsurv observations",
                "sd")

            # Mean on user scale via delta method around θ̂ = log τ̂.
            # E[τ] ≈ exp(θ_mean), Var[τ] ≈ Σθ * (exp θ̂)².
            τ_J = exp(res.θ_mean[1])
            τ_sd_J = sqrt(res.Σθ[1, 1]) * exp(res.θ̂[1])
            @test _rel_lns(τ_J, τ_R) < LNS_SURV_PREC_TOL
            @test _rel_lns(τ_sd_J, τ_sd_R) < LNS_SURV_PREC_SD_TOL

            # --- mlik ---------------------------------------------------------
            # Exposed but not asserted with a tight tolerance: shares the
            # polynomial-form Laplace gap traced under `weibullsurv`
            # (Phase F.5 excavation 2026-05-02 — see
            # `test_synthetic_weibull_survival.jl` header). Closure
            # requires modifying `src/inference/laplace.jl` to match
            # R-INLA's polynomial form; deferred to v0.3.
            @test isfinite(res.log_marginal)
        end
    end
end

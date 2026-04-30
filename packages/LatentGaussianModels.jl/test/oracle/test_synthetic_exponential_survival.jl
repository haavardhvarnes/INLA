# Oracle test: Exponential survival regression on synthetic right-censored
# data vs R-INLA's `family = "exponentialsurv"`. Smallest oracle for the
# censoring-aware ExponentialLikelihood pathway — intercept + one covariate,
# n = 200, ~28% right-censored, no latent random effect.
#
# Because both R-INLA `exponentialsurv` and our `ExponentialLikelihood`
# carry **zero hyperparameters** (the rate is fully determined by η),
# R-INLA's posterior is itself a Laplace approximation. We therefore
# compare against `laplace(...)` directly rather than `inla(...)` — the
# latter currently has a `dim(θ) = 0` integration-grid limitation that
# is out of scope for ADR-018 PR1 (tracked separately for v0.2).
#
# Fixture: scripts/generate-fixtures/lgm/synthetic_exponential_survival.R.
# Skipped transparently if the JLD2 fixture has not been generated.

include("load_fixture.jl")

using Test
using SparseArrays
using LinearAlgebra
using LatentGaussianModels: ExponentialLikelihood, NONE, RIGHT, Censoring,
    Intercept, FixedEffects, LatentGaussianModel, laplace

const EXP_SURV_FIXTURE = "synthetic_exponential_survival"

const EXP_SURV_FIXED_EFFECT_TOL = 0.05   # |Δmean| / max(|R|, 1)
const EXP_SURV_SD_REL_TOL       = 0.10   # |Δsd|  / R-sd
const EXP_SURV_MLIK_REL_TOL     = 0.01   # |Δmlik| / |R-mlik|

_rel_exp(a, b) = abs(a - b) / max(abs(b), 1.0)

function _exp_row(frame, name::AbstractString, col::AbstractString)
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $(join(rn, "; ")))")
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

@testset "synthetic_exponential_survival vs R-INLA" begin
    if !has_oracle_fixture(EXP_SURV_FIXTURE)
        @test_skip "oracle fixture $EXP_SURV_FIXTURE not generated " *
                   "(see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(EXP_SURV_FIXTURE)
        @test fx["name"] == EXP_SURV_FIXTURE
        @test haskey(fx, "summary_fixed")
        # exponentialsurv has no hyperparameters
        @test isempty(fx["summary_hyperpar"])

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate"
        else
            inp = fx["input"]
            time  = Float64.(inp["time"])
            event = Int.(inp["event"])
            xcov  = Float64.(inp["x"])
            n = length(time)

            # event = 1 → uncensored (NONE); event = 0 → right-censored
            censoring = Censoring[e == 1 ? NONE : RIGHT for e in event]
            ℓ = ExponentialLikelihood(censoring = censoring)
            c_int  = Intercept()
            c_beta = FixedEffects(1)
            A = sparse(hcat(ones(n), reshape(xcov, n, 1)))
            model = LatentGaussianModel(ℓ, (c_int, c_beta), A)

            res = laplace(model, time, Float64[])
            @test res.converged
            @test res.iterations < 50

            # Posterior sd from the precision diagonal (exact for Gaussian
            # Laplace approximation around the mode).
            Σ_diag = diag(inv(Matrix(res.precision)))
            sd_J = sqrt.(Σ_diag)

            # --- Fixed effects: posterior mean -------------------------------
            sf = fx["summary_fixed"]
            α_R = _exp_row(sf, "(Intercept)", "mean")
            β_R = _exp_row(sf, "x", "mean")

            @test _rel_exp(res.mode[1], α_R) < EXP_SURV_FIXED_EFFECT_TOL
            @test _rel_exp(res.mode[2], β_R) < EXP_SURV_FIXED_EFFECT_TOL

            # --- Fixed effects: posterior sd ---------------------------------
            α_sd_R = _exp_row(sf, "(Intercept)", "sd")
            β_sd_R = _exp_row(sf, "x", "sd")

            @test abs(sd_J[1] - α_sd_R) / α_sd_R < EXP_SURV_SD_REL_TOL
            @test abs(sd_J[2] - β_sd_R) / β_sd_R < EXP_SURV_SD_REL_TOL

            # --- Marginal log-likelihood -------------------------------------
            mlik_R = Float64(fx["mlik"][1])
            @test _rel_exp(res.log_marginal, mlik_R) <
                  EXP_SURV_MLIK_REL_TOL
        end
    end
end

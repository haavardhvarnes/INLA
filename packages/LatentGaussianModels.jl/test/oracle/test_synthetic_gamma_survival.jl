# Oracle test: Gamma-survival regression on synthetic right-censored
# data vs R-INLA's `family = "gammasurv"`. Smallest oracle for the
# gamma-survival pathway with a single hyperparameter (shape φ on
# Gamma(shape = φ, rate = φ/μ)) on top of the censoring-aware
# likelihood. Intercept + one covariate, n = 200, ~14% right-censored.
#
# Hyperprior parity: loggamma(1, 5e-5) on log φ. Matches both R-INLA's
# `gammasurv` default and our `GammaSurvLikelihood` default
# (`GammaPrecision(1.0, 5.0e-5)`), so this oracle does not need a
# hyperprior override.
#
# Fixture: scripts/generate-fixtures/lgm/synthetic_gamma_survival.R.
# Skipped transparently if the JLD2 fixture has not been generated.

include("load_fixture.jl")

using Test
using SparseArrays
using LinearAlgebra
using LatentGaussianModels: GammaSurvLikelihood, NONE, RIGHT, Censoring,
                            Intercept, FixedEffects, LatentGaussianModel, inla,
                            fixed_effects, hyperparameters

const GS_SURV_FIXTURE = "synthetic_gamma_survival"

const GS_SURV_FE_MEAN_TOL = 0.05    # |Δmean| / max(|R|, 1)
const GS_SURV_FE_SD_TOL = 0.10    # |Δsd|  / R-sd
const GS_SURV_PHI_TOL = 0.10    # |Δφ|   / R-mean
const GS_SURV_PHI_SD_TOL = 0.20    # |Δsd_φ| / R-sd

_rel_gs(a, b) = abs(a - b) / max(abs(b), 1.0)

function _gs_row(frame, name::AbstractString, col::AbstractString)
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $(join(rn, "; ")))")
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

@testset "synthetic_gamma_survival vs R-INLA" begin
    if !has_oracle_fixture(GS_SURV_FIXTURE)
        @test_skip "oracle fixture $GS_SURV_FIXTURE not generated " *
                   "(see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(GS_SURV_FIXTURE)
        @test fx["name"] == GS_SURV_FIXTURE
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
            ℓ = GammaSurvLikelihood(censoring=cens)
            A = sparse(hcat(ones(n), reshape(xcov, n, 1)))
            model = LatentGaussianModel(ℓ, (Intercept(), FixedEffects(1)), A)

            res = inla(model, time)

            # --- Fixed effects: posterior mean + sd --------------------------
            sf = fx["summary_fixed"]
            α_R = _gs_row(sf, "(Intercept)", "mean")
            β_R = _gs_row(sf, "x", "mean")
            α_sd_R = _gs_row(sf, "(Intercept)", "sd")
            β_sd_R = _gs_row(sf, "x", "sd")

            fe = fixed_effects(model, res)
            @test length(fe) == 2
            @test _rel_gs(fe[1].mean, α_R) < GS_SURV_FE_MEAN_TOL
            @test _rel_gs(fe[2].mean, β_R) < GS_SURV_FE_MEAN_TOL
            @test abs(fe[1].sd - α_sd_R) / α_sd_R < GS_SURV_FE_SD_TOL
            @test abs(fe[2].sd - β_sd_R) / β_sd_R < GS_SURV_FE_SD_TOL

            # --- Shape φ: posterior mean + sd on user scale -----------------
            sh = fx["summary_hyperpar"]
            # R-INLA rowname is `Precision-parameter for the Gamma surv`
            # (truncated form stored in summary.hyperpar).
            φ_R = _gs_row(sh, "Precision-parameter for the Gamma surv",
                "mean")
            φ_sd_R = _gs_row(sh, "Precision-parameter for the Gamma surv",
                "sd")

            # Mean on user scale via delta method around θ̂ = log φ̂.
            # E[φ] ≈ exp(θ_mean), Var[φ] ≈ Σθ * (exp θ̂)².
            φ_J = exp(res.θ_mean[1])
            φ_sd_J = sqrt(res.Σθ[1, 1]) * exp(res.θ̂[1])
            @test _rel_gs(φ_J, φ_R) < GS_SURV_PHI_TOL
            @test _rel_gs(φ_sd_J, φ_sd_R) < GS_SURV_PHI_SD_TOL

            # --- mlik ---------------------------------------------------------
            # Exposed but not asserted with a tight tolerance: R-INLA's
            # internal Gaussian normalising constant for `gammasurv` may
            # diverge from Julia's by a constant (cf. the lognormalsurv +
            # weibullsurv gaps tracked as v0.2 calibration items). We assert
            # the value is finite; tightening is a v0.2 task.
            @test isfinite(res.log_marginal)
        end
    end
end

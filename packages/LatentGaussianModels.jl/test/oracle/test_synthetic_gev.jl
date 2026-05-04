# Oracle test: GEV regression on synthetic data vs R-INLA.
#
# Smallest possible oracle for the `family = "gev"` pathway —
# intercept + one covariate, no latent random effect, n = 200. R-INLA
# marks "gev" disabled in current releases (use "bgev"); the fixture
# script re-enables it via `enable.model.likelihood.gev <- TRUE`.
#
# Tests fixed-effect agreement, that the precision and shape joint
# modes match R-INLA's marginal modes within tolerance, and that the
# marginal log-likelihood matches within tolerance. The joint-vs-
# marginal mode gap is wider here than for SN/T because the GEV body
# isn't symmetric and the posterior on θ is more skew at n = 200.
#
# Fixture: scripts/generate-fixtures/lgm/synthetic_gev.R.
# Skipped transparently if the JLD2 fixture has not been generated.
#
# Layout: identity projector (n × 2 design column-stack) + Intercept
# + FixedEffects(1) + GEVLikelihood. Internal-scale hyperparameters
# `θ̂[1] = log τ`, `θ̂[2] = ξ / xi_scale`; user scale via `exp(·)` and
# `xi_scale · θ̂[2]` (default `xi_scale = 0.1`).

include("load_fixture.jl")

using Test
using SparseArrays
using LatentGaussianModels: GEVLikelihood, Intercept, FixedEffects,
                            LatentGaussianModel, inla, fixed_effects, hyperparameters,
                            log_marginal_likelihood

const GEV_FIXTURE = "synthetic_gev"

const GEV_FIXED_EFFECT_TOL = 0.10
const GEV_TAU_REL_TOL = 0.15
const GEV_XI_REL_TOL = 0.30
const GEV_MLIK_REL_TOL = 0.05

_rel_gev(a, b) = abs(a - b) / max(abs(b), 1.0)

function _gev_row_value(frame, name::AbstractString, col::AbstractString)
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $(join(rn, "; ")))")
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

@testset "synthetic_gev vs R-INLA" begin
    if !has_oracle_fixture(GEV_FIXTURE)
        @test_skip "oracle fixture $GEV_FIXTURE not generated (see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(GEV_FIXTURE)
        @test fx["name"] == GEV_FIXTURE
        @test haskey(fx, "summary_fixed")
        @test haskey(fx, "summary_hyperpar")

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate with the current R script"
        else
            inp = fx["input"]
            y = Float64.(inp["y"])
            x = Float64.(inp["x"])
            n = length(y)

            ℓ = GEVLikelihood()       # default xi_scale = 0.1, matching R-INLA
            c_int = Intercept()
            c_beta = FixedEffects(1)
            A = sparse(hcat(ones(n), reshape(x, n, 1)))
            model = LatentGaussianModel(ℓ, (c_int, c_beta), A)

            # Default initial η is zero, which lies inside the support
            # 1 + ξ √(τ s)(y − η) > 0 for the moderate ξ regime here.
            res = inla(model, y; int_strategy = :grid)

            # --- Fixed effects ------------------------------------------------
            sf = fx["summary_fixed"]
            α_R = _gev_row_value(sf, "(Intercept)", "mean")
            β_R = _gev_row_value(sf, "x", "mean")

            fe = fixed_effects(model, res)
            @test length(fe) == 2
            @test _rel_gev(fe[1].mean, α_R) < GEV_FIXED_EFFECT_TOL
            @test _rel_gev(fe[2].mean, β_R) < GEV_FIXED_EFFECT_TOL

            # --- Hyperparameters: τ and ξ on user scale ----------------------
            # Compare Julia's joint mode (user scale) to R-INLA's marginal
            # mode. The shape ξ tolerance is wider than τ's because the
            # GEV posterior on the tail parameter is asymmetric at this n.
            sh = fx["summary_hyperpar"]
            τ_R_mode = _gev_row_value(sh, "precision for GEV observations", "mode")
            ξ_R_mode = _gev_row_value(sh, "tail parameter for GEV observations", "mode")

            τ_J = exp(res.θ̂[1])
            ξ_J = ℓ.xi_scale * res.θ̂[2]

            @test _rel_gev(τ_J, τ_R_mode) < GEV_TAU_REL_TOL
            @test _rel_gev(ξ_J, ξ_R_mode) < GEV_XI_REL_TOL

            hp = hyperparameters(model, res)
            @test length(hp) == 2
            @test all(h -> isfinite(h.mean) && h.sd > 0, hp)

            # --- Marginal log-likelihood --------------------------------------
            mlik_R = Float64(fx["mlik"][1])
            mlik_J = log_marginal_likelihood(res)
            @test _rel_gev(mlik_J, mlik_R) < GEV_MLIK_REL_TOL
        end
    end
end

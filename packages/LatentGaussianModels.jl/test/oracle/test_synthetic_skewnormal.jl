# Oracle test: skew-normal regression on synthetic data vs R-INLA.
#
# Smallest possible oracle for the `family = "sn"` pathway —
# intercept + one covariate, no latent random effect, n = 300. Tests
# fixed-effect agreement, that the precision and skewness joint modes
# match R-INLA's marginal modes within 5%, and that the marginal
# log-likelihood matches within 5%.
#
# Fixture: scripts/generate-fixtures/lgm/synthetic_skewnormal.R.
# Skipped transparently if the JLD2 fixture has not been generated.
#
# Layout: identity projector (n × 2 design column-stack) + Intercept
# + FixedEffects(1) + SkewNormalLikelihood. Internal-scale
# hyperparameters `θ̂[1] = log τ`, `θ̂[2] = logit-skew`; user scale
# via `exp(·)` and `0.988 · tanh(·/2)`.

include("load_fixture.jl")

using Test
using SparseArrays
using LatentGaussianModels: SkewNormalLikelihood, Intercept, FixedEffects,
                            LatentGaussianModel, inla, fixed_effects, hyperparameters,
                            log_marginal_likelihood

const SN_FIXTURE = "synthetic_skewnormal"

const SN_FIXED_EFFECT_TOL = 0.05
const SN_TAU_REL_TOL = 0.05
const SN_GAMMA_REL_TOL = 0.05
const SN_MLIK_REL_TOL = 0.05

_rel_sn(a, b) = abs(a - b) / max(abs(b), 1.0)

function _sn_row_value(frame, name::AbstractString, col::AbstractString)
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $(join(rn, "; ")))")
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

@testset "synthetic_skewnormal vs R-INLA" begin
    if !has_oracle_fixture(SN_FIXTURE)
        @test_skip "oracle fixture $SN_FIXTURE not generated (see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(SN_FIXTURE)
        @test fx["name"] == SN_FIXTURE
        @test haskey(fx, "summary_fixed")
        @test haskey(fx, "summary_hyperpar")

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate with the current R script"
        else
            inp = fx["input"]
            y = Float64.(inp["y"])
            x = Float64.(inp["x"])
            n = length(y)

            ℓ = SkewNormalLikelihood()
            c_int = Intercept()
            c_beta = FixedEffects(1)
            A = sparse(hcat(ones(n), reshape(x, n, 1)))
            model = LatentGaussianModel(ℓ, (c_int, c_beta), A)

            res = inla(model, y; int_strategy = :grid)

            # --- Fixed effects ------------------------------------------------
            sf = fx["summary_fixed"]
            α_R = _sn_row_value(sf, "(Intercept)", "mean")
            β_R = _sn_row_value(sf, "x", "mean")

            fe = fixed_effects(model, res)
            @test length(fe) == 2
            @test _rel_sn(fe[1].mean, α_R) < SN_FIXED_EFFECT_TOL
            @test _rel_sn(fe[2].mean, β_R) < SN_FIXED_EFFECT_TOL

            # --- Hyperparameters: τ and γ on user scale ----------------------
            # Compare Julia's joint mode (user scale) to R-INLA's marginal
            # mode. At n = 300 with a moderate skew (γ = 0.5) the posterior
            # is close to Gaussian on the internal scale and the joint and
            # marginal modes agree within 5%.
            sh = fx["summary_hyperpar"]
            τ_R_mode = _sn_row_value(sh, "precision for skew-normal observations", "mode")
            γ_R_mode = _sn_row_value(sh, "Skewness for skew-normal observations", "mode")

            τ_J = exp(res.θ̂[1])
            γ_J = 0.988 * tanh(res.θ̂[2] / 2)

            @test _rel_sn(τ_J, τ_R_mode) < SN_TAU_REL_TOL
            @test _rel_sn(γ_J, γ_R_mode) < SN_GAMMA_REL_TOL

            hp = hyperparameters(model, res)
            @test length(hp) == 2
            @test all(h -> isfinite(h.mean) && h.sd > 0, hp)

            # --- Marginal log-likelihood --------------------------------------
            mlik_R = Float64(fx["mlik"][1])
            mlik_J = log_marginal_likelihood(res)
            @test _rel_sn(mlik_J, mlik_R) < SN_MLIK_REL_TOL
        end
    end
end

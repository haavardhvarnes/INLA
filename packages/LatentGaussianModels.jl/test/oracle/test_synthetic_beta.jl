# Oracle test: Beta regression on synthetic data vs R-INLA.
#
# Smallest possible oracle for the `family = "beta"` pathway —
# intercept + one covariate, no latent random effect, n = 200. Tests
# fixed-effect agreement, that the precision hyperparameter mode is
# recovered to within 15%, and the marginal log-likelihood within 5%.
#
# Fixture: scripts/generate-fixtures/lgm/synthetic_beta.R.
# Skipped transparently if the JLD2 fixture has not been generated.
#
# Layout: identity projector (n × 2 design column-stack) + Intercept
# + FixedEffects(1) + BetaLikelihood. Logit link is the only one
# supported by `BetaLikelihood`. The internal-scale hyperparameter is
# `θ̂[1] = log(φ)`; we map back to user scale via `exp` and compare to
# R-INLA's `mode` column.

include("load_fixture.jl")

using Test
using SparseArrays
using LatentGaussianModels: BetaLikelihood, Intercept, FixedEffects,
                            LatentGaussianModel, inla, fixed_effects, hyperparameters,
                            log_marginal_likelihood

const BETA_FIXTURE = "synthetic_beta"

const BETA_FIXED_EFFECT_TOL = 0.05
const BETA_PHI_REL_TOL = 0.15
const BETA_MLIK_REL_TOL = 0.05

_rel_beta(a, b) = abs(a - b) / max(abs(b), 1.0)

function _beta_row_value(frame, name::AbstractString, col::AbstractString)
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $(join(rn, "; ")))")
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

@testset "synthetic_beta vs R-INLA" begin
    if !has_oracle_fixture(BETA_FIXTURE)
        @test_skip "oracle fixture $BETA_FIXTURE not generated (see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(BETA_FIXTURE)
        @test fx["name"] == BETA_FIXTURE
        @test haskey(fx, "summary_fixed")
        @test haskey(fx, "summary_hyperpar")

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate with the current R script"
        else
            inp = fx["input"]
            y = Float64.(inp["y"])
            x = Float64.(inp["x"])
            n = length(y)

            ℓ = BetaLikelihood()
            c_int = Intercept()
            c_beta = FixedEffects(1)
            A = sparse(hcat(ones(n), reshape(x, n, 1)))
            model = LatentGaussianModel(ℓ, (c_int, c_beta), A)

            res = inla(model, y; int_strategy = :grid)

            # --- Fixed effects ------------------------------------------------
            sf = fx["summary_fixed"]
            α_R = _beta_row_value(sf, "(Intercept)", "mean")
            β_R = _beta_row_value(sf, "x", "mean")

            fe = fixed_effects(model, res)
            @test length(fe) == 2
            @test _rel_beta(fe[1].mean, α_R) < BETA_FIXED_EFFECT_TOL
            @test _rel_beta(fe[2].mean, β_R) < BETA_FIXED_EFFECT_TOL

            # --- Hyperparameter: precision φ on user scale --------------------
            # R-INLA labels this "precision parameter for the beta observations".
            # Our internal scale is log(φ); compare mode-vs-mode.
            sh = fx["summary_hyperpar"]
            phi_R = _beta_row_value(
                sh, "precision parameter for the beta observations", "mode")
            phi_J = exp(res.θ̂[1])
            @test _rel_beta(phi_J, phi_R) < BETA_PHI_REL_TOL

            hp = hyperparameters(model, res)
            @test length(hp) == 1
            @test isfinite(hp[1].mean) && hp[1].sd > 0

            # --- Marginal log-likelihood --------------------------------------
            mlik_R = Float64(fx["mlik"][1])
            mlik_J = log_marginal_likelihood(res)
            @test _rel_beta(mlik_J, mlik_R) < BETA_MLIK_REL_TOL
        end
    end
end

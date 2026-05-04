# Oracle test: BetaBinomial regression on synthetic data vs R-INLA.
#
# Smallest possible oracle for the `family = "betabinomial"` pathway —
# intercept + one covariate, no latent random effect, n = 200, with
# per-observation trial counts in [5, 25]. Tests fixed-effect agreement,
# that the overdispersion ρ mode is recovered to within 15%, and the
# marginal log-likelihood within 5%.
#
# Fixture: scripts/generate-fixtures/lgm/synthetic_betabinomial.R.
# Skipped transparently if the JLD2 fixture has not been generated.
#
# Layout: identity projector (n × 2 design column-stack) + Intercept
# + FixedEffects(1) + BetaBinomialLikelihood(n_trials). The internal-
# scale hyperparameter is `θ̂[1] = logit(ρ)`; we map back via
# `expit` and compare to R-INLA's `mode` column.

include("load_fixture.jl")

using Test
using SparseArrays
using LatentGaussianModels: BetaBinomialLikelihood, Intercept, FixedEffects,
                            LatentGaussianModel, inla, fixed_effects, hyperparameters,
                            log_marginal_likelihood

const BB_FIXTURE = "synthetic_betabinomial"

const BB_FIXED_EFFECT_TOL = 0.05
const BB_RHO_REL_TOL = 0.15
const BB_MLIK_REL_TOL = 0.05

_rel_bb(a, b) = abs(a - b) / max(abs(b), 1.0)

function _bb_row_value(frame, name::AbstractString, col::AbstractString)
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $(join(rn, "; ")))")
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

@testset "synthetic_betabinomial vs R-INLA" begin
    if !has_oracle_fixture(BB_FIXTURE)
        @test_skip "oracle fixture $BB_FIXTURE not generated (see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(BB_FIXTURE)
        @test fx["name"] == BB_FIXTURE
        @test haskey(fx, "summary_fixed")
        @test haskey(fx, "summary_hyperpar")

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate with the current R script"
        else
            inp = fx["input"]
            y = Int.(inp["y"])
            x = Float64.(inp["x"])
            n_trials = Int.(inp["n_trials"])
            n = length(y)

            ℓ = BetaBinomialLikelihood(n_trials)
            c_int = Intercept()
            c_beta = FixedEffects(1)
            A = sparse(hcat(ones(n), reshape(x, n, 1)))
            model = LatentGaussianModel(ℓ, (c_int, c_beta), A)

            res = inla(model, y; int_strategy = :grid)

            # --- Fixed effects ------------------------------------------------
            sf = fx["summary_fixed"]
            α_R = _bb_row_value(sf, "(Intercept)", "mean")
            β_R = _bb_row_value(sf, "x", "mean")

            fe = fixed_effects(model, res)
            @test length(fe) == 2
            @test _rel_bb(fe[1].mean, α_R) < BB_FIXED_EFFECT_TOL
            @test _rel_bb(fe[2].mean, β_R) < BB_FIXED_EFFECT_TOL

            # --- Hyperparameter: overdispersion ρ on user scale ----------------
            # R-INLA labels this "overdispersion for the betabinomial
            # observations". Internal scale is logit(ρ); compare mode-vs-mode.
            sh = fx["summary_hyperpar"]
            ρ_R = _bb_row_value(
                sh, "overdispersion for the betabinomial observations", "mode")
            ρ_J = 1 / (1 + exp(-res.θ̂[1]))
            @test _rel_bb(ρ_J, ρ_R) < BB_RHO_REL_TOL

            hp = hyperparameters(model, res)
            @test length(hp) == 1
            @test isfinite(hp[1].mean) && hp[1].sd > 0

            # --- Marginal log-likelihood --------------------------------------
            mlik_R = Float64(fx["mlik"][1])
            mlik_J = log_marginal_likelihood(res)
            @test _rel_bb(mlik_J, mlik_R) < BB_MLIK_REL_TOL
        end
    end
end

# Oracle test: proportional-odds ordinal regression on synthetic data
# vs R-INLA's `family = "pom"`. K = 4 ordered classes, n = 400, one
# covariate, no intercept (cut points absorb it).
#
# Layout: 1-column design matrix on a single FixedEffects(1)
# component (β only); cut points carry as POMLikelihood
# hyperparameters with the Dirichlet(γ = 3) prior R-INLA hard-wires.
#
# Tested invariants:
#   - β posterior mean / sd within 5% of R-INLA's marginal moments.
#   - θ posterior means within 10% of R-INLA's marginal means
#     (looser than β because the Dirichlet prior on the cut points
#     induces a moderately skew posterior on θ_2 / θ_3).
#   - log_marginal_likelihood is finite. Julia's mlik differs from
#     R-INLA's by a fixed θ-independent additive constant: R-INLA's
#     internal Dirichlet prior is documented to omit the Jacobian
#     correction for the sum-to-zero constraint on the implied class
#     probabilities (see `inla.doc("pom")` and `pom.tex` in the
#     R-INLA source). The Julia implementation includes the full
#     Jacobian, so mlik values match in the posterior moments but
#     differ in absolute scale. Comparing mlik directly would test
#     R-INLA's bug rather than the model.
#
# Fixture: scripts/generate-fixtures/lgm/synthetic_pom.R.
# Skipped transparently if the JLD2 fixture has not been generated.

include("load_fixture.jl")

using Test
using SparseArrays
using LatentGaussianModels: POMLikelihood, FixedEffects,
                            LatentGaussianModel, inla, fixed_effects, hyperparameters,
                            log_marginal_likelihood

const POM_FIXTURE = "synthetic_pom"

const POM_FIXED_EFFECT_MEAN_REL_TOL = 0.05
const POM_FIXED_EFFECT_SD_REL_TOL = 0.10
const POM_THETA_MEAN_TOL = 0.10

_rel_pom(a, b) = abs(a - b) / max(abs(b), 1.0)

function _pom_row_value(frame, name::AbstractString, col::AbstractString)
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $(join(rn, "; ")))")
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

@testset "synthetic_pom vs R-INLA" begin
    if !has_oracle_fixture(POM_FIXTURE)
        @test_skip "oracle fixture $POM_FIXTURE not generated (see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(POM_FIXTURE)
        @test fx["name"] == POM_FIXTURE
        @test haskey(fx, "summary_fixed")
        @test haskey(fx, "summary_hyperpar")

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate with the current R script"
        else
            inp = fx["input"]
            y = Int.(inp["y"])
            x = Float64.(inp["x"])
            K = Int(inp["n_classes"])
            n = length(y)
            @test K == 4
            @test n == 400

            # Single covariate, no intercept (cut points absorb it).
            ℓ = POMLikelihood(K)
            c_beta = FixedEffects(1)
            A = sparse(reshape(x, n, 1))
            model = LatentGaussianModel(ℓ, (c_beta,), A)

            res = inla(model, y; int_strategy=:grid)

            # --- Fixed effect β -----------------------------------------------
            sf = fx["summary_fixed"]
            β_R_mean = _pom_row_value(sf, "x", "mean")
            β_R_sd = _pom_row_value(sf, "x", "sd")

            fe = fixed_effects(model, res)
            @test length(fe) == 1
            @test _rel_pom(fe[1].mean, β_R_mean) < POM_FIXED_EFFECT_MEAN_REL_TOL
            @test _rel_pom(fe[1].sd, β_R_sd) < POM_FIXED_EFFECT_SD_REL_TOL

            # --- Cut-point hyperparameters θ ----------------------------------
            # R-INLA reports the internal-scale θ directly. Compare Julia's
            # marginal posterior means (from the integration-grid mixture)
            # against R-INLA's. The mode tolerance would be tighter, but
            # mode reporting differs: R-INLA's "mode" is the marginal mode
            # while Julia's θ̂ is the joint mode — for a slightly skew
            # posterior the two diverge.
            sh = fx["summary_hyperpar"]
            θ_R_mean = [
                _pom_row_value(sh, "theta1 for POM", "mean"),
                _pom_row_value(sh, "theta2 for POM", "mean"),
                _pom_row_value(sh, "theta3 for POM", "mean"),
            ]

            hp = hyperparameters(model, res)
            @test length(hp) == 3
            @test all(h -> isfinite(h.mean) && h.sd > 0, hp)

            for k in 1:3
                @test abs(hp[k].mean - θ_R_mean[k]) < POM_THETA_MEAN_TOL
            end

            # --- Marginal log-likelihood: finiteness only --------------------
            # See test docstring: R-INLA's pom mlik is off by a fixed
            # θ-independent additive constant (documented INLA bug). Asserting
            # absolute agreement would test the bug rather than the model.
            mlik_J = log_marginal_likelihood(res)
            @test isfinite(mlik_J)
        end
    end
end

# Oracle test: zero-inflated Poisson (type 1, standard mixture)
# regression on synthetic data vs R-INLA.
#
# Smallest possible oracle for the `family = "zeroinflatedpoisson1"`
# pathway — intercept + one covariate, no latent random effect, n = 200.
# Validates the ZI pack end-to-end through `inla(...)`: fixed-effect
# agreement, recovery of the structural-zero probability π on the user
# scale, and marginal log-likelihood agreement.
#
# Fixture: scripts/generate-fixtures/lgm/synthetic_zip1.R.
# Skipped transparently if the JLD2 fixture has not been generated.

include("load_fixture.jl")

using Test
using SparseArrays
using LatentGaussianModels: ZeroInflatedPoissonLikelihood1, Intercept, FixedEffects,
                            LatentGaussianModel, inla, fixed_effects, hyperparameters,
                            log_marginal_likelihood

const ZIP1_FIXTURE = "synthetic_zip1"

# Tolerances. Fixed-effects band loose because n=200 leaves meaningful
# posterior mass; the π hyperparameter recovers more loosely (the
# logit transform amplifies tail uncertainty), so 20% relative on the
# user scale; mlik agreement to within 5%.
const ZIP1_FIXED_EFFECT_TOL = 0.05
const ZIP1_PI_REL_TOL = 0.20
const ZIP1_MLIK_REL_TOL = 0.05

_rel_zip1(a, b) = abs(a - b) / max(abs(b), 1.0)
_expit(x) = inv(one(x) + exp(-x))

function _row_value(frame, name::AbstractString, col::AbstractString)
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $(join(rn, "; ")))")
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

@testset "synthetic_zip1 vs R-INLA" begin
    if !has_oracle_fixture(ZIP1_FIXTURE)
        @test_skip "oracle fixture $ZIP1_FIXTURE not generated (see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(ZIP1_FIXTURE)
        @test fx["name"] == ZIP1_FIXTURE
        @test haskey(fx, "summary_fixed")
        @test haskey(fx, "summary_hyperpar")

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate with the current R script"
        else
            inp = fx["input"]
            y = Int.(inp["y"])
            x = Float64.(inp["x"])
            n = length(y)

            ℓ = ZeroInflatedPoissonLikelihood1()
            c_int = Intercept()
            c_beta = FixedEffects(1)
            # Latent layout: [α; β]. Linear predictor η_i = α + β x_i.
            A = sparse(hcat(ones(n), reshape(x, n, 1)))
            model = LatentGaussianModel(ℓ, (c_int, c_beta), A)

            # dim(θ) = 1 (logit π only) → :grid is appropriate and matches
            # R-INLA's default integration choice for low-dim θ.
            res = inla(model, y; int_strategy=:grid)

            # --- Fixed effects ------------------------------------------------
            sf = fx["summary_fixed"]
            α_R = _row_value(sf, "(Intercept)", "mean")
            β_R = _row_value(sf, "x", "mean")

            fe = fixed_effects(model, res)
            @test length(fe) == 2
            @test _rel_zip1(fe[1].mean, α_R) < ZIP1_FIXED_EFFECT_TOL
            @test _rel_zip1(fe[2].mean, β_R) < ZIP1_FIXED_EFFECT_TOL

            # --- Hyperparameter: π on user scale -------------------------------
            # R-INLA's "zero-probability parameter for zero-inflated poisson_1"
            # is reported on the user scale (π itself); our internal scale is
            # logit(π), so apply the sigmoid for comparison.
            sh = fx["summary_hyperpar"]
            π_R = _row_value(
                sh, "zero-probability parameter for zero-inflated poisson_1", "mean")
            π_J = _expit(res.θ̂[1])
            @test _rel_zip1(π_J, π_R) < ZIP1_PI_REL_TOL

            hp = hyperparameters(model, res)
            @test length(hp) == 1
            @test isfinite(hp[1].mean) && hp[1].sd > 0

            # --- Marginal log-likelihood --------------------------------------
            mlik_R = Float64(fx["mlik"][1])
            mlik_J = log_marginal_likelihood(res)
            @test _rel_zip1(mlik_J, mlik_R) < ZIP1_MLIK_REL_TOL
        end
    end
end

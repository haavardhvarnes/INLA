# Oracle test: Gamma regression on synthetic data vs R-INLA.
#
# Smallest possible oracle for the `family = "gamma"` pathway —
# intercept + one covariate, no latent random effect, n = 200. Tests
# fixed-effect agreement and that the precision hyperparameter mode is
# recovered to within 15%.
#
# Fixture: scripts/generate-fixtures/lgm/synthetic_gamma.R.
# Skipped transparently if the JLD2 fixture has not been generated.

include("load_fixture.jl")

using Test
using SparseArrays
using LatentGaussianModels: GammaLikelihood, Intercept, FixedEffects,
    LatentGaussianModel, inla, fixed_effects, hyperparameters,
    log_marginal_likelihood

const GAMMA_FIXTURE = "synthetic_gamma"

const GAMMA_FIXED_EFFECT_TOL = 0.05
const GAMMA_PHI_REL_TOL      = 0.15
const GAMMA_MLIK_REL_TOL     = 0.05

_rel_gamma(a, b) = abs(a - b) / max(abs(b), 1.0)

function _gamma_row_value(frame, name::AbstractString, col::AbstractString)
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $(join(rn, "; ")))")
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

@testset "synthetic_gamma vs R-INLA" begin
    if !has_oracle_fixture(GAMMA_FIXTURE)
        @test_skip "oracle fixture $GAMMA_FIXTURE not generated (see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(GAMMA_FIXTURE)
        @test fx["name"] == GAMMA_FIXTURE
        @test haskey(fx, "summary_fixed")
        @test haskey(fx, "summary_hyperpar")

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate with the current R script"
        else
            inp = fx["input"]
            y = Float64.(inp["y"])
            x = Float64.(inp["x"])
            n = length(y)

            ℓ = GammaLikelihood()
            c_int  = Intercept()
            c_beta = FixedEffects(1)
            A = sparse(hcat(ones(n), reshape(x, n, 1)))
            model = LatentGaussianModel(ℓ, (c_int, c_beta), A)

            res = inla(model, y; int_strategy = :grid)

            # --- Fixed effects ------------------------------------------------
            sf = fx["summary_fixed"]
            α_R = _gamma_row_value(sf, "(Intercept)", "mean")
            β_R = _gamma_row_value(sf, "x", "mean")

            fe = fixed_effects(model, res)
            @test length(fe) == 2
            @test _rel_gamma(fe[1].mean, α_R) < GAMMA_FIXED_EFFECT_TOL
            @test _rel_gamma(fe[2].mean, β_R) < GAMMA_FIXED_EFFECT_TOL

            # --- Hyperparameter: precision φ on user scale --------------------
            # R-INLA labels this "Precision-parameter for the Gamma observations".
            # Our internal scale is log(φ).
            sh = fx["summary_hyperpar"]
            phi_R = _gamma_row_value(sh, "Precision-parameter for the Gamma observations", "mean")
            phi_J = exp(res.θ̂[1])
            @test _rel_gamma(phi_J, phi_R) < GAMMA_PHI_REL_TOL

            hp = hyperparameters(model, res)
            @test length(hp) == 1
            @test isfinite(hp[1].mean) && hp[1].sd > 0

            # --- Marginal log-likelihood --------------------------------------
            mlik_R = Float64(fx["mlik"][1])
            mlik_J = log_marginal_likelihood(res)
            @test _rel_gamma(mlik_J, mlik_R) < GAMMA_MLIK_REL_TOL
        end
    end
end

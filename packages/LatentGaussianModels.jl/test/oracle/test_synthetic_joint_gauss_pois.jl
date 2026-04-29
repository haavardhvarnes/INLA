# Oracle test: joint Gaussian + Poisson regression on synthetic data
# vs R-INLA. Smallest oracle for the multi-likelihood pathway introduced
# in Phase G PR2 (ADR-017): two observation blocks share one IID(n)
# random effect plus a shared intercept.
#
# Fixture: scripts/generate-fixtures/lgm/synthetic_joint_gauss_pois.R.
# Skipped transparently if the JLD2 fixture has not been generated.

include("load_fixture.jl")

using Test
using SparseArrays
using LinearAlgebra: I
using LatentGaussianModels: GaussianLikelihood, PoissonLikelihood,
    Intercept, IID, LinearProjector, StackedMapping,
    LatentGaussianModel, inla, fixed_effects, hyperparameters,
    log_marginal_likelihood, n_likelihoods

const JOINT_FIXTURE = "synthetic_joint_gauss_pois"

# Tolerances. n = 50 paired observations leaves meaningful posterior
# mass on each parameter. Comparison is `exp(θ̂_J)` (user-scale value at
# the internal-scale mode) vs R-INLA's `mean` column (user-scale
# posterior mean). For asymmetric log-precision posteriors these differ
# systematically — the τ_g band is widened to absorb that gap on n = 50.
const JOINT_FIXED_EFFECT_TOL = 0.10
const JOINT_TAU_G_REL_TOL    = 0.35
const JOINT_TAU_U_REL_TOL    = 0.10
const JOINT_MLIK_REL_TOL     = 0.05

_rel_joint(a, b) = abs(a - b) / max(abs(b), 1.0)

function _row_joint(frame, name::AbstractString, col::AbstractString)
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $(join(rn, "; ")))")
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

@testset "synthetic_joint_gauss_pois vs R-INLA" begin
    if !has_oracle_fixture(JOINT_FIXTURE)
        @test_skip "oracle fixture $JOINT_FIXTURE not generated (see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(JOINT_FIXTURE)
        @test fx["name"] == JOINT_FIXTURE
        @test haskey(fx, "summary_fixed")
        @test haskey(fx, "summary_hyperpar")

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate with the current R script"
        else
            inp = fx["input"]
            y_g = Float64.(inp["y_g"])
            y_p = Int.(inp["y_p"])
            n = length(y_g)
            @test length(y_p) == n

            # Two blocks share the latent x = [α; u_1, ..., u_n].
            # Each block's mapping is [1 | I_n].
            A_g = sparse(hcat(ones(n), Matrix{Float64}(I, n, n)))
            A_p = sparse(hcat(ones(n), Matrix{Float64}(I, n, n)))
            mapping = StackedMapping(
                (LinearProjector(A_g), LinearProjector(A_p)),
                [1:n, (n + 1):(2n)],
            )

            ℓ_g = GaussianLikelihood()
            ℓ_p = PoissonLikelihood(; E = fill(1.0, n))
            model = LatentGaussianModel((ℓ_g, ℓ_p), (Intercept(), IID(n)), mapping)
            @test n_likelihoods(model) == 2

            # Stacked observation vector: Gaussian rows then Poisson rows.
            y = vcat(y_g, Float64.(y_p))

            # dim(θ) = 2 (τ_g + τ_u) → :auto picks Grid.
            res = inla(model, y; int_strategy = :grid)

            # --- Fixed effects ------------------------------------------------
            sf = fx["summary_fixed"]
            α_R = _row_joint(sf, "intercept", "mean")
            fe = fixed_effects(model, res)
            @test length(fe) == 1
            @test _rel_joint(fe[1].mean, α_R) < JOINT_FIXED_EFFECT_TOL

            # --- Hyperparameters ----------------------------------------------
            sh = fx["summary_hyperpar"]
            τ_g_R = _row_joint(sh, "Precision for the Gaussian observations", "mean")
            τ_u_R = _row_joint(sh, "Precision for idx", "mean")
            τ_g_J = exp(res.θ̂[1])           # likelihood block 1: Gaussian τ
            τ_u_J = exp(res.θ̂[2])           # IID component τ_u
            @test _rel_joint(τ_g_J, τ_g_R) < JOINT_TAU_G_REL_TOL
            @test _rel_joint(τ_u_J, τ_u_R) < JOINT_TAU_U_REL_TOL

            hp = hyperparameters(model, res)
            @test length(hp) == 2

            # --- Marginal log-likelihood --------------------------------------
            mlik_R = Float64(fx["mlik"][1])
            mlik_J = log_marginal_likelihood(res)
            @test _rel_joint(mlik_J, mlik_R) < JOINT_MLIK_REL_TOL
        end
    end
end

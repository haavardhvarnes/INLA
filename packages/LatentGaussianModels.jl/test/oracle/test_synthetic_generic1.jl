# Oracle test: Gaussian regression with a `Generic1` random effect vs
# R-INLA. R-INLA's native `generic1` carries a β-mixing parameter we
# defer beyond v0.1; we validate only the eigenvalue rescaling step
# (Q = τ · C̃ with C̃ = C / λ_max(C)). The R script does the rescaling
# manually and feeds C̃ to `model = "generic0"`; the Julia side calls
# `Generic1(C)` (which performs the same rescaling at construction).
#
# Fixture: scripts/generate-fixtures/lgm/synthetic_generic1.R.

include("load_fixture.jl")

using Test
using LinearAlgebra: Symmetric, eigvals
using SparseArrays
using LatentGaussianModels: GaussianLikelihood, Generic1, GammaPrecision,
                            LatentGaussianModel, inla, hyperparameters,
                            log_marginal_likelihood

const G1_FIXTURE = "synthetic_generic1"

const G1_PREC_REL_TOL = 0.20
const G1_LIK_REL_TOL = 0.20
const G1_MLIK_REL_TOL = 0.05

_rel_g1(a, b) = abs(a - b) / max(abs(b), 1.0)

function _g1_row(frame, name::AbstractString, col::AbstractString)
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $(join(rn, "; ")))")
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

@testset "synthetic_generic1 vs R-INLA" begin
    if !has_oracle_fixture(G1_FIXTURE)
        @test_skip "oracle fixture $G1_FIXTURE not generated (see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(G1_FIXTURE)
        @test fx["name"] == G1_FIXTURE

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate with the current R script"
        else
            inp = fx["input"]
            n_obs = Int(inp["n_obs"])
            n_lat = Int(inp["n_lat"])
            y = Float64.(inp["y"])
            A_vec = Float64.(inp["A"])
            @test length(A_vec) == n_obs * n_lat
            A_dense = reshape(A_vec, n_lat, n_obs)'
            A = sparse(Matrix{Float64}(A_dense))
            C = SparseMatrixCSC{Float64, Int}(inp["C"])

            # R-INLA's `family = "gaussian"` default precision prior is
            # `loggamma(1, 5e-5)`; pass it explicitly for parity (Julia's
            # `GaussianLikelihood` defaults to a PC prior).
            ℓ = GaussianLikelihood(hyperprior=GammaPrecision(1.0, 5.0e-5))
            c_g1 = Generic1(C; rankdef=0,
                hyperprior=GammaPrecision(1.0, 5.0e-5))
            # Sanity check the rescaling actually happened.
            @test maximum(eigvals(Symmetric(Matrix(c_g1.R))))≈1.0 atol=1.0e-10
            @test c_g1.λ_max_original≈Float64(inp["lambda_max"]) atol=1.0e-10

            model = LatentGaussianModel(ℓ, (c_g1,), A)
            res = inla(model, y; int_strategy=:grid)

            # --- Hyperparameters -----------------------------------------
            sh = fx["summary_hyperpar"]
            τ_g1_R = _g1_row(sh, "Precision for idx", "mean")
            τ_lik_R = _g1_row(sh, "Precision for the Gaussian observations", "mean")

            τ_lik_J = exp(res.θ̂[1])
            τ_g1_J = exp(res.θ̂[2])
            @test _rel_g1(τ_lik_J, τ_lik_R) < G1_LIK_REL_TOL
            @test _rel_g1(τ_g1_J, τ_g1_R) < G1_PREC_REL_TOL

            hp = hyperparameters(model, res)
            @test length(hp) == 2
            @test all(isfinite(r.mean) && r.sd > 0 for r in hp)

            # --- Marginal log-likelihood ---------------------------------
            mlik_R = Float64(fx["mlik"][1])
            mlik_J = log_marginal_likelihood(res)
            @test _rel_g1(mlik_J, mlik_R) < G1_MLIK_REL_TOL
            @test isfinite(mlik_J)
        end
    end
end

# Oracle test: Gaussian regression with a `Generic0` random effect vs
# R-INLA. Validates the LGM `Generic0` component against R-INLA's
# `model = "generic0"` end-to-end on synthetic data.
#
# Fixture: scripts/generate-fixtures/lgm/synthetic_generic0.R.
# n_lat = 8, n_obs = 30, random SPD structure matrix C.

include("load_fixture.jl")

using Test
using SparseArrays
using LatentGaussianModels: GaussianLikelihood, Generic0, GammaPrecision,
                            LatentGaussianModel, inla, hyperparameters,
                            log_marginal_likelihood

const G0_FIXTURE = "synthetic_generic0"

const G0_PREC_REL_TOL = 0.20
const G0_LIK_REL_TOL = 0.20
const G0_MLIK_REL_TOL = 0.05

_rel_g0(a, b) = abs(a - b) / max(abs(b), 1.0)

function _g0_row(frame, name::AbstractString, col::AbstractString)
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $(join(rn, "; ")))")
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

@testset "synthetic_generic0 vs R-INLA" begin
    if !has_oracle_fixture(G0_FIXTURE)
        @test_skip "oracle fixture $G0_FIXTURE not generated (see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(G0_FIXTURE)
        @test fx["name"] == G0_FIXTURE

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate with the current R script"
        else
            inp = fx["input"]
            n_obs = Int(inp["n_obs"])
            n_lat = Int(inp["n_lat"])
            y = Float64.(inp["y"])
            # A is stored row-major (`as.numeric(t(A))`).
            A_vec = Float64.(inp["A"])
            @test length(A_vec) == n_obs * n_lat
            A_dense = reshape(A_vec, n_lat, n_obs)'  # row-major -> n_obs × n_lat
            A = sparse(Matrix{Float64}(A_dense))
            C = SparseMatrixCSC{Float64, Int}(inp["C"])

            # R-INLA's `family = "gaussian"` default precision prior is
            # `loggamma(1, 5e-5)`; pass it explicitly for parity (Julia's
            # `GaussianLikelihood` defaults to a PC prior).
            ℓ = GaussianLikelihood(hyperprior=GammaPrecision(1.0, 5.0e-5))
            c_g0 = Generic0(C; rankdef=0,
                hyperprior=GammaPrecision(1.0, 5.0e-5))
            model = LatentGaussianModel(ℓ, (c_g0,), A)

            res = inla(model, y; int_strategy=:grid)

            # --- Hyperparameters -----------------------------------------
            sh = fx["summary_hyperpar"]
            τ_g0_R = _g0_row(sh, "Precision for idx", "mean")
            τ_lik_R = _g0_row(sh, "Precision for the Gaussian observations", "mean")

            # Internal θ layout: [likelihood; component] →
            # θ[1] = log τ_lik, θ[2] = log τ_idx.
            τ_lik_J = exp(res.θ̂[1])
            τ_g0_J = exp(res.θ̂[2])
            @test _rel_g0(τ_lik_J, τ_lik_R) < G0_LIK_REL_TOL
            @test _rel_g0(τ_g0_J, τ_g0_R) < G0_PREC_REL_TOL

            hp = hyperparameters(model, res)
            @test length(hp) == 2
            @test all(isfinite(r.mean) && r.sd > 0 for r in hp)

            # --- Marginal log-likelihood ---------------------------------
            mlik_R = Float64(fx["mlik"][1])
            mlik_J = log_marginal_likelihood(res)
            @test _rel_g0(mlik_J, mlik_R) < G0_MLIK_REL_TOL
            @test isfinite(mlik_J)
        end
    end
end

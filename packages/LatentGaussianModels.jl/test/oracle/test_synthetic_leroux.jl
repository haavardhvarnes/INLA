# Oracle test: Gaussian observations on a 4x4 grid + Leroux CAR random
# effect, vs R-INLA's `model = "besagproper2"`.
#
# Fixture: scripts/generate-fixtures/lgm/synthetic_leroux.R.

include("load_fixture.jl")

using Test
using SparseArrays
using LatentGaussianModels: GaussianLikelihood, Intercept, Leroux,
    GammaPrecision, LogitBeta, PCPrecision, LatentGaussianModel, inla,
    hyperparameters, fixed_effects, log_marginal_likelihood
using GMRFs: GMRFGraph

const LRX_FIXTURE = "synthetic_leroux"

const LRX_FIXED_REL_TOL = 0.10
const LRX_PREC_REL_TOL  = 0.20
const LRX_LIK_REL_TOL   = 0.20
const LRX_LAMBDA_TOL    = 0.20
const LRX_MLIK_REL_TOL  = 0.05

_rel_lrx(a, b) = abs(a - b) / max(abs(b), 1.0)

function _lrx_row(frame, name::AbstractString, col::AbstractString)
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $(join(rn, "; ")))")
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

@testset "synthetic_leroux vs R-INLA" begin
    if !has_oracle_fixture(LRX_FIXTURE)
        @test_skip "oracle fixture $LRX_FIXTURE not generated (see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(LRX_FIXTURE)
        @test fx["name"] == LRX_FIXTURE

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate with the current R script"
        else
            inp     = fx["input"]
            n       = Int(inp["n"])
            n_obs   = Int(inp["n_obs"])
            y       = Float64.(inp["y"])
            region  = Int.(inp["region"])
            W       = SparseMatrixCSC{Float64, Int}(inp["W"])

            ℓ = GaussianLikelihood(hyperprior = GammaPrecision(1.0, 5.0e-5))
            α = Intercept()
            lrx = Leroux(GMRFGraph(W);
                         hyperprior_tau = PCPrecision(1.0, 0.01),
                         hyperprior_rho = LogitBeta(1.0, 1.0))

            # A: column of ones for the intercept; identity-on-region for
            # Leroux. Row i has a 1 at α and at region[i].
            rows_α    = collect(1:n_obs)
            cols_α    = ones(Int, n_obs)
            A_α       = sparse(rows_α, cols_α, ones(Float64, n_obs), n_obs, 1)
            rows_lrx  = collect(1:n_obs)
            cols_lrx  = region
            A_lrx     = sparse(rows_lrx, cols_lrx, ones(Float64, n_obs), n_obs, n)
            A         = hcat(A_α, A_lrx)

            model = LatentGaussianModel(ℓ, (α, lrx), A)
            res   = inla(model, y; int_strategy = :grid)

            # --- Fixed effect ----------------------------------------------
            sf = fx["summary_fixed"]
            α_R = _lrx_row(sf, "(Intercept)", "mean")
            fe = fixed_effects(model, res)
            α_J = fe[1].mean
            @test _rel_lrx(α_J, α_R) < LRX_FIXED_REL_TOL

            # --- Hyperparameters -------------------------------------------
            sh = fx["summary_hyperpar"]
            τ_lik_R = _lrx_row(sh, "Precision for the Gaussian observations", "mean")
            τ_lrx_R = _lrx_row(sh, "Precision for region", "mean")
            λ_lrx_R = _lrx_row(sh, "Lambda for region", "mean")

            # Internal θ layout: [log τ_lik, log τ_leroux, logit ρ_leroux].
            τ_lik_J = exp(res.θ̂[1])
            τ_lrx_J = exp(res.θ̂[2])
            ρ_lrx_J = inv(1 + exp(-res.θ̂[3]))

            @test _rel_lrx(τ_lik_J, τ_lik_R) < LRX_LIK_REL_TOL
            @test _rel_lrx(τ_lrx_J, τ_lrx_R) < LRX_PREC_REL_TOL
            @test abs(ρ_lrx_J - λ_lrx_R)     < LRX_LAMBDA_TOL

            hp = hyperparameters(model, res)
            @test length(hp) == 3
            @test all(isfinite(r.mean) && r.sd > 0 for r in hp)

            # --- Marginal log-likelihood -----------------------------------
            mlik_R = Float64(fx["mlik"][1])
            mlik_J = log_marginal_likelihood(res)
            @test _rel_lrx(mlik_J, mlik_R) < LRX_MLIK_REL_TOL
        end
    end
end

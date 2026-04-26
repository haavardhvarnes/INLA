# Oracle test: Gaussian time series with intercept + Seasonal random
# effect vs R-INLA. Validates the LGM `Seasonal` component end-to-end
# against R-INLA's `model = "seasonal"` on a synthetic period-6 signal.
#
# Known divergence — constraint convention. R-INLA applies a single
# sum-to-zero constraint to seasonal random effects and lets the
# likelihood identify the (s-1)-dim period-s null space. Our
# SeasonalGMRF applies the full s-1 null-space-fixing constraints
# (required by the current Laplace contract `null(Q) = range(C^T)`).
# The intercept, τ_seas, and the marginal log-likelihood all agree
# closely; only τ_lik differs because our `b` cannot absorb the
# periodic component, so ε absorbs the seasonal variance. The
# τ_lik check is `@test_broken` until the Laplace pipeline supports
# `null(Q) ⊋ range(C^T)`.
#
# Fixture: scripts/generate-fixtures/lgm/synthetic_seasonal.R.

include("load_fixture.jl")

using Test
using SparseArrays
using LatentGaussianModels: GaussianLikelihood, Intercept, Seasonal,
    GammaPrecision, LatentGaussianModel, inla, hyperparameters,
    fixed_effects, log_marginal_likelihood

const SEAS_FIXTURE = "synthetic_seasonal"

const SEAS_FIXED_REL_TOL = 0.10
const SEAS_PREC_REL_TOL  = 0.20
const SEAS_LIK_REL_TOL   = 0.20
const SEAS_MLIK_REL_TOL  = 0.05

_rel_seas(a, b) = abs(a - b) / max(abs(b), 1.0)

function _seas_row(frame, name::AbstractString, col::AbstractString)
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $(join(rn, "; ")))")
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

@testset "synthetic_seasonal vs R-INLA" begin
    if !has_oracle_fixture(SEAS_FIXTURE)
        @test_skip "oracle fixture $SEAS_FIXTURE not generated (see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(SEAS_FIXTURE)
        @test fx["name"] == SEAS_FIXTURE

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate with the current R script"
        else
            inp    = fx["input"]
            n      = Int(inp["n"])
            period = Int(inp["period"])
            y      = Float64.(inp["y"])

            # Match R-INLA defaults: loggamma(1, 5e-5) on both likelihood
            # and seasonal precisions.
            ℓ = GaussianLikelihood(hyperprior = GammaPrecision(1.0, 5.0e-5))
            α = Intercept()
            seas = Seasonal(n; period = period,
                            hyperprior = GammaPrecision(1.0, 5.0e-5))

            # Latent layout: [α; b_1, …, b_n].
            # A_α: column of ones; A_seas: identity on the n latent slots.
            A_α    = sparse(ones(n, 1))
            A_seas = sparse(I, n, n)
            A      = hcat(A_α, A_seas)

            model = LatentGaussianModel(ℓ, (α, seas), A)
            res = inla(model, y; int_strategy = :grid)

            # --- Fixed effects -------------------------------------------
            sf = fx["summary_fixed"]
            α_R = _seas_row(sf, "(Intercept)", "mean")
            fe = fixed_effects(model, res)
            α_J = fe[1].mean
            @test _rel_seas(α_J, α_R) < SEAS_FIXED_REL_TOL

            # --- Hyperparameters -----------------------------------------
            sh = fx["summary_hyperpar"]
            τ_lik_R = _seas_row(sh, "Precision for the Gaussian observations", "mean")
            τ_seas_R = _seas_row(sh, "Precision for t", "mean")

            # Internal θ layout: [log τ_lik; log τ_seas].
            τ_lik_J  = exp(res.θ̂[1])
            τ_seas_J = exp(res.θ̂[2])
            # τ_lik mismatch is the constraint-convention symptom: R-INLA
            # uses 1 sum-to-zero, we use s-1 null-space constraints, so our
            # b ≈ 0 and ε absorbs the seasonal variance.
            @test_broken _rel_seas(τ_lik_J,  τ_lik_R)  < SEAS_LIK_REL_TOL
            @test _rel_seas(τ_seas_J, τ_seas_R) < SEAS_PREC_REL_TOL

            hp = hyperparameters(model, res)
            @test length(hp) == 2
            @test all(isfinite(r.mean) && r.sd > 0 for r in hp)

            # --- Marginal log-likelihood ---------------------------------
            mlik_R = Float64(fx["mlik"][1])
            mlik_J = log_marginal_likelihood(res)
            @test _rel_seas(mlik_J, mlik_R) < SEAS_MLIK_REL_TOL
        end
    end
end

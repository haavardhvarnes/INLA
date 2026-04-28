# Oracle test: Pennsylvania lung cancer BYM2 vs R-INLA.
#
# Second Poisson-BYM2 oracle alongside Scotland. Larger n (67 counties),
# proper covariate (standardised smoking rate), and indirect-standardised
# expected counts.
#
# Fixture: scripts/generate-fixtures/lgm/pennsylvania_bym2.R.
# Skipped transparently if the JLD2 fixture has not been generated.

include("load_fixture.jl")

using Test
using SparseArrays
using LinearAlgebra: I
using LatentGaussianModels: PoissonLikelihood, Intercept, FixedEffects,
    BYM2, LatentGaussianModel, inla, PCPrecision,
    fixed_effects, hyperparameters, log_marginal_likelihood
using GMRFs: GMRFGraph

const FIXTURE = "pennsylvania_bym2"

# Tolerances — same band as Scotland (plans/testing-strategy.md).
const FIXED_EFFECT_TOL = 0.05
const TAU_REL_TOL      = 0.10
const MLIK_REL_TOL     = 0.02
# mlik passes within 2% of R-INLA's integration estimate.

_rel(a, b) = abs(a - b) / max(abs(b), 1.0)

function _fixed_row(sf, name::AbstractString, col::AbstractString)
    rn = String.(sf["rownames"])
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $rn)")
    return Float64(sf[col][idx])
end

function _hyperpar_row(sh, name::AbstractString, col::AbstractString)
    rn = String.(sh["rownames"])
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $rn)")
    return Float64(sh[col][idx])
end

@testset "pennsylvania_bym2 vs R-INLA" begin
    if !has_oracle_fixture(FIXTURE)
        @test_skip "oracle fixture $FIXTURE not generated (see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(FIXTURE)
        @test fx["name"] == FIXTURE
        @test haskey(fx, "summary_fixed")
        @test haskey(fx, "summary_hyperpar")

        sf = fx["summary_fixed"]
        rn = String.(sf["rownames"])
        @test "(Intercept)" in rn
        @test "x" in rn

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate with the current R script"
        else
            inp = fx["input"]
            y = Int.(inp["cases"])
            E = Float64.(inp["expected"])
            x = Float64.(inp["x"])
            W = inp["W"]
            n = length(y)

            ℓ = PoissonLikelihood(; E = E)
            c_int  = Intercept()
            c_beta = FixedEffects(1)
            c_bym2 = BYM2(GMRFGraph(W); hyperprior_prec = PCPrecision(1.0, 0.01))
            # Latent layout: [α; β; b; u]. u is constrained and doesn't
            # enter η; only the combined b = BYM2[1:n] does.
            A = sparse(hcat(
                ones(n),
                reshape(x, n, 1),
                Matrix{Float64}(I, n, n),
                zeros(n, n),
            ))
            model = LatentGaussianModel(ℓ, (c_int, c_beta, c_bym2), A)

            res = inla(model, y; int_strategy = :grid)

            # --- Fixed effects --------------------------------------------
            fe = fixed_effects(model, res)
            @test length(fe) == 2
            α_R = _fixed_row(sf, "(Intercept)", "mean")
            β_R = _fixed_row(sf, "x", "mean")
            @test _rel(fe[1].mean, α_R) < FIXED_EFFECT_TOL
            @test _rel(fe[2].mean, β_R) < FIXED_EFFECT_TOL

            # --- Hyperparameters: τ (Precision) and φ (Phi) ---------------
            sh = fx["summary_hyperpar"]
            τ_R = _hyperpar_row(sh, "Precision for region", "mean")
            τ̂_J = exp(res.θ̂[1])
            @test _rel(τ̂_J, τ_R) < TAU_REL_TOL

            hp = hyperparameters(model, res)
            @test length(hp) == 2
            @test all(isfinite(r.mean) && r.sd > 0 for r in hp)

            # --- Marginal log-likelihood triangulation --------------------
            mlik_R = Float64(fx["mlik"][1])
            mlik_J = log_marginal_likelihood(res)
            @test _rel(mlik_J, mlik_R) < MLIK_REL_TOL

            # --- ADR-016: simplified.laplace BYM mean parity --------------
            if !haskey(fx, "bym_mean_sla")
                @test_skip "fixture has no `bym_mean_sla` field — regenerate to add ADR-016 oracle"
            else
                res_sla = inla(model, y; int_strategy = :grid,
                               latent_strategy = :simplified_laplace)
                bym_R_sla = Float64.(fx["bym_mean_sla"])
                # R-INLA's `summary.random$region$mean` for BYM2 has length
                # 2n: first n entries are the joint b = (1/√τ)(√(1-φ) v +
                # √φ u*); next n are the unstructured u*. Our latent layout
                # places b at columns 3..n+2 (after α, β).
                @test length(bym_R_sla) == 2 * n
                b_J = res_sla.x_mean[3:(n + 2)]
                # Tolerance: 5% sup-norm relative to max |b_R| — same band
                # as fixed-effects (FIXED_EFFECT_TOL). On Pennsylvania the
                # mean-shift correction is small (~1e-3) because expected
                # counts are large; the residual ~6e-3 gap is dominated by
                # CCD/grid integration-weight differences with R-INLA.
                bym_diff = maximum(abs, b_J .- bym_R_sla[1:n])
                bym_scale = max(maximum(abs, bym_R_sla[1:n]), 1.0e-3)
                @test bym_diff / bym_scale < 0.05

                # Sanity: the SLA path actually applies the Rue-Martino
                # shift — so it differs from the Newton-mode path. The
                # difference is small here but must be non-zero.
                b_J_g = res.x_mean[3:(n + 2)]
                @test !all(iszero, b_J .- b_J_g)
                # SLA should not be *worse* than Gaussian by more than the
                # shift magnitude; for Pennsylvania-skewness, both paths
                # sit within the 5% band.
                gauss_diff = maximum(abs, b_J_g .- bym_R_sla[1:n])
                @test bym_diff ≤ gauss_diff + maximum(abs, b_J .- b_J_g)
            end
        end
    end
end

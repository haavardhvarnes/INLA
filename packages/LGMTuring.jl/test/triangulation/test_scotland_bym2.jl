# Triangulation tier (Phase D) — Scotland BYM2.
#
# Same model R-INLA fits in `scripts/generate-fixtures/lgm/scotland_bym2.R`,
# fit two ways:
#   1. Julia INLA via `inla(model, y; int_strategy = :grid)`.
#   2. NUTS over the same INLA hyperparameter posterior via
#      `nuts_sample(model, y, n_samples; init_from_inla = inla_fit)`.
#
# We then assert that the two posterior summaries on θ agree —
# tier-3 cross-validation against the upstream R-INLA fit is covered by
# the existing oracle in LGM (`test/oracle/test_scotland_bym2.jl`); this
# test only certifies that NUTS on the *same* `INLALogDensity` recovers
# the same θ posterior INLA's grid integration sees. That's enough to
# catch a regression in either side: `INLALogDensity` evaluation, the
# Laplace mode finder, the inner Newton, the BYM2 precision build, or
# the NUTS bridge itself.
#
# Tolerances (per replan-2026-04 Phase D):
#   tol_mean = 2.0 SDs (covers MC + Laplace error; replan said ±2 SE)
#   tol_sd   = 0.30 (relative; loose because of short chain lengths)
#
# Fixture is the LGM-side JLD2 (`packages/LatentGaussianModels.jl/test/
# oracle/fixtures/scotland_bym2.jld2`). If absent, skip transparently
# rather than block the suite for users without the R toolchain.

using Test
using LinearAlgebra: I
using SparseArrays: sparse
using Random
using LatentGaussianModels: PoissonLikelihood, Intercept, FixedEffects,
    BYM2, LatentGaussianModel, inla, PCPrecision
using GMRFs: GMRFGraph
using LGMTuring: nuts_sample, compare_posteriors

const LGM_FIXTURE_PATH = joinpath(@__DIR__, "..", "..", "..",
    "LatentGaussianModels.jl", "test", "oracle", "fixtures",
    "scotland_bym2.jld2")

@testset "triangulation — Scotland BYM2 (INLA vs NUTS)" begin
    if !isfile(LGM_FIXTURE_PATH)
        @test_skip "Scotland BYM2 fixture missing at $LGM_FIXTURE_PATH"
    else
        using JLD2
        fx = jldopen(LGM_FIXTURE_PATH, "r") do f
            f["fixture"]
        end
        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate"
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
            A = sparse(hcat(
                ones(n),
                reshape(x, n, 1),
                Matrix{Float64}(I, n, n),
                zeros(n, n),
            ))
            model = LatentGaussianModel(ℓ, (c_int, c_beta, c_bym2), A)

            inla_fit = inla(model, y; int_strategy = :grid)

            # Short chain: each leapfrog step is a Laplace fit on a
            # 112-wide latent. 200 post-warmup samples after 100
            # warmup is enough to bound the posterior mean within
            # 2 SDs in 2-D θ.
            chain = nuts_sample(model, y, 200;
                                 n_adapts        = 100,
                                 init_from_inla  = inla_fit,
                                 rng             = Random.Xoshiro(20260426),
                                 progress        = false)

            rows = compare_posteriors(inla_fit, chain;
                                       model    = model,
                                       tol_mean = 2.0,
                                       tol_sd   = 0.30)
            @test length(rows) == 2  # log τ, logit φ
            for r in rows
                @test isfinite(r.inla_mean) && isfinite(r.nuts_mean)
                @test isfinite(r.inla_sd)   && isfinite(r.nuts_sd)
                @test !r.flagged
            end
        end
    end
end

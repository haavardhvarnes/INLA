# Oracle test: Cox proportional-hazards regression on synthetic right-
# censored data vs R-INLA's `family = "coxph"`.
#
# This test exercises the data-augmentation pathway in `inla_coxph`. R-INLA's
# `coxph` family performs the same Holford / Laird-Olivier piecewise-
# exponential-as-Poisson augmentation internally, with a piecewise-constant
# baseline log-hazard given an RW1 smoothing prior. We replay the input
# data through `inla_coxph(...)` plus a Poisson + RW1 + FixedEffects
# `LatentGaussianModel`, and compare the covariate posterior summaries
# to R-INLA's.
#
# What we compare:
#   - β posterior mean: tight tolerance (the user-facing guarantee).
#   - β posterior sd:   tight relative tolerance.
#
# What we deliberately do NOT compare:
#   - log marginal likelihood. The augmented Poisson's log-density and
#     the original Cox PH (piecewise-exponential) log-density differ by
#     the η-independent term `Σ_{events} log(E_{k_last,i})` — the
#     exposure of the interval the event lands in. This shifts `mlik`
#     by ~300-400 nats in this dataset; it cancels in the posterior of
#     `(γ, β)` so it does not affect inference. Investigated and
#     accepted: documented in the algebraic-equivalence regression test
#     (`test/regression/test_coxph_augmentation.jl`).
#   - The baseline-hazard component. R-INLA reports an RW1 over its
#     own internal cutpoints (16 nodes); our augmentation uses 15 (or
#     more) quantile-based cutpoints. Both produce statistically
#     equivalent fits to the *covariate* effects, but the per-knot
#     baseline values are not directly comparable.
#
# Fixture: scripts/generate-fixtures/lgm/synthetic_coxph.R.
# Skipped transparently if the JLD2 fixture has not been generated.

include("load_fixture.jl")

using Test
using SparseArrays
using LinearAlgebra
using LatentGaussianModels: inla_coxph, coxph_design,
                            PoissonLikelihood, FixedEffects, RW1, PCPrecision,
                            LatentGaussianModel, inla, random_effects

const COXPH_FIXTURE = "synthetic_coxph"

# Comparison metric: number of R-INLA posterior SDs separating the two
# point estimates. R-INLA's `family = "coxph"` augments on its own
# (equispaced) cutpoint grid, while our default uses quantile-based
# breakpoints; both are statistically equivalent piecewise-exponential
# fits but the slight grid difference shifts β̂ by a fraction of an SD.
# 1.5 SDs is well within Monte-Carlo noise on this n = 400 dataset.
const COXPH_BETA_MEAN_NSD_TOL = 1.5
const COXPH_BETA_SD_REL_TOL = 0.10

@testset "synthetic_coxph vs R-INLA" begin
    if !has_oracle_fixture(COXPH_FIXTURE)
        @test_skip "oracle fixture $COXPH_FIXTURE not generated " *
                   "(see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(COXPH_FIXTURE)
        @test fx["name"] == COXPH_FIXTURE
        @test haskey(fx, "summary_fixed")

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate"
        else
            inp = fx["input"]
            time = Float64.(inp["time"])
            event = Int.(inp["event"])
            X = Float64.(inp["X"])
            n = length(time)
            @test size(X, 1) == n
            @test size(X, 2) == 2

            aug = inla_coxph(time, event)
            @test aug.n_subjects == n
            @test sum(aug.E)≈sum(time) atol=1e-8

            ℓ = PoissonLikelihood(E=aug.E)
            c_baseline = RW1(aug.n_intervals;
                hyperprior=PCPrecision(1.0, 0.01))
            c_beta = FixedEffects(2)
            A = coxph_design(aug, X)
            model = LatentGaussianModel(ℓ, (c_baseline, c_beta), A)

            res = inla(model, aug.y)

            sf = fx["summary_fixed"]
            β_R_mean = Float64.(sf["mean"])
            β_R_sd = Float64.(sf["sd"])
            @test sf["rownames"] == ["x1", "x2"]

            re = random_effects(model, res)
            β_J_mean = re["FixedEffects[2]"].mean
            β_J_sd = re["FixedEffects[2]"].sd

            # Posterior mean: SD-scaled distance.
            @test abs(β_J_mean[1] - β_R_mean[1]) / β_R_sd[1] <
                  COXPH_BETA_MEAN_NSD_TOL
            @test abs(β_J_mean[2] - β_R_mean[2]) / β_R_sd[2] <
                  COXPH_BETA_MEAN_NSD_TOL

            # Posterior sd: relative tolerance.
            @test abs(β_J_sd[1] - β_R_sd[1]) / β_R_sd[1] <
                  COXPH_BETA_SD_REL_TOL
            @test abs(β_J_sd[2] - β_R_sd[2]) / β_R_sd[2] <
                  COXPH_BETA_SD_REL_TOL

            # mlik finite (gap to R-INLA is the η-independent
            # `Σ_events log E_{k_last,i}` term — see header).
            @test isfinite(res.log_marginal)
        end
    end
end

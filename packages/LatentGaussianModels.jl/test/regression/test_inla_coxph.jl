# End-to-end synthetic recovery test for `inla_coxph`. Simulates a
# Cox-PH dataset with a known piecewise-constant baseline log-hazard
# and known covariate effects, augments via `inla_coxph`, fits the
# resulting Poisson + RW1 + FixedEffects model with `inla(...)`, and
# checks that the posterior mean of the covariate coefficients lands
# within ~4 SD of truth.
#
# We do not validate the baseline log-hazard recovery here — it is a
# functional quantity and only its identifiability up to a constant
# (overlapping with the level of the linear predictor) matters for
# correctness. The covariate-coefficient recovery is the contractful
# user-facing guarantee.

using Test
using Random
using SparseArrays
using LinearAlgebra
using Statistics
using LatentGaussianModels: inla_coxph, coxph_design,
                            PoissonLikelihood, FixedEffects, RW1, PCPrecision,
                            LatentGaussianModel, inla, fixed_effects, random_effects

# Inverse-CDF sampler for the piecewise-exponential survival distribution
# with covariate-shifted hazards `λ_i(t) = exp(γ_k + xᵀβ)` on
# `t ∈ [bp[k], bp[k+1])`. Returns a finite event time within the support
# of `bp`, or `bp[end]` if the random draw exceeds the cumulative hazard
# at the end of the support (the caller treats that as an administrative
# censoring event).
function _sample_pwexp_time(rng, γ, bp, x, β)
    K = length(bp) - 1
    R = -log(rand(rng))         # standard exponential
    cum = 0.0
    shift = dot(x, β)
    for k in 1:K
        λ = exp(γ[k] + shift)
        Δ = bp[k + 1] - bp[k]
        if cum + λ * Δ ≥ R
            return bp[k] + (R - cum) / λ
        end
        cum += λ * Δ
    end
    return bp[end]
end

@testset "inla_coxph: synthetic Cox PH recovery" begin
    rng = MersenneTwister(20260430)

    # --- Truth -----------------------------------------------------------
    n = 400
    bp = collect(range(0.0, 5.0; length=11))   # 10 intervals
    K = length(bp) - 1
    γ_true = [-1.0, -0.7, -0.4, -0.1, 0.1, 0.0, -0.2, -0.5, -0.8, -1.1]
    β_true = [0.50, -0.30]

    # Covariates: standardised.
    X = randn(rng, n, 2)

    # --- Simulate event times --------------------------------------------
    time_event = [_sample_pwexp_time(rng, γ_true, bp, X[i, :], β_true)
                  for i in 1:n]

    # Random administrative censoring time: Uniform(2.5, 5.0). If the
    # simulated event time exceeds this, observe the censoring time and
    # set δ = 0; otherwise observe the event time with δ = 1.
    cens_time = 2.5 .+ 4.0 .* rand(rng, n)
    time = min.(time_event, cens_time)
    event = Int.(time_event .≤ cens_time)
    # Numerical safety: avoid t = 0 exactly.
    time .= max.(time, 1e-6)

    @test count(==(1), event) > n / 4   # ≥25% events — sanity

    # --- Augmentation ----------------------------------------------------
    aug = inla_coxph(time, event; breakpoints=bp)
    @test aug.n_intervals == K
    @test aug.n_subjects == n
    @test sum(aug.E)≈sum(time) atol=1e-8

    # --- Fit -------------------------------------------------------------
    ℓ = PoissonLikelihood(E=aug.E)
    c_baseline = RW1(aug.n_intervals; hyperprior=PCPrecision(1.0, 0.01))
    c_beta = FixedEffects(2)
    A = coxph_design(aug, X)
    model = LatentGaussianModel(ℓ, (c_baseline, c_beta), A)

    res = inla(model, aug.y)

    # --- Coefficient recovery -------------------------------------------
    # FixedEffects(2) (length > 1) is surfaced via random_effects with
    # the auto-generated component name "FixedEffects[2]" (component
    # index = 2 in the model tuple).
    re = random_effects(model, res)
    β_block = re["FixedEffects[2]"]
    β_hat = β_block.mean
    β_sd = β_block.sd
    @test length(β_hat) == 2
    @test all(β_sd .> 0)

    # 4-SD bracket on each coefficient.
    @test abs(β_hat[1] - β_true[1]) < 4 * β_sd[1]
    @test abs(β_hat[2] - β_true[2]) < 4 * β_sd[2]

    # Sign recovery (much weaker, but a useful smoke check for
    # accidental sign-flip bugs in the augmentation).
    @test sign(β_hat[1]) == sign(β_true[1])
    @test sign(β_hat[2]) == sign(β_true[2])

    # Marginal log-likelihood is finite.
    @test isfinite(res.log_marginal)
end

# PC-Matern prior: rate derivation, marginal tail probabilities, and
# the (log ρ, log σ) density used by SPDE2's `log_hyperprior`.
#
# The density on (log ρ, log σ) has two independent components:
#   π(ρ) = (d/2) λ_ρ ρ^(-d/2-1) exp(-λ_ρ ρ^(-d/2))       (d = 2)
#   π(σ) = λ_σ exp(-λ_σ σ)
# with  λ_ρ = -log(range_α) · range_U^(d/2),
#       λ_σ = -log(sigma_α) / sigma_U.
#
# The evaluator includes the log ρ + log σ Jacobian that converts to
# density on (log ρ, log σ). Tests below confirm both the rate formulas
# and the density construction against closed-form references.

using QuadGK: quadgk

@testset "PCMatern — rate derivation from user specification" begin
    pc = PCMatern(range_U=0.5, range_α=0.05,
        sigma_U=2.0, sigma_α=0.01)
    @test pc.λ_ρ≈-log(0.05) * 0.5 rtol=1.0e-12
    @test pc.λ_σ≈-log(0.01) / 2.0 rtol=1.0e-12
    @test pc.range_U == 0.5
    @test pc.range_α == 0.05
    @test pc.sigma_U == 2.0
    @test pc.sigma_α == 0.01
end

@testset "PCMatern — argument validation" begin
    @test_throws ArgumentError PCMatern(range_U=-1.0)
    @test_throws ArgumentError PCMatern(sigma_U=0.0)
    @test_throws ArgumentError PCMatern(range_α=0.0)
    @test_throws ArgumentError PCMatern(range_α=1.0)
    @test_throws ArgumentError PCMatern(sigma_α=-0.1)
end

@testset "PCMatern — tail probabilities match specification" begin
    # P(ρ < range_U) = range_α and P(σ > sigma_U) = sigma_α are the
    # defining relations. Verify numerically by integrating π(ρ) over
    # [0, U] and π(σ) over [U, ∞).
    ru, rα = 1.5, 0.05
    su, sα = 0.8, 0.01
    pc = PCMatern(range_U=ru, range_α=rα, sigma_U=su, sigma_α=sα)

    d = 2
    π_ρ(ρ) = (d / 2) * pc.λ_ρ * ρ^(-d / 2 - 1) * exp(-pc.λ_ρ * ρ^(-d / 2))
    π_σ(σ) = pc.λ_σ * exp(-pc.λ_σ * σ)

    # Both marginals integrate to 1. The range density has a heavy
    # tail π(ρ) ~ λ_ρ · ρ^(-2) for large ρ (d = 2), so the upper
    # integration bound has to run to Inf to capture the full mass.
    tot_ρ, _ = quadgk(π_ρ, 1.0e-8, Inf; rtol=1.0e-10)
    tot_σ, _ = quadgk(π_σ, 0.0, Inf; rtol=1.0e-10)
    @test tot_ρ≈1 rtol=1.0e-6
    @test tot_σ≈1 rtol=1.0e-6

    # Specified tail probabilities.
    left_tail_ρ, _ = quadgk(π_ρ, 1.0e-10, ru; rtol=1.0e-10)
    right_tail_σ, _ = quadgk(π_σ, su, Inf; rtol=1.0e-10)
    @test left_tail_ρ≈rα rtol=1.0e-6
    @test right_tail_σ≈sα rtol=1.0e-6
end

@testset "pc_matern_log_density — matches closed-form for d = 2" begin
    pc = PCMatern(range_U=1.0, range_α=0.05,
        sigma_U=1.0, sigma_α=0.01)

    for log_ρ in (-1.0, 0.0, 0.7), log_σ in (-2.0, 0.0, 0.3)
        ρ = exp(log_ρ)
        σ = exp(log_σ)
        # Direct closed form: log π_{ρ,σ}(ρ, σ) + log ρ + log σ
        #   = log π_ρ(ρ) + log π_σ(σ) + log ρ + log σ
        log_π_ρ = log(pc.λ_ρ) - 2 * log_ρ - pc.λ_ρ / ρ         # d = 2
        log_π_σ = log(pc.λ_σ) - pc.λ_σ * σ
        expected = log_π_ρ + log_π_σ + log_ρ + log_σ
        @test pc_matern_log_density(pc, log_ρ, log_σ)≈expected rtol=1.0e-12
    end
end

@testset "pc_matern_log_density — factorises, each marginal integrates to 1" begin
    # The density on (log ρ, log σ) factorises into independent range
    # and sigma pieces. Each one-dimensional marginal must integrate to
    # 1. This is the change-of-variables sanity check.
    pc = PCMatern(range_U=1.0, range_α=0.05,
        sigma_U=1.0, sigma_α=0.01)

    # Holding log_σ fixed at 0.0, the log_ρ marginal equals
    # `pc_matern_log_density(pc, log_ρ, 0.0) - pc_matern_log_density(pc, 0.0, 0.0)
    #  + (log π_σ(1) + 0)` — simpler: exp(f(x, 0) - f(0, 0)) gives unnormalised
    # marginal up to a constant; integrate the full log-density in log ρ at
    # fixed log σ and divide by the constant for that σ.
    log_σ0 = 0.0
    const_σ = log(pc.λ_σ) - pc.λ_σ + log_σ0     # == lp_sigma at log σ = 0
    integral_ρ, _ = quadgk(
        log_ρ -> exp(pc_matern_log_density(pc, log_ρ, log_σ0) - const_σ),
        -30.0, 30.0; rtol=1.0e-9
    )
    @test integral_ρ≈1 rtol=1.0e-5

    # Same for the σ marginal at fixed log_ρ.
    log_ρ0 = 0.0
    const_ρ = log(1.0) + log(pc.λ_ρ) - 0.0 - pc.λ_ρ    # lp_range at log ρ = 0 with d = 2
    integral_σ, _ = quadgk(
        log_σ -> exp(pc_matern_log_density(pc, log_ρ0, log_σ) - const_ρ),
        -30.0, 30.0; rtol=1.0e-9
    )
    @test integral_σ≈1 rtol=1.0e-5
end

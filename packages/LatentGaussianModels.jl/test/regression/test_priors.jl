using LatentGaussianModels: PCPrecision, GammaPrecision, LogNormalPrecision,
                            WeakPrior, PCAlphaW, log_prior_density, user_scale, prior_name

@testset "PCPrecision" begin
    p = PCPrecision(1.0, 0.01)
    @test prior_name(p) == :pc_prec
    @test user_scale(p, 0.0) ≈ 1.0
    @test isfinite(log_prior_density(p, 0.0))

    # Check integral of exp(log_prior_density) ≈ 1 via trapezoid
    θ_grid = range(-15, 15; length=5000)
    dθ = step(θ_grid)
    vals = [exp(log_prior_density(p, θ)) for θ in θ_grid]
    total = sum(vals) * dθ
    @test isapprox(total, 1.0; rtol=5.0e-3)

    # Constructor validation
    @test_throws ArgumentError PCPrecision(-1.0, 0.01)
    @test_throws ArgumentError PCPrecision(1.0, 1.5)
end

@testset "GammaPrecision" begin
    p = GammaPrecision(1.0, 5.0e-5)
    @test prior_name(p) == :gamma_prec
    @test user_scale(p, 0.0) ≈ 1.0
    @test isfinite(log_prior_density(p, 0.0))
    @test_throws ArgumentError GammaPrecision(-1.0, 1.0)
    @test_throws ArgumentError GammaPrecision(1.0, -1.0)
end

@testset "LogNormalPrecision" begin
    p = LogNormalPrecision(0.0, 1.0)
    # Prior on θ = log τ is N(μ, σ²); integrate
    θ_grid = range(-20, 20; length=5000)
    dθ = step(θ_grid)
    total = sum(exp(log_prior_density(p, θ)) for θ in θ_grid) * dθ
    @test isapprox(total, 1.0; rtol=5.0e-3)
    @test_throws ArgumentError LogNormalPrecision(0.0, -1.0)
end

@testset "WeakPrior" begin
    p = WeakPrior()
    @test prior_name(p) == :weak
    @test log_prior_density(p, 1.23) == 0.0
end

@testset "PCAlphaW" begin
    p = PCAlphaW(5.0)
    @test prior_name(p) == :pc_alphaw
    @test user_scale(p, 0.0) ≈ 1.0
    @test user_scale(p, log(2.0)) ≈ 2.0

    # Constructor validation
    @test_throws ArgumentError PCAlphaW(0.0)
    @test_throws ArgumentError PCAlphaW(-1.0)

    # Reference values from R-INLA's `inla.pc.dalphaw(α, lambda = 5)`.
    # R-INLA returns the density on α-scale; we evaluate Julia on θ-scale
    # and convert via `log π_α(α) = log π_θ(log α) - log α`. R-INLA's
    # internal spline-derivative carries ~1e-7 numerical error; our
    # closed-form analytical derivative is exact to floating-point.
    for (α, ld_α_R) in (
        (0.5, -3.9771693750),
        (1.0, 1.2167190471),
        (2.0, -3.6822490132),
        (3.0, -5.9533703540),
        (5.0, -8.4361117109)
    )
        θ = log(α)
        ld_α_J = log_prior_density(p, θ) - log(α)
        @test isapprox(ld_α_J, ld_α_R; atol=1.0e-6)
    end

    # Continuity at α = 1 (Taylor-limit branch ↔ closed-form branch).
    # `log_prior_density` should be smooth across the branch boundary.
    ld_just_below = log_prior_density(p, -1.0e-4)
    ld_at_one = log_prior_density(p, 0.0)
    ld_just_above = log_prior_density(p, +1.0e-4)
    @test isapprox(ld_at_one, ld_just_below; atol=1.0e-3)
    @test isapprox(ld_at_one, ld_just_above; atol=1.0e-3)
    @test isfinite(ld_at_one)

    # The Taylor-limit log-density at θ = 0:
    #   log π_θ(0) = log(λ/2) + log √K''(0) = log(λ/2) + ½ log K''(0).
    # K''(0) = (1 - γ_E)² + π²/6.
    γ = MathConstants.eulergamma
    K2_at_0 = (1 - γ)^2 + π^2 / 6
    expected_ld_at_one = log(p.λ / 2) + 0.5 * log(K2_at_0)
    @test isapprox(ld_at_one, expected_ld_at_one; atol=1.0e-12)

    # Integrates to 1 over θ ∈ ℝ via trapezoid — proper density on the
    # internal scale. Use an asymmetric grid because the prior is only
    # symmetric in d, not in θ (the |dα/dθ| Jacobian breaks symmetry).
    θ_grid = range(-8, 8; length=8001)
    dθ = step(θ_grid)
    total = sum(exp(log_prior_density(p, θ)) for θ in θ_grid) * dθ
    @test isapprox(total, 1.0; rtol=5.0e-3)

    # Symmetry of d(α): d(α) ≡ d(?) does NOT hold — d is a function of
    # α only, with the reference at α = 1. But the *distance* d(α) > 0
    # for any α ≠ 1, growing with |log α| asymptotically. Sanity check:
    @test log_prior_density(p, log(0.5)) < log_prior_density(p, 0.0)
    @test log_prior_density(p, log(2.0)) < log_prior_density(p, 0.0)
    @test log_prior_density(p, log(10.0)) < log_prior_density(p, log(2.0))

    # Different λ scales the tail correctly: a larger λ shrinks more
    # mass to the reference, so density at α far from 1 drops.
    p_loose = PCAlphaW(1.0)
    p_tight = PCAlphaW(20.0)
    @test log_prior_density(p_tight, log(3.0)) <
          log_prior_density(p_loose, log(3.0))
end

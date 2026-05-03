using LatentGaussianModels: PCPrecision, GammaPrecision, LogNormalPrecision,
                            WeakPrior, PCAlphaW, PCCor0, PCCor1, log_prior_density, user_scale, prior_name

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

@testset "PCCor0" begin
    p = PCCor0(0.5, 0.5)   # P(|ρ| > 0.5) = 0.5
    @test prior_name(p) == :pc_cor0
    @test user_scale(p, 0.0) ≈ 0.0
    @test user_scale(p, atanh(0.7)) ≈ 0.7
    @test isfinite(log_prior_density(p, 0.0))

    # λ closed form: d(0.5) = √(-log(1 - 0.25)) = √(log(4/3))
    d_U = sqrt(-log1p(-0.5^2))
    @test p.λ ≈ -log(0.5) / d_U

    # Symmetry in θ: π_θ(θ) = π_θ(-θ) since ρ ↔ -ρ symmetry holds.
    for θ in (0.1, 0.5, 1.2, 2.0)
        @test log_prior_density(p, θ) ≈ log_prior_density(p, -θ)
    end

    # Density at θ = 0 equals log(λ/2).
    @test log_prior_density(p, 0.0) ≈ log(p.λ / 2)

    # Continuity across the Taylor↔formula branch boundary. Both
    # branches share the exact `d = √(-log1p(-ρ²))`; only the
    # `log(d²/ρ²)` term differs (Taylor truncation `ρ²/2` vs full
    # `log(d²/ρ²)`). At the threshold, the discrepancy is bounded by
    # the leading `ρ⁴/12` Taylor remainder.
    ρ² = 1.0e-7
    nlog = -log1p(-ρ²)
    common = log(p.λ) - log(2) - p.λ * sqrt(nlog)
    formula_val = common - 0.5 * log(nlog / ρ²)
    taylor_val = common - 0.5 * (ρ² / 2)
    @test isapprox(formula_val, taylor_val; atol=ρ²^2 / 6)

    # Larger λ ⇒ tighter shrinkage to ρ = 0.
    p_loose = PCCor0(0.9, 0.5)   # small λ
    p_tight = PCCor0(0.1, 0.5)   # large λ
    @test p_tight.λ > p_loose.λ
    @test log_prior_density(p_tight, atanh(0.7)) <
          log_prior_density(p_loose, atanh(0.7))

    # Integrates to 1 over θ ∈ ℝ. Use a wide grid because the prior is
    # heavy-tailed in θ (light-tailed in ρ but the atanh map stretches).
    θ_grid = range(-15, 15; length=10001)
    dθ = step(θ_grid)
    total = sum(exp(log_prior_density(p, θ)) for θ in θ_grid) * dθ
    @test isapprox(total, 1.0; rtol=5.0e-3)

    # Constructor validation
    @test_throws ArgumentError PCCor0(0.0, 0.5)
    @test_throws ArgumentError PCCor0(1.0, 0.5)
    @test_throws ArgumentError PCCor0(-0.1, 0.5)
    @test_throws ArgumentError PCCor0(0.5, 0.0)
    @test_throws ArgumentError PCCor0(0.5, 1.0)
end

@testset "PCCor1" begin
    p = PCCor1(0.7, 0.7)   # P(ρ > 0.7) = 0.7, R-INLA textbook default
    @test prior_name(p) == :pc_cor1
    @test user_scale(p, 0.0) ≈ 0.0
    @test user_scale(p, atanh(0.5)) ≈ 0.5
    @test isfinite(log_prior_density(p, 0.0))

    # λ root-finder accuracy: F(λ) = (1 - exp(-aλ))/(1 - exp(-bλ)) must
    # match α at the solved λ to floating-point precision.
    a = sqrt(1 - 0.7)
    b = sqrt(2)
    F_at_λ = -expm1(-a * p.λ) / -expm1(-b * p.λ)
    @test isapprox(F_at_λ, 0.7; atol=1.0e-10)

    # Closed-form check on the ρ-scale density `inla.pc.dcor1` at a few
    # ρ values. The Julia prior is on `θ = atanh(ρ)`, so convert via
    # `log π_ρ(ρ) = log π_θ(θ) - log|dρ/dθ| = log π_θ(θ) - log(1-ρ²)`.
    # The reference is the R-INLA published formula
    #   log π_ρ(ρ) = log λ - λ√(1-ρ) - log(1-exp(-√2 λ)) - log(2√(1-ρ))
    # evaluated at the same `λ` Julia solved. This pins the
    # ρ ↔ θ Jacobian conversion (a primary place to introduce bugs).
    log_norm_R = log(-expm1(-sqrt(2) * p.λ))
    for ρ in (-0.5, 0.0, 0.3, 0.7, 0.9)
        θ = atanh(ρ)
        sqrt_1m = sqrt(1 - ρ)
        ld_ρ_R = log(p.λ) - p.λ * sqrt_1m - log_norm_R - log(2 * sqrt_1m)
        ld_ρ_J = log_prior_density(p, θ) - log1p(-ρ^2)
        @test isapprox(ld_ρ_J, ld_ρ_R; atol=1.0e-12)
    end

    # Larger λ ⇒ tighter shrinkage to the reference ρ = 1. Holding U
    # fixed, raising α toward 1 raises λ and concentrates more mass
    # near ρ = 1 — so density at any ρ < 1 strictly drops as α grows.
    p_loose = PCCor1(0.7, 0.55)   # α near α_min = √0.15 ≈ 0.387, small λ
    p_tight = PCCor1(0.7, 0.95)   # α near 1, large λ
    @test p_tight.λ > p_loose.λ
    @test log_prior_density(p_tight, atanh(0.0)) <
          log_prior_density(p_loose, atanh(0.0))

    # Saturation stability: |θ| up to 25 stays finite. Symmetric in θ
    # the prior is *not* — penalises ρ → -1 (θ → -∞) more strongly
    # than ρ → +1 (θ → +∞), since +1 is the reference.
    for θ in (-25.0, -5.0, 0.0, 5.0, 25.0)
        @test isfinite(log_prior_density(p, θ))
    end
    @test log_prior_density(p, 5.0) > log_prior_density(p, -5.0)

    # Integrates to 1 over θ ∈ ℝ via trapezoid. The prior is heavy in
    # the +θ tail (concentrated near ρ = 1) so the grid must extend
    # further on the +θ side.
    θ_grid = range(-20, 25; length=20001)
    dθ = step(θ_grid)
    total = sum(exp(log_prior_density(p, θ)) for θ in θ_grid) * dθ
    @test isapprox(total, 1.0; rtol=5.0e-3)

    # Constructor validation: U bounds, α bounds.
    @test_throws ArgumentError PCCor1(-1.0, 0.7)
    @test_throws ArgumentError PCCor1(1.0, 0.7)
    α_min = sqrt((1 - 0.7) / 2)
    @test_throws ArgumentError PCCor1(0.7, α_min)        # α at lower bound
    @test_throws ArgumentError PCCor1(0.7, α_min - 0.01) # α below lower bound
    @test_throws ArgumentError PCCor1(0.7, 1.0)          # α at upper bound

    # Default kwargs: PCCor1() ≡ PCCor1(0.7, 0.7).
    @test PCCor1().λ ≈ PCCor1(0.7, 0.7).λ
end

using LatentGaussianModels: GammaSurvLikelihood, LogLink, IdentityLink,
                            Censoring, NONE, RIGHT, LEFT, INTERVAL,
                            log_density, ∇_η_log_density, ∇²_η_log_density,
                            ∇³_η_log_density,
                            pointwise_log_density, pointwise_cdf,
                            nhyperparameters, initial_hyperparameters, link, log_hyperprior,
                            GammaPrecision

using Distributions: Distributions
using SpecialFunctions: SpecialFunctions

# Reuses the same FD helper used in test_likelihoods.jl.
function fd_grad_gs(f, η, h=1.0e-6)
    g = similar(η)
    for i in eachindex(η)
        ep = copy(η)
        ep[i] += h
        em = copy(η)
        em[i] -= h
        g[i] = (f(ep) - f(em)) / (2h)
    end
    return g
end

const GS_LOG_PHIS = (-0.7, 0.0, 0.7)   # φ ≈ 0.50, 1.00, 2.01

@testset "GammaSurvLikelihood — defaults / contract" begin
    ℓ = GammaSurvLikelihood()
    @test nhyperparameters(ℓ) == 1
    @test initial_hyperparameters(ℓ) == [0.0]
    @test link(ℓ) isa LogLink
    @test ℓ.censoring === nothing
    @test ℓ.time_hi === nothing
    @test ℓ.hyperprior isa GammaPrecision

    # log-hyperprior delegates to the GammaPrecision prior on log φ
    @test log_hyperprior(ℓ, [0.0]) isa Real
    @test log_hyperprior(ℓ, [-0.5]) isa Real

    # Non-LogLink rejected
    @test_throws ArgumentError GammaSurvLikelihood(link=IdentityLink())
end

@testset "GammaSurvLikelihood — fast path (all uncensored)" begin
    rng = Random.Xoshiro(301)
    y = abs.(randn(rng, 10)) .+ 0.1
    η = randn(rng, 10) .* 0.5

    ℓ = GammaSurvLikelihood()

    for log_φ in GS_LOG_PHIS
        θ = [log_φ]
        φ = exp(log_φ)

        lp = log_density(ℓ, y, η, θ)
        @test isfinite(lp)
        # Closed form: φ log φ - φ η - log Γ(φ) + (φ-1) log y - φ y exp(-η)
        lgamma_φ = SpecialFunctions.loggamma(φ)
        expected = sum(φ * log_φ - φ * η[i] - lgamma_φ +
                       (φ - 1) * log(y[i]) - φ * y[i] * exp(-η[i])
        for i in eachindex(y))
        @test lp ≈ expected
        @test sum(pointwise_log_density(ℓ, y, η, θ)) ≈ lp

        g = ∇_η_log_density(ℓ, y, η, θ)
        g_fd = fd_grad_gs(h -> log_density(ℓ, y, h, θ), η)
        @test g≈g_fd atol=1.0e-4
        # Closed form: φ (y/μ - 1)
        @test g ≈ @. φ * (y * exp(-η) - 1)

        H = ∇²_η_log_density(ℓ, y, η, θ)
        H_fd = fd_grad_gs(h -> sum(∇_η_log_density(ℓ, y, h, θ)), η)
        @test H≈H_fd atol=1.0e-4
        @test H ≈ @. -φ * y * exp(-η)

        T = ∇³_η_log_density(ℓ, y, η, θ)
        T_fd = fd_grad_gs(h -> sum(∇²_η_log_density(ℓ, y, h, θ)), η)
        @test T≈T_fd atol=1.0e-4
        @test T ≈ @. φ * y * exp(-η)
    end
end

@testset "GammaSurvLikelihood — mixed censoring" begin
    rng = Random.Xoshiro(302)
    y_lo = abs.(randn(rng, 8)) .+ 0.1
    η = randn(rng, 8) .* 0.4
    censoring = [NONE, RIGHT, LEFT, INTERVAL, NONE, RIGHT, LEFT, INTERVAL]
    time_hi = y_lo .+ abs.(randn(rng, 8)) .+ 0.5

    ℓ = GammaSurvLikelihood(censoring=censoring, time_hi=time_hi)

    for log_φ in GS_LOG_PHIS
        θ = [log_φ]

        lp = log_density(ℓ, y_lo, η, θ)
        @test isfinite(lp)
        @test sum(pointwise_log_density(ℓ, y_lo, η, θ)) ≈ lp

        g = ∇_η_log_density(ℓ, y_lo, η, θ)
        g_fd = fd_grad_gs(h -> log_density(ℓ, y_lo, h, θ), η)
        @test g≈g_fd atol=1.0e-4

        H = ∇²_η_log_density(ℓ, y_lo, η, θ)
        H_fd = fd_grad_gs(h -> sum(∇_η_log_density(ℓ, y_lo, h, θ)), η)
        @test H≈H_fd atol=1.0e-4

        T = ∇³_η_log_density(ℓ, y_lo, η, θ)
        T_fd = fd_grad_gs(h -> sum(∇²_η_log_density(ℓ, y_lo, h, θ)), η)
        @test T≈T_fd atol=1.0e-4
    end
end

@testset "GammaSurvLikelihood — fast vs mixed agree on all-NONE" begin
    rng = Random.Xoshiro(303)
    y = abs.(randn(rng, 6)) .+ 0.1
    η = randn(rng, 6) .* 0.4

    ℓ_fast = GammaSurvLikelihood()
    ℓ_mixed = GammaSurvLikelihood(censoring=fill(NONE, 6))

    for log_φ in GS_LOG_PHIS
        θ = [log_φ]
        @test log_density(ℓ_fast, y, η, θ) ≈ log_density(ℓ_mixed, y, η, θ)
        @test ∇_η_log_density(ℓ_fast, y, η, θ) ≈
              ∇_η_log_density(ℓ_mixed, y, η, θ)
        @test ∇²_η_log_density(ℓ_fast, y, η, θ) ≈
              ∇²_η_log_density(ℓ_mixed, y, η, θ)
        @test ∇³_η_log_density(ℓ_fast, y, η, θ) ≈
              ∇³_η_log_density(ℓ_mixed, y, η, θ)
    end
end

@testset "GammaSurvLikelihood — Symbol coercion" begin
    censoring_sym = [:none, :right, :left, :interval]
    time_hi = [0.0, 0.0, 0.0, 5.0]
    ℓ = GammaSurvLikelihood(censoring=censoring_sym, time_hi=time_hi)
    @test ℓ.censoring isa Vector{Censoring}
    @test ℓ.censoring == [NONE, RIGHT, LEFT, INTERVAL]
end

@testset "GammaSurvLikelihood — bad censoring type rejected" begin
    @test_throws ArgumentError GammaSurvLikelihood(censoring=[1, 2, 3])
    @test_throws ArgumentError GammaSurvLikelihood(censoring=["none", "right"])
end

@testset "GammaSurvLikelihood — RIGHT-only closed form" begin
    # log S(t) = log Q(φ, φ t exp(-η))
    rng = Random.Xoshiro(304)
    y = abs.(randn(rng, 5)) .+ 0.1
    η = randn(rng, 5) .* 0.5
    log_φ = 0.4
    φ = exp(log_φ)
    θ = [log_φ]
    ℓ = GammaSurvLikelihood(censoring=fill(RIGHT, 5))

    expected = 0.0
    for i in eachindex(y)
        x = φ * y[i] * exp(-η[i])
        _, Q = SpecialFunctions.gamma_inc(φ, x)
        expected += log(Q)
    end
    @test log_density(ℓ, y, η, θ) ≈ expected
end

@testset "GammaSurvLikelihood — LEFT-only closed form" begin
    rng = Random.Xoshiro(305)
    y = abs.(randn(rng, 5)) .+ 0.1
    η = randn(rng, 5) .* 0.5
    log_φ = 0.4
    φ = exp(log_φ)
    θ = [log_φ]
    ℓ = GammaSurvLikelihood(censoring=fill(LEFT, 5))

    expected = 0.0
    for i in eachindex(y)
        x = φ * y[i] * exp(-η[i])
        P, _ = SpecialFunctions.gamma_inc(φ, x)
        expected += log(P)
    end
    @test log_density(ℓ, y, η, θ) ≈ expected
end

@testset "GammaSurvLikelihood — INTERVAL closed form" begin
    rng = Random.Xoshiro(306)
    y_lo = abs.(randn(rng, 4)) .+ 0.1
    time_hi = y_lo .+ 0.5 .+ abs.(randn(rng, 4))
    η = randn(rng, 4) .* 0.3
    log_φ = 0.3
    φ = exp(log_φ)
    θ = [log_φ]
    ℓ = GammaSurvLikelihood(
        censoring=fill(INTERVAL, 4), time_hi=time_hi)

    expected = 0.0
    for i in eachindex(y_lo)
        x_lo = φ * y_lo[i] * exp(-η[i])
        x_hi = φ * time_hi[i] * exp(-η[i])
        P_lo, Q_lo = SpecialFunctions.gamma_inc(φ, x_lo)
        P_hi, Q_hi = SpecialFunctions.gamma_inc(φ, x_hi)
        D = Q_lo > 0.5 ? P_hi - P_lo : Q_lo - Q_hi
        expected += log(D)
    end
    @test log_density(ℓ, y_lo, η, θ) ≈ expected
end

@testset "GammaSurvLikelihood — pointwise_cdf" begin
    rng = Random.Xoshiro(307)
    y = abs.(randn(rng, 4)) .+ 0.1
    η = randn(rng, 4) .* 0.5
    log_φ = 0.2
    φ = exp(log_φ)
    θ = [log_φ]

    # Fast path: F(t) = P(φ, φ t exp(-η))
    ℓ_fast = GammaSurvLikelihood()
    cdf_fast = pointwise_cdf(ℓ_fast, y, η, θ)
    expected = [SpecialFunctions.gamma_inc(φ, φ * y[i] * exp(-η[i]))[1]
                for i in eachindex(y)]
    @test cdf_fast ≈ expected
    @test all(0 .≤ cdf_fast .≤ 1)

    # All-NONE censored variant: same numbers
    ℓ_none = GammaSurvLikelihood(censoring=fill(NONE, 4))
    @test pointwise_cdf(ℓ_none, y, η, θ) ≈ cdf_fast

    # Censored variant with non-NONE rows: undefined, throws
    ℓ_cens = GammaSurvLikelihood(censoring=[NONE, RIGHT, NONE, NONE])
    @test_throws ArgumentError pointwise_cdf(ℓ_cens, y, η, θ)
end

@testset "GammaSurvLikelihood — pointwise_log_density per-mode" begin
    rng = Random.Xoshiro(308)
    y_lo = abs.(randn(rng, 4)) .+ 0.1
    time_hi = y_lo .+ 0.5
    η = randn(rng, 4) .* 0.4
    log_φ = -0.3
    φ = exp(log_φ)
    θ = [log_φ]
    ℓ = GammaSurvLikelihood(
        censoring=[NONE, RIGHT, LEFT, INTERVAL], time_hi=time_hi)

    pp = pointwise_log_density(ℓ, y_lo, η, θ)
    lgamma_φ = SpecialFunctions.loggamma(φ)

    # NONE: log gamma density
    x1 = φ * y_lo[1] * exp(-η[1])
    @test pp[1] ≈ φ * log_φ - φ * η[1] - lgamma_φ +
                  (φ - 1) * log(y_lo[1]) - x1

    # RIGHT: log Q(φ, x)
    x2 = φ * y_lo[2] * exp(-η[2])
    _, Q2 = SpecialFunctions.gamma_inc(φ, x2)
    @test pp[2] ≈ log(Q2)

    # LEFT: log P(φ, x)
    x3 = φ * y_lo[3] * exp(-η[3])
    P3, _ = SpecialFunctions.gamma_inc(φ, x3)
    @test pp[3] ≈ log(P3)

    # INTERVAL
    x4_lo = φ * y_lo[4] * exp(-η[4])
    x4_hi = φ * time_hi[4] * exp(-η[4])
    P_lo, Q_lo = SpecialFunctions.gamma_inc(φ, x4_lo)
    P_hi, Q_hi = SpecialFunctions.gamma_inc(φ, x4_hi)
    D = Q_lo > 0.5 ? P_hi - P_lo : Q_lo - Q_hi
    @test pp[4] ≈ log(D)
end

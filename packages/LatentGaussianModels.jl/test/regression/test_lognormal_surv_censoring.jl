using LatentGaussianModels: LognormalSurvLikelihood, IdentityLink,
    Censoring, NONE, RIGHT, LEFT, INTERVAL,
    log_density, ∇_η_log_density, ∇²_η_log_density, ∇³_η_log_density,
    pointwise_log_density, pointwise_cdf,
    nhyperparameters, initial_hyperparameters, link, log_hyperprior,
    PCPrecision

using Distributions: Distributions

# Reuses the same FD helper used in test_likelihoods.jl.
function fd_grad_lns(f, η, h = 1.0e-6)
    g = similar(η)
    for i in eachindex(η)
        ep = copy(η); ep[i] += h
        em = copy(η); em[i] -= h
        g[i] = (f(ep) - f(em)) / (2h)
    end
    return g
end

const LNS_LOG_TAUS = (-0.7, 0.0, 0.7)   # σ ≈ 1.42, 1.00, 0.71

@testset "LognormalSurvLikelihood — defaults / contract" begin
    ℓ = LognormalSurvLikelihood()
    @test nhyperparameters(ℓ) == 1
    @test initial_hyperparameters(ℓ) == [0.0]
    @test link(ℓ) isa IdentityLink
    @test ℓ.censoring === nothing
    @test ℓ.time_hi === nothing
    @test ℓ.hyperprior isa PCPrecision

    # log-hyperprior delegates to the PCPrecision prior on log τ
    @test log_hyperprior(ℓ, [0.0]) isa Real
    @test log_hyperprior(ℓ, [-0.5]) isa Real

    # Non-IdentityLink rejected
    @test_throws ArgumentError LognormalSurvLikelihood(
        link = LatentGaussianModels.LogLink())
end

@testset "LognormalSurvLikelihood — fast path (all uncensored)" begin
    rng = Random.Xoshiro(201)
    y = abs.(randn(rng, 10)) .+ 0.1
    η = randn(rng, 10) .* 0.5

    ℓ = LognormalSurvLikelihood()

    for log_τ in LNS_LOG_TAUS
        θ = [log_τ]
        τ = exp(log_τ)

        lp = log_density(ℓ, y, η, θ)
        @test isfinite(lp)
        # Closed form: -log y - 0.5 log(2π) + 0.5 log τ - 0.5 τ (log y - η)²
        log2π = log(2π)
        expected = sum(-log(y[i]) - 0.5 * log2π + 0.5 * log_τ -
                       0.5 * τ * (log(y[i]) - η[i])^2 for i in eachindex(y))
        @test lp ≈ expected
        @test sum(pointwise_log_density(ℓ, y, η, θ)) ≈ lp

        g = ∇_η_log_density(ℓ, y, η, θ)
        g_fd = fd_grad_lns(h -> log_density(ℓ, y, h, θ), η)
        @test g ≈ g_fd atol = 1.0e-4
        @test g ≈ @. τ * (log(y) - η)

        H = ∇²_η_log_density(ℓ, y, η, θ)
        H_fd = fd_grad_lns(h -> sum(∇_η_log_density(ℓ, y, h, θ)), η)
        @test H ≈ H_fd atol = 1.0e-4
        @test H ≈ fill(-τ, length(y))

        T = ∇³_η_log_density(ℓ, y, η, θ)
        T_fd = fd_grad_lns(h -> sum(∇²_η_log_density(ℓ, y, h, θ)), η)
        @test T ≈ T_fd atol = 1.0e-4
        @test T == zeros(length(y))
    end
end

@testset "LognormalSurvLikelihood — mixed censoring" begin
    rng = Random.Xoshiro(202)
    y_lo = abs.(randn(rng, 8)) .+ 0.1
    η = randn(rng, 8) .* 0.4
    censoring = [NONE, RIGHT, LEFT, INTERVAL, NONE, RIGHT, LEFT, INTERVAL]
    time_hi = y_lo .+ abs.(randn(rng, 8)) .+ 0.5

    ℓ = LognormalSurvLikelihood(censoring = censoring, time_hi = time_hi)

    for log_τ in LNS_LOG_TAUS
        θ = [log_τ]

        lp = log_density(ℓ, y_lo, η, θ)
        @test isfinite(lp)
        @test sum(pointwise_log_density(ℓ, y_lo, η, θ)) ≈ lp

        g = ∇_η_log_density(ℓ, y_lo, η, θ)
        g_fd = fd_grad_lns(h -> log_density(ℓ, y_lo, h, θ), η)
        @test g ≈ g_fd atol = 1.0e-4

        H = ∇²_η_log_density(ℓ, y_lo, η, θ)
        H_fd = fd_grad_lns(h -> sum(∇_η_log_density(ℓ, y_lo, h, θ)), η)
        @test H ≈ H_fd atol = 1.0e-4

        T = ∇³_η_log_density(ℓ, y_lo, η, θ)
        T_fd = fd_grad_lns(h -> sum(∇²_η_log_density(ℓ, y_lo, h, θ)), η)
        @test T ≈ T_fd atol = 1.0e-4
    end
end

@testset "LognormalSurvLikelihood — fast vs mixed agree on all-NONE" begin
    rng = Random.Xoshiro(203)
    y = abs.(randn(rng, 6)) .+ 0.1
    η = randn(rng, 6) .* 0.4

    ℓ_fast = LognormalSurvLikelihood()
    ℓ_mixed = LognormalSurvLikelihood(censoring = fill(NONE, 6))

    for log_τ in LNS_LOG_TAUS
        θ = [log_τ]
        @test log_density(ℓ_fast, y, η, θ) ≈ log_density(ℓ_mixed, y, η, θ)
        @test ∇_η_log_density(ℓ_fast, y, η, θ) ≈
              ∇_η_log_density(ℓ_mixed, y, η, θ)
        @test ∇²_η_log_density(ℓ_fast, y, η, θ) ≈
              ∇²_η_log_density(ℓ_mixed, y, η, θ)
        @test ∇³_η_log_density(ℓ_fast, y, η, θ) ≈
              ∇³_η_log_density(ℓ_mixed, y, η, θ)
    end
end

@testset "LognormalSurvLikelihood — Symbol coercion" begin
    censoring_sym = [:none, :right, :left, :interval]
    time_hi = [0.0, 0.0, 0.0, 5.0]
    ℓ = LognormalSurvLikelihood(censoring = censoring_sym, time_hi = time_hi)
    @test ℓ.censoring isa Vector{Censoring}
    @test ℓ.censoring == [NONE, RIGHT, LEFT, INTERVAL]
end

@testset "LognormalSurvLikelihood — bad censoring type rejected" begin
    @test_throws ArgumentError LognormalSurvLikelihood(censoring = [1, 2, 3])
    @test_throws ArgumentError LognormalSurvLikelihood(
        censoring = ["none", "right"])
end

@testset "LognormalSurvLikelihood — RIGHT-only closed form" begin
    # log S(t) = log Φ((η − log t) σ⁻¹) = logccdf(N, (log t − η)/σ)
    rng = Random.Xoshiro(204)
    y = abs.(randn(rng, 5)) .+ 0.1
    η = randn(rng, 5) .* 0.5
    log_τ = 0.4
    τ = exp(log_τ)
    σ = sqrt(1 / τ)
    θ = [log_τ]
    N = Distributions.Normal()
    ℓ = LognormalSurvLikelihood(censoring = fill(RIGHT, 5))

    expected = sum(Distributions.logccdf(N, (log(y[i]) - η[i]) / σ)
                   for i in eachindex(y))
    @test log_density(ℓ, y, η, θ) ≈ expected
end

@testset "LognormalSurvLikelihood — LEFT-only closed form" begin
    rng = Random.Xoshiro(205)
    y = abs.(randn(rng, 5)) .+ 0.1
    η = randn(rng, 5) .* 0.5
    log_τ = 0.4
    τ = exp(log_τ)
    σ = sqrt(1 / τ)
    θ = [log_τ]
    N = Distributions.Normal()
    ℓ = LognormalSurvLikelihood(censoring = fill(LEFT, 5))

    expected = sum(Distributions.logcdf(N, (log(y[i]) - η[i]) / σ)
                   for i in eachindex(y))
    @test log_density(ℓ, y, η, θ) ≈ expected
end

@testset "LognormalSurvLikelihood — INTERVAL closed form" begin
    rng = Random.Xoshiro(206)
    y_lo = abs.(randn(rng, 4)) .+ 0.1
    time_hi = y_lo .+ 0.5 .+ abs.(randn(rng, 4))
    η = randn(rng, 4) .* 0.3
    log_τ = 0.3
    τ = exp(log_τ)
    σ = sqrt(1 / τ)
    θ = [log_τ]
    N = Distributions.Normal()
    ℓ = LognormalSurvLikelihood(
        censoring = fill(INTERVAL, 4), time_hi = time_hi)

    # log[F(t_hi) - F(t_lo)] computed via Distributions.logdiffcdf
    # equivalent: logsubexp(logcdf(N, w_hi), logcdf(N, w_lo))
    expected = 0.0
    for i in eachindex(y_lo)
        w_hi = (log(time_hi[i]) - η[i]) / σ
        w_lo = (log(y_lo[i]) - η[i]) / σ
        expected += LatentGaussianModels.logsubexp(
            Distributions.logcdf(N, w_hi), Distributions.logcdf(N, w_lo))
    end
    @test log_density(ℓ, y_lo, η, θ) ≈ expected
end

@testset "LognormalSurvLikelihood — pointwise_cdf" begin
    rng = Random.Xoshiro(207)
    y = abs.(randn(rng, 4)) .+ 0.1
    η = randn(rng, 4) .* 0.5
    log_τ = 0.2
    τ = exp(log_τ)
    σ = sqrt(1 / τ)
    θ = [log_τ]
    N = Distributions.Normal()

    # Fast path: F(t) = Φ((log t − η)/σ)
    ℓ_fast = LognormalSurvLikelihood()
    cdf_fast = pointwise_cdf(ℓ_fast, y, η, θ)
    @test cdf_fast ≈ [Distributions.cdf(N, (log(y[i]) - η[i]) / σ)
                      for i in eachindex(y)]
    @test all(0 .≤ cdf_fast .≤ 1)

    # All-NONE censored variant: same numbers
    ℓ_none = LognormalSurvLikelihood(censoring = fill(NONE, 4))
    @test pointwise_cdf(ℓ_none, y, η, θ) ≈ cdf_fast

    # Censored variant with non-NONE rows: undefined, throws
    ℓ_cens = LognormalSurvLikelihood(censoring = [NONE, RIGHT, NONE, NONE])
    @test_throws ArgumentError pointwise_cdf(ℓ_cens, y, η, θ)
end

@testset "LognormalSurvLikelihood — pointwise_log_density per-mode" begin
    rng = Random.Xoshiro(208)
    y_lo = abs.(randn(rng, 4)) .+ 0.1
    time_hi = y_lo .+ 0.5
    η = randn(rng, 4) .* 0.4
    log_τ = -0.3
    τ = exp(log_τ)
    σ = sqrt(1 / τ)
    θ = [log_τ]
    N = Distributions.Normal()
    ℓ = LognormalSurvLikelihood(
        censoring = [NONE, RIGHT, LEFT, INTERVAL], time_hi = time_hi)

    pp = pointwise_log_density(ℓ, y_lo, η, θ)

    # NONE: log lognormal density
    @test pp[1] ≈ -log(y_lo[1]) - 0.5 * log(2π) + 0.5 * log_τ -
                  0.5 * τ * (log(y_lo[1]) - η[1])^2

    # RIGHT: log Φ((η − log t)/σ) = logccdf(N, (log t − η)/σ)
    @test pp[2] ≈ Distributions.logccdf(N, (log(y_lo[2]) - η[2]) / σ)

    # LEFT: log Φ((log t − η)/σ)
    @test pp[3] ≈ Distributions.logcdf(N, (log(y_lo[3]) - η[3]) / σ)

    # INTERVAL
    w_hi = (log(time_hi[4]) - η[4]) / σ
    w_lo = (log(y_lo[4]) - η[4]) / σ
    @test pp[4] ≈ LatentGaussianModels.logsubexp(
        Distributions.logcdf(N, w_hi), Distributions.logcdf(N, w_lo))
end

using LatentGaussianModels: ExponentialLikelihood, LogLink,
    Censoring, NONE, RIGHT, LEFT, INTERVAL,
    log_density, ∇_η_log_density, ∇²_η_log_density, ∇³_η_log_density,
    pointwise_log_density, pointwise_cdf,
    nhyperparameters, link
using LatentGaussianModels: validate_censoring, logsubexp

# Reuses the same FD helper used in test_likelihoods.jl.
function fd_grad_exp(f, η, h = 1.0e-6)
    g = similar(η)
    for i in eachindex(η)
        ep = copy(η); ep[i] += h
        em = copy(η); em[i] -= h
        g[i] = (f(ep) - f(em)) / (2h)
    end
    return g
end

@testset "Censoring constructors" begin
    @test Censoring(:none) === NONE
    @test Censoring(:right) === RIGHT
    @test Censoring(:left) === LEFT
    @test Censoring(:interval) === INTERVAL
    @test_throws ArgumentError Censoring(:invalid)
end

@testset "logsubexp helper" begin
    @test logsubexp(2.0, 1.0) ≈ log(exp(2.0) - exp(1.0))
    @test logsubexp(0.5, -3.0) ≈ log(exp(0.5) - exp(-3.0))
    @test_throws ArgumentError logsubexp(1.0, 2.0)
    @test_throws ArgumentError logsubexp(1.0, 1.0)
end

@testset "validate_censoring" begin
    y = [1.0, 2.0, 3.0]
    @test validate_censoring(nothing, nothing, y) === nothing
    @test validate_censoring([NONE, RIGHT, LEFT], nothing, y) === nothing

    @test_throws DimensionMismatch validate_censoring([NONE, RIGHT], nothing, y)

    @test_throws ArgumentError validate_censoring(
        [NONE, INTERVAL, NONE], nothing, y)
    @test_throws DimensionMismatch validate_censoring(
        [NONE, INTERVAL, NONE], [1.5, 2.5], y)
    @test_throws ArgumentError validate_censoring(
        [NONE, INTERVAL, NONE], [0.0, 1.5, 0.0], y)
    @test validate_censoring(
        [NONE, INTERVAL, NONE], [0.0, 3.0, 0.0], y) === nothing
end

@testset "ExponentialLikelihood — fast path (all uncensored)" begin
    rng = Random.Xoshiro(42)
    y = abs.(randn(rng, 10)) .+ 0.1
    η = randn(rng, 10) .* 0.5
    θ = Float64[]

    ℓ = ExponentialLikelihood()
    @test nhyperparameters(ℓ) == 0
    @test link(ℓ) isa LogLink

    lp = log_density(ℓ, y, η, θ)
    @test isfinite(lp)
    @test lp ≈ sum(η .- exp.(η) .* y)
    @test sum(pointwise_log_density(ℓ, y, η, θ)) ≈ lp

    g = ∇_η_log_density(ℓ, y, η, θ)
    g_fd = fd_grad_exp(h -> log_density(ℓ, y, h, θ), η)
    @test g ≈ g_fd atol = 1.0e-4
    @test g ≈ 1 .- exp.(η) .* y

    H = ∇²_η_log_density(ℓ, y, η, θ)
    H_fd = fd_grad_exp(h -> sum(∇_η_log_density(ℓ, y, h, θ)), η)
    @test H ≈ H_fd atol = 1.0e-4
    @test H ≈ -exp.(η) .* y

    T = ∇³_η_log_density(ℓ, y, η, θ)
    T_fd = fd_grad_exp(h -> sum(∇²_η_log_density(ℓ, y, h, θ)), η)
    @test T ≈ T_fd atol = 1.0e-4
end

@testset "ExponentialLikelihood — mixed censoring" begin
    rng = Random.Xoshiro(43)
    y_lo = abs.(randn(rng, 8)) .+ 0.1
    η = randn(rng, 8) .* 0.5
    θ = Float64[]
    censoring = [NONE, RIGHT, LEFT, INTERVAL, NONE, RIGHT, LEFT, INTERVAL]
    time_hi = y_lo .+ abs.(randn(rng, 8)) .+ 0.1

    ℓ = ExponentialLikelihood(censoring = censoring, time_hi = time_hi)

    lp = log_density(ℓ, y_lo, η, θ)
    @test isfinite(lp)
    @test sum(pointwise_log_density(ℓ, y_lo, η, θ)) ≈ lp

    g = ∇_η_log_density(ℓ, y_lo, η, θ)
    g_fd = fd_grad_exp(h -> log_density(ℓ, y_lo, h, θ), η)
    @test g ≈ g_fd atol = 1.0e-4

    H = ∇²_η_log_density(ℓ, y_lo, η, θ)
    H_fd = fd_grad_exp(h -> sum(∇_η_log_density(ℓ, y_lo, h, θ)), η)
    @test H ≈ H_fd atol = 1.0e-4

    T = ∇³_η_log_density(ℓ, y_lo, η, θ)
    T_fd = fd_grad_exp(h -> sum(∇²_η_log_density(ℓ, y_lo, h, θ)), η)
    @test T ≈ T_fd atol = 1.0e-4
end

@testset "ExponentialLikelihood — fast vs mixed agree on all-NONE" begin
    rng = Random.Xoshiro(44)
    y = abs.(randn(rng, 6)) .+ 0.1
    η = randn(rng, 6) .* 0.5
    θ = Float64[]

    ℓ_fast = ExponentialLikelihood()
    ℓ_mixed = ExponentialLikelihood(censoring = fill(NONE, 6))

    @test log_density(ℓ_fast, y, η, θ) ≈ log_density(ℓ_mixed, y, η, θ)
    @test ∇_η_log_density(ℓ_fast, y, η, θ) ≈
          ∇_η_log_density(ℓ_mixed, y, η, θ)
    @test ∇²_η_log_density(ℓ_fast, y, η, θ) ≈
          ∇²_η_log_density(ℓ_mixed, y, η, θ)
    @test ∇³_η_log_density(ℓ_fast, y, η, θ) ≈
          ∇³_η_log_density(ℓ_mixed, y, η, θ)
end

@testset "ExponentialLikelihood — Symbol coercion" begin
    censoring_sym = [:none, :right, :left, :interval]
    time_hi = [0.0, 0.0, 0.0, 5.0]
    ℓ = ExponentialLikelihood(censoring = censoring_sym, time_hi = time_hi)
    @test ℓ.censoring isa Vector{Censoring}
    @test ℓ.censoring == [NONE, RIGHT, LEFT, INTERVAL]
end

@testset "ExponentialLikelihood — bad censoring type rejected" begin
    @test_throws ArgumentError ExponentialLikelihood(censoring = [1, 2, 3])
    @test_throws ArgumentError ExponentialLikelihood(
        censoring = ["none", "right"])
end

@testset "ExponentialLikelihood — RIGHT-only closed form" begin
    rng = Random.Xoshiro(45)
    y = abs.(randn(rng, 5)) .+ 0.1
    η = randn(rng, 5) .* 0.5
    θ = Float64[]
    ℓ = ExponentialLikelihood(censoring = fill(RIGHT, 5))
    @test log_density(ℓ, y, η, θ) ≈ -sum(exp.(η) .* y)
    @test ∇_η_log_density(ℓ, y, η, θ) ≈ -exp.(η) .* y
    @test ∇²_η_log_density(ℓ, y, η, θ) ≈ -exp.(η) .* y
    @test ∇³_η_log_density(ℓ, y, η, θ) ≈ -exp.(η) .* y
end

@testset "ExponentialLikelihood — INTERVAL closed form" begin
    # log p = -u_lo + log(1 - exp(-Δ)),  Δ = λ(t_hi - t_lo)
    rng = Random.Xoshiro(46)
    y_lo = abs.(randn(rng, 4)) .+ 0.1
    time_hi = y_lo .+ 0.5 .+ abs.(randn(rng, 4))
    η = randn(rng, 4) .* 0.3
    θ = Float64[]
    ℓ = ExponentialLikelihood(
        censoring = fill(INTERVAL, 4), time_hi = time_hi)

    λ = exp.(η)
    u_lo = λ .* y_lo
    Δ = λ .* (time_hi .- y_lo)
    expected = sum(-u_lo .+ log.(-expm1.(-Δ)))
    @test log_density(ℓ, y_lo, η, θ) ≈ expected
end

@testset "ExponentialLikelihood — pointwise_cdf" begin
    rng = Random.Xoshiro(47)
    y = abs.(randn(rng, 4)) .+ 0.1
    η = randn(rng, 4) .* 0.5
    θ = Float64[]

    # Fast path
    ℓ_fast = ExponentialLikelihood()
    cdf_fast = pointwise_cdf(ℓ_fast, y, η, θ)
    @test cdf_fast ≈ 1 .- exp.(-exp.(η) .* y)
    @test all(0 .≤ cdf_fast .≤ 1)

    # All-NONE censored variant: same numbers
    ℓ_none = ExponentialLikelihood(censoring = fill(NONE, 4))
    @test pointwise_cdf(ℓ_none, y, η, θ) ≈ cdf_fast

    # Censored variant with non-NONE rows: undefined, throws
    ℓ_cens = ExponentialLikelihood(censoring = [NONE, RIGHT, NONE, NONE])
    @test_throws ArgumentError pointwise_cdf(ℓ_cens, y, η, θ)
end

@testset "ExponentialLikelihood — pointwise_log_density per-mode" begin
    rng = Random.Xoshiro(48)
    y_lo = abs.(randn(rng, 4)) .+ 0.1
    time_hi = y_lo .+ 0.5
    η = randn(rng, 4) .* 0.4
    θ = Float64[]
    ℓ = ExponentialLikelihood(
        censoring = [NONE, RIGHT, LEFT, INTERVAL], time_hi = time_hi)

    pp = pointwise_log_density(ℓ, y_lo, η, θ)
    λ = exp.(η)
    @test pp[1] ≈ η[1] - λ[1] * y_lo[1]                      # NONE
    @test pp[2] ≈ -λ[2] * y_lo[2]                            # RIGHT
    @test pp[3] ≈ log(-expm1(-λ[3] * y_lo[3]))               # LEFT
    Δ4 = λ[4] * (time_hi[4] - y_lo[4])
    @test pp[4] ≈ -λ[4] * y_lo[4] + log(-expm1(-Δ4))         # INTERVAL
end

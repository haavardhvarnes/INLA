using LatentGaussianModels: WeibullLikelihood, ExponentialLikelihood, LogLink,
                            Censoring, NONE, RIGHT, LEFT, INTERVAL,
                            log_density, ∇_η_log_density, ∇²_η_log_density,
                            ∇³_η_log_density,
                            pointwise_log_density, pointwise_cdf,
                            nhyperparameters, initial_hyperparameters, link, log_hyperprior,
                            GammaPrecision

# Reuses the same FD helper used in test_likelihoods.jl.
function fd_grad_wb(f, η, h=1.0e-6)
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

const WB_LOG_ALPHAS = (-0.7, 0.0, 0.7)   # α ≈ 0.5, 1.0, 2.0

@testset "WeibullLikelihood — defaults / contract" begin
    ℓ = WeibullLikelihood()
    @test nhyperparameters(ℓ) == 1
    @test initial_hyperparameters(ℓ) == [0.0]
    @test link(ℓ) isa LogLink
    @test ℓ.censoring === nothing
    @test ℓ.hyperprior isa GammaPrecision

    # log-hyperprior delegates to the GammaPrecision prior on log α
    @test log_hyperprior(ℓ, [0.0]) isa Real
    @test log_hyperprior(ℓ, [-0.5]) isa Real

    # Non-LogLink rejected
    struct _DummyLink <: LatentGaussianModels.AbstractLinkFunction end
    @test_throws ArgumentError WeibullLikelihood(link=_DummyLink())
end

@testset "WeibullLikelihood — fast path (all uncensored)" begin
    rng = Random.Xoshiro(101)
    y = abs.(randn(rng, 10)) .+ 0.1
    η = randn(rng, 10) .* 0.5

    ℓ = WeibullLikelihood()

    for log_α in WB_LOG_ALPHAS
        θ = [log_α]
        α = exp(log_α)

        lp = log_density(ℓ, y, η, θ)
        @test isfinite(lp)
        @test lp ≈ sum(@. log_α + η + (α - 1) * log(y) - exp(η) * y^α)
        @test sum(pointwise_log_density(ℓ, y, η, θ)) ≈ lp

        g = ∇_η_log_density(ℓ, y, η, θ)
        g_fd = fd_grad_wb(h -> log_density(ℓ, y, h, θ), η)
        @test g≈g_fd atol=1.0e-4
        @test g ≈ @. 1 - exp(η) * y^α

        H = ∇²_η_log_density(ℓ, y, η, θ)
        H_fd = fd_grad_wb(h -> sum(∇_η_log_density(ℓ, y, h, θ)), η)
        @test H≈H_fd atol=1.0e-4
        @test H ≈ @. -exp(η) * y^α

        T = ∇³_η_log_density(ℓ, y, η, θ)
        T_fd = fd_grad_wb(h -> sum(∇²_η_log_density(ℓ, y, h, θ)), η)
        @test T≈T_fd atol=1.0e-4
    end
end

@testset "WeibullLikelihood — mixed censoring" begin
    rng = Random.Xoshiro(102)
    y_lo = abs.(randn(rng, 8)) .+ 0.1
    η = randn(rng, 8) .* 0.5
    censoring = [NONE, RIGHT, LEFT, INTERVAL, NONE, RIGHT, LEFT, INTERVAL]
    time_hi = y_lo .+ abs.(randn(rng, 8)) .+ 0.1

    ℓ = WeibullLikelihood(censoring=censoring, time_hi=time_hi)

    for log_α in WB_LOG_ALPHAS
        θ = [log_α]

        lp = log_density(ℓ, y_lo, η, θ)
        @test isfinite(lp)
        @test sum(pointwise_log_density(ℓ, y_lo, η, θ)) ≈ lp

        g = ∇_η_log_density(ℓ, y_lo, η, θ)
        g_fd = fd_grad_wb(h -> log_density(ℓ, y_lo, h, θ), η)
        @test g≈g_fd atol=1.0e-4

        H = ∇²_η_log_density(ℓ, y_lo, η, θ)
        H_fd = fd_grad_wb(h -> sum(∇_η_log_density(ℓ, y_lo, h, θ)), η)
        @test H≈H_fd atol=1.0e-4

        T = ∇³_η_log_density(ℓ, y_lo, η, θ)
        T_fd = fd_grad_wb(h -> sum(∇²_η_log_density(ℓ, y_lo, h, θ)), η)
        @test T≈T_fd atol=1.0e-4
    end
end

@testset "WeibullLikelihood — fast vs mixed agree on all-NONE" begin
    rng = Random.Xoshiro(103)
    y = abs.(randn(rng, 6)) .+ 0.1
    η = randn(rng, 6) .* 0.5

    ℓ_fast = WeibullLikelihood()
    ℓ_mixed = WeibullLikelihood(censoring=fill(NONE, 6))

    for log_α in WB_LOG_ALPHAS
        θ = [log_α]
        @test log_density(ℓ_fast, y, η, θ) ≈ log_density(ℓ_mixed, y, η, θ)
        @test ∇_η_log_density(ℓ_fast, y, η, θ) ≈
              ∇_η_log_density(ℓ_mixed, y, η, θ)
        @test ∇²_η_log_density(ℓ_fast, y, η, θ) ≈
              ∇²_η_log_density(ℓ_mixed, y, η, θ)
        @test ∇³_η_log_density(ℓ_fast, y, η, θ) ≈
              ∇³_η_log_density(ℓ_mixed, y, η, θ)
    end
end

@testset "WeibullLikelihood — α = 1 reduces to Exponential" begin
    # At log α = 0 (α = 1): log α + (α-1) log t = 0, t^α = t, so the
    # Weibull log-density and η-derivatives must match Exponential exactly.
    rng = Random.Xoshiro(104)
    y = abs.(randn(rng, 7)) .+ 0.1
    η = randn(rng, 7) .* 0.5
    θ_w = [0.0]
    θ_e = Float64[]

    ℓ_w = WeibullLikelihood()
    ℓ_e = ExponentialLikelihood()

    @test log_density(ℓ_w, y, η, θ_w) ≈ log_density(ℓ_e, y, η, θ_e)
    @test ∇_η_log_density(ℓ_w, y, η, θ_w) ≈ ∇_η_log_density(ℓ_e, y, η, θ_e)
    @test ∇²_η_log_density(ℓ_w, y, η, θ_w) ≈ ∇²_η_log_density(ℓ_e, y, η, θ_e)
    @test ∇³_η_log_density(ℓ_w, y, η, θ_w) ≈ ∇³_η_log_density(ℓ_e, y, η, θ_e)

    # Mixed-censoring agreement at α = 1 too
    cens = [NONE, RIGHT, LEFT, INTERVAL, NONE, RIGHT, LEFT]
    t_hi = y .+ 0.5
    ℓ_w_m = WeibullLikelihood(censoring=cens, time_hi=t_hi)
    ℓ_e_m = ExponentialLikelihood(censoring=cens, time_hi=t_hi)
    @test log_density(ℓ_w_m, y, η, θ_w) ≈ log_density(ℓ_e_m, y, η, θ_e)
    @test ∇_η_log_density(ℓ_w_m, y, η, θ_w) ≈
          ∇_η_log_density(ℓ_e_m, y, η, θ_e)
    @test ∇²_η_log_density(ℓ_w_m, y, η, θ_w) ≈
          ∇²_η_log_density(ℓ_e_m, y, η, θ_e)
end

@testset "WeibullLikelihood — Symbol coercion" begin
    censoring_sym = [:none, :right, :left, :interval]
    time_hi = [0.0, 0.0, 0.0, 5.0]
    ℓ = WeibullLikelihood(censoring=censoring_sym, time_hi=time_hi)
    @test ℓ.censoring isa Vector{Censoring}
    @test ℓ.censoring == [NONE, RIGHT, LEFT, INTERVAL]
end

@testset "WeibullLikelihood — bad censoring type rejected" begin
    @test_throws ArgumentError WeibullLikelihood(censoring=[1, 2, 3])
    @test_throws ArgumentError WeibullLikelihood(
        censoring=["none", "right"])
end

@testset "WeibullLikelihood — RIGHT-only closed form" begin
    rng = Random.Xoshiro(105)
    y = abs.(randn(rng, 5)) .+ 0.1
    η = randn(rng, 5) .* 0.5
    log_α = 0.4
    α = exp(log_α)
    θ = [log_α]
    ℓ = WeibullLikelihood(censoring=fill(RIGHT, 5))
    @test log_density(ℓ, y, η, θ) ≈ -sum(@. exp(η) * y^α)
    @test ∇_η_log_density(ℓ, y, η, θ) ≈ @. -exp(η) * y^α
    @test ∇²_η_log_density(ℓ, y, η, θ) ≈ @. -exp(η) * y^α
    @test ∇³_η_log_density(ℓ, y, η, θ) ≈ @. -exp(η) * y^α
end

@testset "WeibullLikelihood — INTERVAL closed form" begin
    # log p = -u_lo + log(1 - exp(-Δ)),  Δ = λ(t_hi^α - t_lo^α)
    rng = Random.Xoshiro(106)
    y_lo = abs.(randn(rng, 4)) .+ 0.1
    time_hi = y_lo .+ 0.5 .+ abs.(randn(rng, 4))
    η = randn(rng, 4) .* 0.3
    log_α = 0.3
    α = exp(log_α)
    θ = [log_α]
    ℓ = WeibullLikelihood(
        censoring=fill(INTERVAL, 4), time_hi=time_hi)

    λ = exp.(η)
    u_lo = λ .* y_lo .^ α
    Δ = λ .* (time_hi .^ α .- y_lo .^ α)
    expected = sum(-u_lo .+ log.(-expm1.(-Δ)))
    @test log_density(ℓ, y_lo, η, θ) ≈ expected
end

@testset "WeibullLikelihood — pointwise_cdf" begin
    rng = Random.Xoshiro(107)
    y = abs.(randn(rng, 4)) .+ 0.1
    η = randn(rng, 4) .* 0.5
    log_α = 0.2
    α = exp(log_α)
    θ = [log_α]

    # Fast path: F(t) = 1 - exp(-λ t^α)
    ℓ_fast = WeibullLikelihood()
    cdf_fast = pointwise_cdf(ℓ_fast, y, η, θ)
    @test cdf_fast ≈ 1 .- exp.(-exp.(η) .* y .^ α)
    @test all(0 .≤ cdf_fast .≤ 1)

    # All-NONE censored variant: same numbers
    ℓ_none = WeibullLikelihood(censoring=fill(NONE, 4))
    @test pointwise_cdf(ℓ_none, y, η, θ) ≈ cdf_fast

    # Censored variant with non-NONE rows: undefined, throws
    ℓ_cens = WeibullLikelihood(censoring=[NONE, RIGHT, NONE, NONE])
    @test_throws ArgumentError pointwise_cdf(ℓ_cens, y, η, θ)
end

@testset "WeibullLikelihood — pointwise_log_density per-mode" begin
    rng = Random.Xoshiro(108)
    y_lo = abs.(randn(rng, 4)) .+ 0.1
    time_hi = y_lo .+ 0.5
    η = randn(rng, 4) .* 0.4
    log_α = -0.3
    α = exp(log_α)
    θ = [log_α]
    ℓ = WeibullLikelihood(
        censoring=[NONE, RIGHT, LEFT, INTERVAL], time_hi=time_hi)

    pp = pointwise_log_density(ℓ, y_lo, η, θ)
    λ = exp.(η)
    t_α = y_lo .^ α
    @test pp[1] ≈ log_α + η[1] + (α - 1) * log(y_lo[1]) - λ[1] * t_α[1]   # NONE
    @test pp[2] ≈ -λ[2] * t_α[2]                                          # RIGHT
    @test pp[3] ≈ log(-expm1(-λ[3] * t_α[3]))                             # LEFT
    Δ4 = λ[4] * (time_hi[4]^α - t_α[4])
    @test pp[4] ≈ -λ[4] * t_α[4] + log(-expm1(-Δ4))                       # INTERVAL
end

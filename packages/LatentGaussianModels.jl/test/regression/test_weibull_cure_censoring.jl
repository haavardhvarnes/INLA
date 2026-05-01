using LatentGaussianModels: WeibullCureLikelihood, WeibullLikelihood, LogLink,
    Censoring, NONE, RIGHT, LEFT, INTERVAL,
    log_density, ∇_η_log_density, ∇²_η_log_density, ∇³_η_log_density,
    pointwise_log_density, pointwise_cdf,
    nhyperparameters, initial_hyperparameters, link, log_hyperprior,
    GammaPrecision, LogitBeta

using DifferentiationInterface: gradient, AutoForwardDiff
using ForwardDiff: ForwardDiff   # required by `AutoForwardDiff()` extension on Julia 1.10

# AD-based oracles (independent of the closed-form chain rule in production).
# Backend kept local to this file; ForwardDiff + DI are test-only deps.
const _AD = AutoForwardDiff()

ad_grad_wc(f, η) = gradient(f, _AD, η)
ad_hess_diag_wc(ℓ, y, θ) = (η -> ad_grad_wc(h -> sum(∇_η_log_density(ℓ, y, h, θ)), η))
ad_third_diag_wc(ℓ, y, θ) = (η -> ad_grad_wc(h -> sum(∇²_η_log_density(ℓ, y, h, θ)), η))

const WC_LOG_ALPHAS = (-0.5, 0.0, 0.5)        # α ≈ 0.61, 1.0, 1.65
const WC_LOGIT_PS = (-2.197, -0.847, 0.0)     # p ≈ 0.10, 0.30, 0.50

@testset "WeibullCureLikelihood — defaults / contract" begin
    ℓ = WeibullCureLikelihood()
    @test nhyperparameters(ℓ) == 2
    @test initial_hyperparameters(ℓ) == [0.0, log(0.1 / 0.9)]
    @test link(ℓ) isa LogLink
    @test ℓ.censoring === nothing
    @test ℓ.time_hi === nothing
    @test ℓ.hyperprior_alpha isa GammaPrecision
    @test ℓ.hyperprior_p isa LogitBeta

    # log-hyperprior sums shape + cure-fraction priors
    @test log_hyperprior(ℓ, [0.0, 0.0]) isa Real
    @test log_hyperprior(ℓ, [-0.5, 1.0]) isa Real

    # Non-LogLink rejected
    struct _DummyLinkWC <: LatentGaussianModels.AbstractLinkFunction end
    @test_throws ArgumentError WeibullCureLikelihood(link = _DummyLinkWC())
end

@testset "WeibullCureLikelihood — fast path (all uncensored)" begin
    rng = Random.Xoshiro(401)
    y = abs.(randn(rng, 8)) .+ 0.2
    η = randn(rng, 8) .* 0.4

    ℓ = WeibullCureLikelihood()

    for log_α in WC_LOG_ALPHAS, logit_p in WC_LOGIT_PS
        θ = [log_α, logit_p]
        α = exp(log_α)
        p = inv(1 + exp(-logit_p))

        # Closed-form NONE branch: log f = log(1-p) + log α + (α-1) log y + η - exp(η) y^α
        lp = log_density(ℓ, y, η, θ)
        @test isfinite(lp)
        @test lp ≈ sum(@. log1p(-p) + log_α + (α - 1) * log(y) + η - exp(η) * y^α)
        @test sum(pointwise_log_density(ℓ, y, η, θ)) ≈ lp

        # Closed-form gradient: ∂η log f = 1 - exp(η) y^α (independent of p)
        g = ∇_η_log_density(ℓ, y, η, θ)
        @test g ≈ @. 1 - exp(η) * y^α
        # AD cross-check
        g_ad = ad_grad_wc(h -> log_density(ℓ, y, h, θ), η)
        @test g ≈ g_ad rtol = 1.0e-10

        # Closed-form Hessian diagonal: -exp(η) y^α
        H = ∇²_η_log_density(ℓ, y, η, θ)
        @test H ≈ @. -exp(η) * y^α
        H_ad = ad_hess_diag_wc(ℓ, y, θ)(η)
        @test H ≈ H_ad rtol = 1.0e-10

        # Closed-form ∇³: -exp(η) y^α
        T = ∇³_η_log_density(ℓ, y, η, θ)
        @test T ≈ @. -exp(η) * y^α
        T_ad = ad_third_diag_wc(ℓ, y, θ)(η)
        @test T ≈ T_ad rtol = 1.0e-10
    end
end

@testset "WeibullCureLikelihood — mixed censoring (AD oracle)" begin
    rng = Random.Xoshiro(402)
    y_lo = abs.(randn(rng, 8)) .+ 0.2
    η = randn(rng, 8) .* 0.3
    censoring = [NONE, RIGHT, LEFT, INTERVAL, NONE, RIGHT, LEFT, INTERVAL]
    time_hi = y_lo .+ abs.(randn(rng, 8)) .+ 0.5

    ℓ = WeibullCureLikelihood(censoring = censoring, time_hi = time_hi)

    for log_α in WC_LOG_ALPHAS, logit_p in WC_LOGIT_PS
        θ = [log_α, logit_p]

        lp = log_density(ℓ, y_lo, η, θ)
        @test isfinite(lp)
        @test sum(pointwise_log_density(ℓ, y_lo, η, θ)) ≈ lp

        g = ∇_η_log_density(ℓ, y_lo, η, θ)
        g_ad = ad_grad_wc(h -> log_density(ℓ, y_lo, h, θ), η)
        @test g ≈ g_ad rtol = 1.0e-10

        H = ∇²_η_log_density(ℓ, y_lo, η, θ)
        H_ad = ad_hess_diag_wc(ℓ, y_lo, θ)(η)
        @test H ≈ H_ad rtol = 1.0e-10

        T = ∇³_η_log_density(ℓ, y_lo, η, θ)
        T_ad = ad_third_diag_wc(ℓ, y_lo, θ)(η)
        @test T ≈ T_ad rtol = 1.0e-9
    end
end

@testset "WeibullCureLikelihood — fast vs mixed agree on all-NONE" begin
    rng = Random.Xoshiro(403)
    y = abs.(randn(rng, 6)) .+ 0.2
    η = randn(rng, 6) .* 0.3

    ℓ_fast = WeibullCureLikelihood()
    ℓ_mixed = WeibullCureLikelihood(censoring = fill(NONE, 6))

    for log_α in WC_LOG_ALPHAS, logit_p in WC_LOGIT_PS
        θ = [log_α, logit_p]
        @test log_density(ℓ_fast, y, η, θ) ≈ log_density(ℓ_mixed, y, η, θ)
        @test ∇_η_log_density(ℓ_fast, y, η, θ) ≈
              ∇_η_log_density(ℓ_mixed, y, η, θ)
        @test ∇²_η_log_density(ℓ_fast, y, η, θ) ≈
              ∇²_η_log_density(ℓ_mixed, y, η, θ)
        @test ∇³_η_log_density(ℓ_fast, y, η, θ) ≈
              ∇³_η_log_density(ℓ_mixed, y, η, θ)
    end
end

@testset "WeibullCureLikelihood — η-derivatives match Weibull on NONE/LEFT/INTERVAL" begin
    # The mixture-cure log-density factorises as log(1-p) + (Weibull log f/F/Δ
    # piece) for NONE/LEFT/INTERVAL rows. Since log(1-p) is η-independent, the
    # η-derivatives must match plain `WeibullLikelihood` *exactly*. RIGHT is
    # excluded — that branch genuinely depends on p.
    rng = Random.Xoshiro(404)
    n = 6
    y_lo = abs.(randn(rng, n)) .+ 0.2
    η = randn(rng, n) .* 0.3
    time_hi = y_lo .+ abs.(randn(rng, n)) .+ 0.5
    censoring = [NONE, LEFT, INTERVAL, NONE, LEFT, INTERVAL]

    ℓ_w = WeibullLikelihood(censoring = censoring, time_hi = time_hi)
    ℓ_wc = WeibullCureLikelihood(censoring = censoring, time_hi = time_hi)

    for log_α in WC_LOG_ALPHAS, logit_p in WC_LOGIT_PS
        θ_w = [log_α]
        θ_wc = [log_α, logit_p]

        @test ∇_η_log_density(ℓ_wc, y_lo, η, θ_wc) ≈
              ∇_η_log_density(ℓ_w, y_lo, η, θ_w)
        @test ∇²_η_log_density(ℓ_wc, y_lo, η, θ_wc) ≈
              ∇²_η_log_density(ℓ_w, y_lo, η, θ_w)
        @test ∇³_η_log_density(ℓ_wc, y_lo, η, θ_wc) ≈
              ∇³_η_log_density(ℓ_w, y_lo, η, θ_w)
    end
end

@testset "WeibullCureLikelihood — RIGHT-only closed form (gradient)" begin
    # log S = log[p + (1-p) exp(-u)],  u = exp(η) y^α
    # ∂η log S = -(1-p) u v / [p + (1-p) v],  v = exp(-u)
    rng = Random.Xoshiro(405)
    y = abs.(randn(rng, 5)) .+ 0.2
    η = randn(rng, 5) .* 0.5
    log_α = 0.3
    α = exp(log_α)

    for logit_p in WC_LOGIT_PS
        p = inv(1 + exp(-logit_p))
        θ = [log_α, logit_p]
        ℓ = WeibullCureLikelihood(censoring = fill(RIGHT, 5))

        u = @. exp(η) * y^α
        v = @. exp(-u)
        D = @. p + (1 - p) * v
        @test log_density(ℓ, y, η, θ) ≈ sum(@. log(D))

        Dp = @. -(1 - p) * u * v
        @test ∇_η_log_density(ℓ, y, η, θ) ≈ @. Dp / D

        # Hessian + ∇³ via AD (chain rule too unwieldy to hand-code as an
        # *independent* oracle — closed form would just retrace production).
        H_ad = ad_hess_diag_wc(ℓ, y, θ)(η)
        @test ∇²_η_log_density(ℓ, y, η, θ) ≈ H_ad rtol = 1.0e-10
        T_ad = ad_third_diag_wc(ℓ, y, θ)(η)
        @test ∇³_η_log_density(ℓ, y, η, θ) ≈ T_ad rtol = 1.0e-9
    end
end

@testset "WeibullCureLikelihood — p → 0 reduces to Weibull (RIGHT)" begin
    # As p → 0, RIGHT branch must reduce to plain Weibull RIGHT. Use a small
    # but finite p; tolerance scales with p (the leading deviation is O(p)).
    rng = Random.Xoshiro(406)
    y = abs.(randn(rng, 5)) .+ 0.2
    η = randn(rng, 5) .* 0.3
    log_α = 0.0
    p = 1.0e-8
    logit_p_small = log(p / (1 - p))
    θ_wc = [log_α, logit_p_small]
    θ_w = [log_α]

    ℓ_wc = WeibullCureLikelihood(censoring = fill(RIGHT, 5))
    ℓ_w  = WeibullLikelihood(censoring = fill(RIGHT, 5))

    g_wc = ∇_η_log_density(ℓ_wc, y, η, θ_wc)
    g_w  = ∇_η_log_density(ℓ_w,  y, η, θ_w)
    @test maximum(abs, g_wc .- g_w) < 1.0e-6

    H_wc = ∇²_η_log_density(ℓ_wc, y, η, θ_wc)
    H_w  = ∇²_η_log_density(ℓ_w,  y, η, θ_w)
    @test maximum(abs, H_wc .- H_w) < 1.0e-6
end

@testset "WeibullCureLikelihood — Symbol coercion" begin
    censoring_sym = [:none, :right, :left, :interval]
    time_hi = [0.0, 0.0, 0.0, 5.0]
    ℓ = WeibullCureLikelihood(censoring = censoring_sym, time_hi = time_hi)
    @test ℓ.censoring isa Vector{Censoring}
    @test ℓ.censoring == [NONE, RIGHT, LEFT, INTERVAL]
end

@testset "WeibullCureLikelihood — bad censoring type rejected" begin
    @test_throws ArgumentError WeibullCureLikelihood(censoring = [1, 2, 3])
    @test_throws ArgumentError WeibullCureLikelihood(censoring = ["none", "right"])
end

@testset "WeibullCureLikelihood — pointwise_log_density per-mode" begin
    rng = Random.Xoshiro(407)
    y_lo = abs.(randn(rng, 4)) .+ 0.2
    time_hi = y_lo .+ 0.5
    η = randn(rng, 4) .* 0.3
    log_α = 0.2
    α = exp(log_α)
    logit_p = -1.0
    p = inv(1 + exp(-logit_p))
    θ = [log_α, logit_p]
    ℓ = WeibullCureLikelihood(
        censoring = [NONE, RIGHT, LEFT, INTERVAL], time_hi = time_hi)

    pp = pointwise_log_density(ℓ, y_lo, η, θ)

    # NONE: log(1-p) + log α + (α-1) log y + η - u
    u1 = exp(η[1]) * y_lo[1]^α
    @test pp[1] ≈ log1p(-p) + log_α + (α - 1) * log(y_lo[1]) + η[1] - u1

    # RIGHT: log[p + (1-p) exp(-u)]
    u2 = exp(η[2]) * y_lo[2]^α
    @test pp[2] ≈ log(p + (1 - p) * exp(-u2))

    # LEFT: log(1-p) + log(1 - exp(-u))
    u3 = exp(η[3]) * y_lo[3]^α
    @test pp[3] ≈ log1p(-p) + log(-expm1(-u3))

    # INTERVAL: log(1-p) - u_lo + log(1 - exp(-Δ))
    u4_lo = exp(η[4]) * y_lo[4]^α
    Δ = exp(η[4]) * (time_hi[4]^α - y_lo[4]^α)
    @test pp[4] ≈ log1p(-p) - u4_lo + log(-expm1(-Δ))
end

@testset "WeibullCureLikelihood — pointwise_cdf" begin
    rng = Random.Xoshiro(408)
    y = abs.(randn(rng, 4)) .+ 0.2
    η = randn(rng, 4) .* 0.4
    log_α = -0.2
    α = exp(log_α)
    logit_p = -0.5
    p = inv(1 + exp(-logit_p))
    θ = [log_α, logit_p]

    # Fast path: F(t) = (1 - p) (1 - exp(-u))
    ℓ_fast = WeibullCureLikelihood()
    cdf_fast = pointwise_cdf(ℓ_fast, y, η, θ)
    expected = @. (1 - p) * (-expm1(-exp(η) * y^α))
    @test cdf_fast ≈ expected
    @test all(0 .≤ cdf_fast .≤ 1 - p + 1.0e-12)

    # All-NONE censored variant: same numbers
    ℓ_none = WeibullCureLikelihood(censoring = fill(NONE, 4))
    @test pointwise_cdf(ℓ_none, y, η, θ) ≈ cdf_fast

    # Non-NONE rows: undefined under censoring, throws
    ℓ_cens = WeibullCureLikelihood(censoring = [NONE, RIGHT, NONE, NONE])
    @test_throws ArgumentError pointwise_cdf(ℓ_cens, y, η, θ)
end

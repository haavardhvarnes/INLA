using LatentGaussianModels: GaussianLikelihood, PoissonLikelihood,
    BinomialLikelihood, NegativeBinomialLikelihood, GammaLikelihood,
    IdentityLink, LogLink, LogitLink, ProbitLink,
    log_density, ∇_η_log_density, ∇²_η_log_density, ∇³_η_log_density,
    pointwise_log_density, pointwise_cdf,
    nhyperparameters, initial_hyperparameters, log_hyperprior
import Distributions

# Finite-difference sanity check on ∇_η and ∇²_η.
function fd_grad(f, η, h = 1.0e-6)
    g = similar(η)
    for i in eachindex(η)
        ep = copy(η); ep[i] += h
        em = copy(η); em[i] -= h
        g[i] = (f(ep) - f(em)) / (2h)
    end
    return g
end

@testset "GaussianLikelihood — IdentityLink" begin
    ℓ = GaussianLikelihood()
    rng = Random.Xoshiro(0)
    y = randn(rng, 8)
    η = randn(rng, 8)
    θ = [0.4]

    # Analytical gradient
    τ = exp(θ[1])
    @test ∇_η_log_density(ℓ, y, η, θ) ≈ τ .* (y .- η)
    @test all(∇²_η_log_density(ℓ, y, η, θ) .≈ -τ)

    # FD check on log_density
    g_fd = fd_grad(h -> log_density(ℓ, y, h, θ), η)
    @test ∇_η_log_density(ℓ, y, η, θ) ≈ g_fd atol = 1.0e-4
end

@testset "PoissonLikelihood — LogLink" begin
    ℓ = PoissonLikelihood()
    rng = Random.Xoshiro(1)
    y = [2, 0, 5, 3, 1]
    η = [0.1, -0.3, 1.2, 0.5, 0.0]
    θ = Float64[]

    lp = log_density(ℓ, y, η, θ)
    @test isfinite(lp)

    g = ∇_η_log_density(ℓ, y, η, θ)
    g_fd = fd_grad(h -> log_density(ℓ, y, h, θ), η)
    @test g ≈ g_fd atol = 1.0e-4

    H = ∇²_η_log_density(ℓ, y, η, θ)
    H_fd = fd_grad(h -> sum(∇_η_log_density(ℓ, y, h, θ)), η)
    # H_fd is row-sum of full Jacobian of g; diagonal extraction below:
    # Each g_i depends only on η_i, so Jacobian is diagonal and H_fd ≈ H.
    @test H ≈ H_fd atol = 1.0e-4
end

@testset "BinomialLikelihood — LogitLink" begin
    n_trials = [10, 10, 10, 10, 10]
    y = [3, 7, 0, 10, 5]
    ℓ = BinomialLikelihood(n_trials)
    η = [-0.5, 0.8, -2.0, 3.0, 0.0]
    θ = Float64[]

    @test isfinite(log_density(ℓ, y, η, θ))

    g = ∇_η_log_density(ℓ, y, η, θ)
    g_fd = fd_grad(h -> log_density(ℓ, y, h, θ), η)
    @test g ≈ g_fd atol = 1.0e-4

    # Diagonal Hessian is exactly -n p(1-p)
    p = 1 ./ (1 .+ exp.(-η))
    @test ∇²_η_log_density(ℓ, y, η, θ) ≈ -n_trials .* p .* (1 .- p)
end

@testset "NegativeBinomialLikelihood — LogLink, default" begin
    ℓ = NegativeBinomialLikelihood()
    @test nhyperparameters(ℓ) == 1
    @test initial_hyperparameters(ℓ) == [0.0]

    rng = Random.Xoshiro(2)
    y = [3, 0, 8, 2, 12, 1]
    η = [0.4, -0.5, 1.1, 0.2, 1.5, 0.0]
    θ = [log(2.5)]           # size = 2.5

    lp = log_density(ℓ, y, η, θ)
    @test isfinite(lp)

    # Agreement with Distributions.NegativeBinomial under the same parameterisation
    r = exp(θ[1])
    expected = sum(Distributions.logpdf(Distributions.NegativeBinomial(r, r / (r + exp(ηi))), yi)
                   for (yi, ηi) in zip(y, η))
    @test lp ≈ expected

    # FD grad
    g = ∇_η_log_density(ℓ, y, η, θ)
    g_fd = fd_grad(h -> log_density(ℓ, y, h, θ), η)
    @test g ≈ g_fd atol = 1.0e-4

    # FD hessian diagonal
    H = ∇²_η_log_density(ℓ, y, η, θ)
    H_fd = fd_grad(h -> sum(∇_η_log_density(ℓ, y, h, θ)), η)
    @test H ≈ H_fd atol = 1.0e-4

    # Closed-form third derivative matches FD of ∇²
    t3 = ∇³_η_log_density(ℓ, y, η, θ)
    t3_fd = fd_grad(h -> sum(∇²_η_log_density(ℓ, y, h, θ)), η)
    @test t3 ≈ t3_fd atol = 1.0e-4

    # Poisson limit: as size → ∞, the NegBin HESSIAN should approach -μ (Poisson diagonal)
    θ_big = [log(1.0e6)]
    H_big = ∇²_η_log_density(ℓ, y, η, θ_big)
    @test all(isapprox.(H_big, -exp.(η); rtol = 1.0e-3))

    # Pointwise sum agrees with total log density
    @test sum(pointwise_log_density(ℓ, y, η, θ)) ≈ lp

    # CDF monotone in y (at a common η/θ)
    y_mono = collect(0:15)
    η_const = fill(0.5, length(y_mono))
    cdfs = pointwise_cdf(ℓ, y_mono, η_const, θ)
    @test all(diff(cdfs) .>= -1.0e-12)
    @test cdfs[end] > 0.5 && cdfs[end] ≤ 1.0
end

@testset "NegativeBinomialLikelihood — ProbitLink (generic path)" begin
    ℓ = NegativeBinomialLikelihood(; link = ProbitLink())
    y = [2, 0, 5, 1]
    η = [0.3, -0.2, 0.8, 0.1]
    θ = [log(3.0)]

    @test isfinite(log_density(ℓ, y, η, θ))

    g = ∇_η_log_density(ℓ, y, η, θ)
    g_fd = fd_grad(h -> log_density(ℓ, y, h, θ), η)
    @test g ≈ g_fd atol = 1.0e-4

    H = ∇²_η_log_density(ℓ, y, η, θ)
    H_fd = fd_grad(h -> sum(∇_η_log_density(ℓ, y, h, θ)), η)
    @test H ≈ H_fd atol = 1.0e-4
end

@testset "NegativeBinomialLikelihood — exposure offset" begin
    E = [0.5, 2.0, 1.0, 3.5]
    ℓ = NegativeBinomialLikelihood(; E = E)
    y = [2, 8, 3, 12]
    η = [0.0, 0.4, -0.3, 0.2]
    θ = [log(1.5)]

    lp = log_density(ℓ, y, η, θ)
    r = exp(θ[1])
    expected = sum(Distributions.logpdf(Distributions.NegativeBinomial(r, r / (r + Ei * exp(ηi))), yi)
                   for (yi, ηi, Ei) in zip(y, η, E))
    @test lp ≈ expected

    g = ∇_η_log_density(ℓ, y, η, θ)
    g_fd = fd_grad(h -> log_density(ℓ, y, h, θ), η)
    @test g ≈ g_fd atol = 1.0e-4
end

@testset "GammaLikelihood — LogLink, default" begin
    ℓ = GammaLikelihood()
    @test nhyperparameters(ℓ) == 1
    @test initial_hyperparameters(ℓ) == [0.0]

    rng = Random.Xoshiro(3)
    y = [0.5, 1.2, 2.7, 0.8, 3.4]
    η = [-0.2, 0.3, 0.9, 0.1, 1.2]
    θ = [log(4.0)]           # φ = 4

    lp = log_density(ℓ, y, η, θ)
    @test isfinite(lp)

    # Agreement with Distributions.Gamma under the shape-scale parameterisation
    φ = exp(θ[1])
    expected = sum(Distributions.logpdf(Distributions.Gamma(φ, exp(ηi) / φ), yi)
                   for (yi, ηi) in zip(y, η))
    @test lp ≈ expected

    g = ∇_η_log_density(ℓ, y, η, θ)
    g_fd = fd_grad(h -> log_density(ℓ, y, h, θ), η)
    @test g ≈ g_fd atol = 1.0e-4

    H = ∇²_η_log_density(ℓ, y, η, θ)
    H_fd = fd_grad(h -> sum(∇_η_log_density(ℓ, y, h, θ)), η)
    @test H ≈ H_fd atol = 1.0e-4

    # ∂³ = -∂² under log link
    t3 = ∇³_η_log_density(ℓ, y, η, θ)
    @test t3 ≈ -H

    # Pointwise sum agrees with total
    @test sum(pointwise_log_density(ℓ, y, η, θ)) ≈ lp

    # Negative y → log density -Inf
    y_bad = [1.0, -0.1, 0.5]
    η_bad = [0.0, 0.0, 0.0]
    @test log_density(ℓ, y_bad, η_bad, θ) == -Inf
    @test pointwise_log_density(ℓ, y_bad, η_bad, θ)[2] == -Inf

    # CDF monotone
    y_mono = collect(0.1:0.2:3.0)
    η_const = fill(0.5, length(y_mono))
    cdfs = pointwise_cdf(ℓ, y_mono, η_const, θ)
    @test all(diff(cdfs) .>= -1.0e-12)
end

@testset "GammaLikelihood — IdentityLink (generic path)" begin
    ℓ = GammaLikelihood(; link = IdentityLink())
    y = [1.3, 0.7, 2.4]
    η = [1.5, 2.1, 0.9]         # positive → μ = η valid
    θ = [log(2.0)]

    @test isfinite(log_density(ℓ, y, η, θ))

    g = ∇_η_log_density(ℓ, y, η, θ)
    g_fd = fd_grad(h -> log_density(ℓ, y, h, θ), η)
    @test g ≈ g_fd atol = 1.0e-4

    H = ∇²_η_log_density(ℓ, y, η, θ)
    H_fd = fd_grad(h -> sum(∇_η_log_density(ℓ, y, h, θ)), η)
    @test H ≈ H_fd atol = 1.0e-4
end

@testset "NegativeBinomial + Gamma — log_hyperprior scales with prior" begin
    ℓ_nb = NegativeBinomialLikelihood()
    ℓ_g = GammaLikelihood()
    θ = [0.3]
    # With GammaPrecision(1.0, 0.1) and internal θ = log(x):
    # log π(θ) = 1·log(0.1) - loggamma(1) + 1·θ - 0.1·exp(θ)
    expected = log(0.1) - Distributions.loggamma(1.0) + θ[1] - 0.1 * exp(θ[1])
    @test log_hyperprior(ℓ_nb, θ) ≈ expected
    @test log_hyperprior(ℓ_g, θ) ≈ expected
end

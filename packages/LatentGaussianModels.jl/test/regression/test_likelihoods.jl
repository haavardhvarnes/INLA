using LatentGaussianModels: GaussianLikelihood, PoissonLikelihood,
    BinomialLikelihood, NegativeBinomialLikelihood, GammaLikelihood,
    IdentityLink, LogLink, LogitLink,
    log_density, ∇_η_log_density, ∇²_η_log_density,
    ∇³_η_log_density, log_hyperprior, nhyperparameters,
    pointwise_log_density

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

@testset "NegativeBinomialLikelihood — LogLink" begin
    rng = Random.Xoshiro(2)
    y = [3, 0, 5, 12, 1, 8, 2, 4]
    η = [0.1, -0.5, 0.8, 1.5, 0.0, 1.2, 0.3, 0.6]

    @testset "no exposure" begin
        ℓ = NegativeBinomialLikelihood()
        @test nhyperparameters(ℓ) == 1
        for θ_val in (-0.7, 0.0, 1.3)
            θ = [θ_val]
            n_param = exp(θ_val)

            lp = log_density(ℓ, y, η, θ)
            @test isfinite(lp)
            # Sum-of-pointwise == joint log-density.
            @test sum(pointwise_log_density(ℓ, y, η, θ)) ≈ lp

            # Closed-form gradient and Hessian against finite differences.
            g = ∇_η_log_density(ℓ, y, η, θ)
            g_fd = fd_grad(h -> log_density(ℓ, y, h, θ), η)
            @test g ≈ g_fd atol = 1.0e-4

            H = ∇²_η_log_density(ℓ, y, η, θ)
            # Each ∂g_i/∂η_i, on the diagonal:
            H_fd = fd_grad(h -> sum(∇_η_log_density(ℓ, y, h, θ)), η)
            @test H ≈ H_fd atol = 1.0e-4

            # Closed form vs analytic restatement.
            μ = exp.(η)
            @test g ≈ n_param .* (y .- μ) ./ (n_param .+ μ)
            @test H ≈ -n_param .* μ .* (n_param .+ y) ./ (n_param .+ μ).^2

            # Third derivative against finite differences of H_ii.
            T = ∇³_η_log_density(ℓ, y, η, θ)
            T_fd = fd_grad(h -> sum(∇²_η_log_density(ℓ, y, h, θ)), η)
            @test T ≈ T_fd atol = 1.0e-4
        end
    end

    @testset "with exposure" begin
        E = [1.0, 0.5, 2.0, 1.5, 0.8, 1.2, 0.9, 1.7]
        ℓ = NegativeBinomialLikelihood(; E = E)
        θ = [0.5]
        g = ∇_η_log_density(ℓ, y, η, θ)
        g_fd = fd_grad(h -> log_density(ℓ, y, h, θ), η)
        @test g ≈ g_fd atol = 1.0e-4
    end

    @testset "Poisson limit (n → ∞)" begin
        # As n → ∞ the NegBin variance μ + μ²/n → μ, recovering Poisson.
        ℓ_nb = NegativeBinomialLikelihood()
        ℓ_p  = PoissonLikelihood()
        θ_large = [log(1.0e6)]
        @test ∇_η_log_density(ℓ_nb, y, η, θ_large) ≈
              ∇_η_log_density(ℓ_p, y, η, Float64[]) atol = 1.0e-3
        @test ∇²_η_log_density(ℓ_nb, y, η, θ_large) ≈
              ∇²_η_log_density(ℓ_p, y, η, Float64[]) atol = 1.0e-3
    end

    @testset "log_hyperprior wired to GammaPrecision(1, 0.1)" begin
        ℓ = NegativeBinomialLikelihood()
        θ = [0.5]
        # GammaPrecision(a=1, b=0.1) → log π(θ) = a log b - logΓ(a) + a θ - b exp(θ)
        expected = log(0.1) - 0.0 + 0.5 - 0.1 * exp(0.5)
        @test log_hyperprior(ℓ, θ) ≈ expected
    end
end

@testset "GammaLikelihood — LogLink" begin
    rng = Random.Xoshiro(3)
    y = [0.42, 1.8, 0.95, 3.2, 0.6, 2.1, 1.4, 0.8]
    η = [0.0, 0.7, 0.1, 1.2, -0.3, 0.9, 0.5, 0.0]

    @testset "closed-form derivatives" begin
        ℓ = GammaLikelihood()
        @test nhyperparameters(ℓ) == 1
        for θ_val in (-0.5, 0.0, 1.0)
            θ = [θ_val]
            φ = exp(θ_val)

            lp = log_density(ℓ, y, η, θ)
            @test isfinite(lp)
            @test sum(pointwise_log_density(ℓ, y, η, θ)) ≈ lp

            g = ∇_η_log_density(ℓ, y, η, θ)
            g_fd = fd_grad(h -> log_density(ℓ, y, h, θ), η)
            @test g ≈ g_fd atol = 1.0e-4

            μ = exp.(η)
            @test g ≈ φ .* (y ./ μ .- 1)

            H = ∇²_η_log_density(ℓ, y, η, θ)
            H_fd = fd_grad(h -> sum(∇_η_log_density(ℓ, y, h, θ)), η)
            @test H ≈ H_fd atol = 1.0e-4
            @test H ≈ -φ .* y ./ μ

            T = ∇³_η_log_density(ℓ, y, η, θ)
            T_fd = fd_grad(h -> sum(∇²_η_log_density(ℓ, y, h, θ)), η)
            @test T ≈ T_fd atol = 1.0e-4
        end
    end

    @testset "y = 0 yields -Inf" begin
        ℓ = GammaLikelihood()
        @test log_density(ℓ, [0.0], [0.0], [0.0]) == -Inf
    end

    @testset "log_hyperprior wired to GammaPrecision(1, 5e-5)" begin
        ℓ = GammaLikelihood()
        θ = [0.5]
        expected = log(5.0e-5) + 0.5 - 5.0e-5 * exp(0.5)
        @test log_hyperprior(ℓ, θ) ≈ expected
    end
end

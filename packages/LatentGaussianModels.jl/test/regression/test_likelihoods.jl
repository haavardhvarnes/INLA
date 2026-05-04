using LatentGaussianModels: GaussianLikelihood, PoissonLikelihood,
                            BinomialLikelihood, NegativeBinomialLikelihood, GammaLikelihood,
                            BetaLikelihood, BetaBinomialLikelihood,
                            StudentTLikelihood, SkewNormalLikelihood,
                            IdentityLink, LogLink, LogitLink,
                            log_density, ∇_η_log_density, ∇²_η_log_density,
                            ∇³_η_log_density, log_hyperprior, nhyperparameters,
                            pointwise_log_density

# Finite-difference sanity check on ∇_η and ∇²_η.
function fd_grad(f, η, h=1.0e-6)
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
    @test ∇_η_log_density(ℓ, y, η, θ)≈g_fd atol=1.0e-4
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
    @test g≈g_fd atol=1.0e-4

    H = ∇²_η_log_density(ℓ, y, η, θ)
    H_fd = fd_grad(h -> sum(∇_η_log_density(ℓ, y, h, θ)), η)
    # H_fd is row-sum of full Jacobian of g; diagonal extraction below:
    # Each g_i depends only on η_i, so Jacobian is diagonal and H_fd ≈ H.
    @test H≈H_fd atol=1.0e-4
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
    @test g≈g_fd atol=1.0e-4

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
            @test g≈g_fd atol=1.0e-4

            H = ∇²_η_log_density(ℓ, y, η, θ)
            # Each ∂g_i/∂η_i, on the diagonal:
            H_fd = fd_grad(h -> sum(∇_η_log_density(ℓ, y, h, θ)), η)
            @test H≈H_fd atol=1.0e-4

            # Closed form vs analytic restatement.
            μ = exp.(η)
            @test g ≈ n_param .* (y .- μ) ./ (n_param .+ μ)
            @test H ≈ -n_param .* μ .* (n_param .+ y) ./ (n_param .+ μ) .^ 2

            # Third derivative against finite differences of H_ii.
            T = ∇³_η_log_density(ℓ, y, η, θ)
            T_fd = fd_grad(h -> sum(∇²_η_log_density(ℓ, y, h, θ)), η)
            @test T≈T_fd atol=1.0e-4
        end
    end

    @testset "with exposure" begin
        E = [1.0, 0.5, 2.0, 1.5, 0.8, 1.2, 0.9, 1.7]
        ℓ = NegativeBinomialLikelihood(; E=E)
        θ = [0.5]
        g = ∇_η_log_density(ℓ, y, η, θ)
        g_fd = fd_grad(h -> log_density(ℓ, y, h, θ), η)
        @test g≈g_fd atol=1.0e-4
    end

    @testset "Poisson limit (n → ∞)" begin
        # As n → ∞ the NegBin variance μ + μ²/n → μ, recovering Poisson.
        ℓ_nb = NegativeBinomialLikelihood()
        ℓ_p = PoissonLikelihood()
        θ_large = [log(1.0e6)]
        @test ∇_η_log_density(ℓ_nb, y, η, θ_large)≈
        ∇_η_log_density(ℓ_p, y, η, Float64[]) atol=1.0e-3
        @test ∇²_η_log_density(ℓ_nb, y, η, θ_large)≈
        ∇²_η_log_density(ℓ_p, y, η, Float64[]) atol=1.0e-3
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
            @test g≈g_fd atol=1.0e-4

            μ = exp.(η)
            @test g ≈ φ .* (y ./ μ .- 1)

            H = ∇²_η_log_density(ℓ, y, η, θ)
            H_fd = fd_grad(h -> sum(∇_η_log_density(ℓ, y, h, θ)), η)
            @test H≈H_fd atol=1.0e-4
            @test H ≈ -φ .* y ./ μ

            T = ∇³_η_log_density(ℓ, y, η, θ)
            T_fd = fd_grad(h -> sum(∇²_η_log_density(ℓ, y, h, θ)), η)
            @test T≈T_fd atol=1.0e-4
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

@testset "BetaLikelihood — LogitLink" begin
    rng = Random.Xoshiro(4)
    y = [0.10, 0.45, 0.80, 0.30, 0.65, 0.95, 0.20, 0.55]
    η = [-1.5, 0.0, 1.2, -0.4, 0.8, 1.7, -1.0, 0.2]

    @testset "closed-form derivatives" begin
        ℓ = BetaLikelihood()
        @test nhyperparameters(ℓ) == 1
        for θ_val in (-0.3, 0.0, 1.5)
            θ = [θ_val]

            lp = log_density(ℓ, y, η, θ)
            @test isfinite(lp)
            @test sum(pointwise_log_density(ℓ, y, η, θ)) ≈ lp

            g = ∇_η_log_density(ℓ, y, η, θ)
            g_fd = fd_grad(h -> log_density(ℓ, y, h, θ), η)
            @test g≈g_fd atol=1.0e-4

            H = ∇²_η_log_density(ℓ, y, η, θ)
            H_fd = fd_grad(h -> sum(∇_η_log_density(ℓ, y, h, θ)), η)
            @test H≈H_fd atol=1.0e-4

            T = ∇³_η_log_density(ℓ, y, η, θ)
            T_fd = fd_grad(h -> sum(∇²_η_log_density(ℓ, y, h, θ)), η)
            @test T≈T_fd atol=1.0e-3
        end
    end

    @testset "boundary y ∈ {0, 1} yields -Inf" begin
        ℓ = BetaLikelihood()
        @test log_density(ℓ, [0.0], [0.0], [0.0]) == -Inf
        @test log_density(ℓ, [1.0], [0.0], [0.0]) == -Inf
    end

    @testset "non-LogitLink rejected" begin
        @test_throws ArgumentError BetaLikelihood(link=LogLink())
        @test_throws ArgumentError BetaLikelihood(link=IdentityLink())
    end

    @testset "log_hyperprior wired to GammaPrecision(1, 0.01)" begin
        ℓ = BetaLikelihood()
        θ = [0.5]
        # GammaPrecision(a=1, b=0.01) → a log b - logΓ(a) + a θ - b·exp(θ)
        expected = log(0.01) - 0.0 + 0.5 - 0.01 * exp(0.5)
        @test log_hyperprior(ℓ, θ) ≈ expected
    end
end

@testset "BetaBinomialLikelihood — LogitLink" begin
    n_trials = [10, 20, 5, 15, 8, 12, 25, 6]
    y = [3, 14, 1, 9, 5, 4, 18, 2]
    η = [-0.5, 0.6, -1.2, 0.8, 0.2, -0.4, 0.9, -1.0]

    @testset "closed-form derivatives" begin
        ℓ = BetaBinomialLikelihood(n_trials)
        @test nhyperparameters(ℓ) == 1
        for θ_val in (-1.5, 0.0, 1.5)
            θ = [θ_val]

            lp = log_density(ℓ, y, η, θ)
            @test isfinite(lp)
            @test sum(pointwise_log_density(ℓ, y, η, θ)) ≈ lp

            g = ∇_η_log_density(ℓ, y, η, θ)
            g_fd = fd_grad(h -> log_density(ℓ, y, h, θ), η)
            @test g≈g_fd atol=1.0e-4

            H = ∇²_η_log_density(ℓ, y, η, θ)
            H_fd = fd_grad(h -> sum(∇_η_log_density(ℓ, y, h, θ)), η)
            @test H≈H_fd atol=1.0e-4

            T = ∇³_η_log_density(ℓ, y, η, θ)
            T_fd = fd_grad(h -> sum(∇²_η_log_density(ℓ, y, h, θ)), η)
            @test T≈T_fd atol=1.0e-3
        end
    end

    @testset "Binomial limit (ρ → 0, s → ∞)" begin
        # As ρ → 0 (θ → +∞, s → ∞) the BetaBinomial collapses to Binomial.
        # Pick a moderately large s; gradient magnitudes will agree.
        ℓ_bb = BetaBinomialLikelihood(n_trials)
        ℓ_b = BinomialLikelihood(n_trials)
        θ_small_ρ = [-log(1.0e6)]   # s = 1e6
        @test ∇_η_log_density(ℓ_bb, y, η, θ_small_ρ)≈
        ∇_η_log_density(ℓ_b, y, η, Float64[]) atol=1.0e-3
        @test ∇²_η_log_density(ℓ_bb, y, η, θ_small_ρ)≈
        ∇²_η_log_density(ℓ_b, y, η, Float64[]) atol=1.0e-3
    end

    @testset "non-LogitLink rejected" begin
        @test_throws ArgumentError BetaBinomialLikelihood(n_trials, link=LogLink())
        @test_throws ArgumentError BetaBinomialLikelihood(n_trials, link=IdentityLink())
    end

    @testset "log_hyperprior wired to GaussianPrior(0, √2)" begin
        ℓ = BetaBinomialLikelihood(n_trials)
        θ = [0.3]
        # Gaussian(0, √2): -½ log(2π) - log(√2) - ½ (0.3/√2)²
        σ = sqrt(2.0)
        expected = -0.5 * log(2π) - log(σ) - 0.5 * (0.3 / σ)^2
        @test log_hyperprior(ℓ, θ) ≈ expected
    end
end

@testset "StudentTLikelihood — IdentityLink" begin
    rng = Random.Xoshiro(5)
    y = randn(rng, 10) .+ 0.3
    η = randn(rng, 10) .* 0.4

    @testset "closed-form derivatives" begin
        ℓ = StudentTLikelihood()
        @test nhyperparameters(ℓ) == 2
        for (θ_τ, θ_ν) in ((0.0, 1.5), (-0.5, 2.5), (0.7, 3.0))
            θ = [θ_τ, θ_ν]
            τ = exp(θ_τ)
            ν = exp(θ_ν) + 2

            lp = log_density(ℓ, y, η, θ)
            @test isfinite(lp)
            @test sum(pointwise_log_density(ℓ, y, η, θ)) ≈ lp

            g = ∇_η_log_density(ℓ, y, η, θ)
            g_fd = fd_grad(h -> log_density(ℓ, y, h, θ), η)
            @test g≈g_fd atol=1.0e-4

            # Closed-form vs analytic restatement of the score.
            r = y .- η
            @test g ≈ (ν + 1) .* τ .* r ./ (ν .+ τ .* r .^ 2)

            H = ∇²_η_log_density(ℓ, y, η, θ)
            H_fd = fd_grad(h -> sum(∇_η_log_density(ℓ, y, h, θ)), η)
            @test H≈H_fd atol=1.0e-4

            denom = ν .+ τ .* r .^ 2
            @test H ≈ (ν + 1) .* τ .* (τ .* r .^ 2 .- ν) ./ denom .^ 2

            T = ∇³_η_log_density(ℓ, y, η, θ)
            T_fd = fd_grad(h -> sum(∇²_η_log_density(ℓ, y, h, θ)), η)
            @test T≈T_fd atol=1.0e-3
        end
    end

    @testset "Gaussian limit (ν → ∞)" begin
        # As ν → ∞ the Student-t collapses to N(η, 1/τ); gradients
        # converge to τ · (y − η) and Hessian to −τ.
        ℓ = StudentTLikelihood()
        θ = [0.0, log(1.0e6 - 2)]   # τ = 1, ν = 1e6
        g = ∇_η_log_density(ℓ, y, η, θ)
        @test g≈(y .- η) atol=1.0e-3
        H = ∇²_η_log_density(ℓ, y, η, θ)
        @test all(abs.(H .+ 1) .< 1.0e-3)
    end

    @testset "non-IdentityLink rejected" begin
        @test_throws ArgumentError StudentTLikelihood(link=LogLink())
        @test_throws ArgumentError StudentTLikelihood(link=LogitLink())
    end

    @testset "log_hyperprior wires both blocks" begin
        ℓ = StudentTLikelihood()
        θ = [0.4, 1.8]
        # τ-block: GammaPrecision(1, 1e-4) → log(1e-4) - 0 + 0.4 - 1e-4·exp(0.4)
        # ν-block: Gaussian(2.5, 1) → -½ log(2π) - log(1) - ½(1.8 - 2.5)²
        expected_τ = log(1.0e-4) + 0.4 - 1.0e-4 * exp(0.4)
        expected_ν = -0.5 * log(2π) - 0.5 * (1.8 - 2.5)^2
        @test log_hyperprior(ℓ, θ) ≈ expected_τ + expected_ν
    end
end

@testset "SkewNormalLikelihood — IdentityLink" begin
    rng = Random.Xoshiro(7)
    y = randn(rng, 12) .+ 0.2
    η = randn(rng, 12) .* 0.3

    @testset "closed-form derivatives" begin
        ℓ = SkewNormalLikelihood()
        @test nhyperparameters(ℓ) == 2
        for (θ_τ, θ_γ) in ((0.0, 0.0), (0.5, 0.7), (-0.4, -1.2))
            θ = [θ_τ, θ_γ]

            lp = log_density(ℓ, y, η, θ)
            @test isfinite(lp)
            @test sum(pointwise_log_density(ℓ, y, η, θ)) ≈ lp

            g = ∇_η_log_density(ℓ, y, η, θ)
            g_fd = fd_grad(h -> log_density(ℓ, y, h, θ), η)
            @test g≈g_fd atol=1.0e-4

            H = ∇²_η_log_density(ℓ, y, η, θ)
            H_fd = fd_grad(h -> sum(∇_η_log_density(ℓ, y, h, θ)), η)
            @test H≈H_fd atol=1.0e-4

            T = ∇³_η_log_density(ℓ, y, η, θ)
            T_fd = fd_grad(h -> sum(∇²_η_log_density(ℓ, y, h, θ)), η)
            @test T≈T_fd atol=1.0e-3
        end
    end

    @testset "Gaussian limit (γ → 0)" begin
        # At θ[2] = 0, γ = 0 and the skew-normal collapses to N(η, 1/τ).
        ℓ = SkewNormalLikelihood()
        for θ_τ in (-0.5, 0.0, 0.7)
            τ = exp(θ_τ)
            θ = [θ_τ, 0.0]
            lp_sn = log_density(ℓ, y, η, θ)
            n = length(y)
            lp_gauss = -n / 2 * log(2π) + n / 2 * θ_τ - 0.5 * τ * sum((y .- η) .^ 2)
            @test lp_sn ≈ lp_gauss

            g = ∇_η_log_density(ℓ, y, η, θ)
            @test g ≈ τ .* (y .- η)
            H = ∇²_η_log_density(ℓ, y, η, θ)
            @test all(abs.(H .+ τ) .< 1.0e-12)
        end
    end

    @testset "non-IdentityLink rejected" begin
        @test_throws ArgumentError SkewNormalLikelihood(link=LogLink())
        @test_throws ArgumentError SkewNormalLikelihood(link=LogitLink())
    end

    @testset "log_hyperprior wires both blocks" begin
        ℓ = SkewNormalLikelihood()
        θ = [0.6, -0.3]
        # τ-block: GammaPrecision(1, 5e-5) → log(5e-5) - 0 + 0.6 - 5e-5·exp(0.6)
        # γ-block: GaussianPrior(0, 1) → -½ log(2π) - 0 - ½(-0.3)²
        expected_τ = log(5.0e-5) + 0.6 - 5.0e-5 * exp(0.6)
        expected_γ = -0.5 * log(2π) - 0.5 * (-0.3)^2
        @test log_hyperprior(ℓ, θ) ≈ expected_τ + expected_γ
    end
end

using LatentGaussianModels: GaussianLikelihood, PoissonLikelihood,
                            BinomialLikelihood, NegativeBinomialLikelihood, GammaLikelihood,
                            BetaLikelihood, BetaBinomialLikelihood,
                            StudentTLikelihood, SkewNormalLikelihood, GEVLikelihood,
                            POMLikelihood,
                            IdentityLink, LogLink, LogitLink,
                            log_density, ∇_η_log_density, ∇²_η_log_density,
                            ∇³_η_log_density, log_hyperprior, nhyperparameters,
                            pointwise_log_density
import SpecialFunctions

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

@testset "GEVLikelihood — IdentityLink" begin
    rng = Random.Xoshiro(42)
    n = 16
    η = randn(rng, n) .* 0.3

    # Sample y inside the support 1 + ξ z > 0 by inverting the GEV CDF.
    function _gev_sample(rng, η, τ, ξ)
        n = length(η)
        σ = 1 / sqrt(τ)
        U = rand(rng, n)
        if abs(ξ) < 1.0e-8
            return η .- σ .* log.(-log.(U))
        else
            return η .+ (σ / ξ) .* ((-log.(U)) .^ (-ξ) .- 1)
        end
    end

    @testset "closed-form derivatives" begin
        ℓ = GEVLikelihood()
        @test nhyperparameters(ℓ) == 2
        # Mix of (log τ, ξ/xi_scale) — moderate ξ in both signs.
        for (θ_τ, θ_ξ) in ((1.0, 0.0), (1.5, 1.0), (0.7, -1.5))
            τ = exp(θ_τ)
            ξ = 0.1 * θ_ξ
            y = _gev_sample(rng, η, τ, ξ)
            θ = [θ_τ, θ_ξ]

            lp = log_density(ℓ, y, η, θ)
            @test isfinite(lp)
            @test sum(pointwise_log_density(ℓ, y, η, θ)) ≈ lp

            g = ∇_η_log_density(ℓ, y, η, θ)
            g_fd = fd_grad(h -> log_density(ℓ, y, h, θ), η)
            @test g≈g_fd atol=1.0e-4

            H = ∇²_η_log_density(ℓ, y, η, θ)
            H_fd = fd_grad(h -> sum(∇_η_log_density(ℓ, y, h, θ)), η)
            @test H≈H_fd atol=1.0e-4

            # ∇³ via finite-difference fallback.
            T = ∇³_η_log_density(ℓ, y, η, θ)
            T_fd = fd_grad(h -> sum(∇²_η_log_density(ℓ, y, h, θ)), η)
            @test T≈T_fd atol=1.0e-3
        end
    end

    @testset "Gumbel limit (ξ → 0)" begin
        # Below |ξ| < _GEV_XI_EPS the implementation switches to the
        # closed-form Gumbel branch. Check it agrees with the direct
        # Gumbel formula at ξ = 0 *exactly* (not just within FD).
        ℓ = GEVLikelihood()
        τ = exp(1.2)
        y = _gev_sample(rng, η, τ, 0.0)
        θ = [1.2, 0.0]
        z = sqrt(τ) .* (y .- η)
        lp_direct = sum(0.5 * log(τ) .- z .- exp.(-z))
        @test log_density(ℓ, y, η, θ) ≈ lp_direct

        g_direct = sqrt(τ) .* (1 .- exp.(-z))
        @test ∇_η_log_density(ℓ, y, η, θ) ≈ g_direct
        H_direct = -τ .* exp.(-z)
        @test ∇²_η_log_density(ℓ, y, η, θ) ≈ H_direct
    end

    @testset "weights scale per-observation precision" begin
        # weights[i] = s_i acts as a multiplier on τ inside √(τ s).
        ℓ_w = GEVLikelihood(weights=fill(2.0, n))
        ℓ_τ2 = GEVLikelihood()
        τ = exp(1.0); ξ = 0.05
        y = _gev_sample(rng, η, τ * 2.0, ξ)
        θ = [1.0, ξ / 0.1]
        θ_doubled = [1.0 + log(2.0), ξ / 0.1]
        # log-density depends on (τ s); equivalently, multiply τ by 2.
        # Up to the per-obs ½ log s constant absorbed identically:
        #   ½ log(τ · 2) = ½ log(τ) + ½ log 2  (matches ½ log(τ s) with s=2).
        lp_w = log_density(ℓ_w, y, η, [1.0, ξ / 0.1])
        lp_d = log_density(ℓ_τ2, y, η, θ_doubled)
        @test lp_w ≈ lp_d
    end

    @testset "support boundary returns -Inf" begin
        ℓ = GEVLikelihood()
        # Strong negative ξ with y far above η pushes 1 + ξ z ≤ 0 for some i.
        bad_y = η .+ 100.0  # huge positive shift
        θ = [0.0, -2.0]    # ξ = -0.2
        @test log_density(ℓ, bad_y, η, θ) == -Inf
    end

    @testset "non-IdentityLink rejected" begin
        @test_throws ArgumentError GEVLikelihood(link=LogLink())
        @test_throws ArgumentError GEVLikelihood(link=LogitLink())
    end

    @testset "constructor validation" begin
        @test_throws ArgumentError GEVLikelihood(xi_scale=0.0)
        @test_throws ArgumentError GEVLikelihood(xi_scale=-0.1)
        @test_throws ArgumentError GEVLikelihood(weights=[1.0, -1.0, 2.0])
    end

    @testset "log_hyperprior wires both blocks" begin
        ℓ = GEVLikelihood()
        θ = [0.4, -0.5]
        # τ-block: GammaPrecision(1, 5e-5) at θ[1]=0.4
        # ξ-block: GaussianPrior(μ=0, σ=2.0) at θ[2]=-0.5 →
        #   −½ log(2π) − log(2.0) − ½ (−0.5/2.0)²
        # (σ=2.0 on internal θ[2] = R-INLA's gaussian(0, prec=25) on
        # user-scale ξ via prec_internal = 25·xi_scale²)
        expected_τ = log(5.0e-5) + 0.4 - 5.0e-5 * exp(0.4)
        expected_ξ = -0.5 * log(2π) - log(2.0) - 0.5 * (-0.5 / 2.0)^2
        @test log_hyperprior(ℓ, θ) ≈ expected_τ + expected_ξ
    end
end

@testset "POMLikelihood — LogitLink" begin
    rng = Random.Xoshiro(11)

    # Sample y_i ∈ {1, …, K} via the cumulative-logit cut-point
    # construction. With α the K−1 cut points and η_i the latent,
    # y_i = 1 + sum_k 1{α_k − η_i < logit(U_i)} for U_i ~ Uniform(0,1).
    function _pom_sample(rng, η, α)
        n = length(η)
        K = length(α) + 1
        y = Vector{Int}(undef, n)
        for i in 1:n
            u = rand(rng)
            t = log(u / (1 - u))
            k = K
            for j in 1:(K - 1)
                if α[j] - η[i] >= t
                    k = j
                    break
                end
            end
            y[i] = k
        end
        return y
    end

    @testset "closed-form derivatives (K = 4)" begin
        K = 4
        n = 24
        η = randn(rng, n) .* 0.4
        ℓ = POMLikelihood(K)
        @test nhyperparameters(ℓ) == K - 1

        # Three (θ_1, log Δ_2, log Δ_3) interior points, with cut
        # points α covering both negative and positive ranges.
        for θ in ([-1.0, log(0.7), log(1.3)],
                  [0.2, log(1.1), log(0.9)],
                  [-0.5, log(0.5), log(2.0)])
            α = [θ[1], θ[1] + exp(θ[2]), θ[1] + exp(θ[2]) + exp(θ[3])]
            # Ensure all four classes are populated by drawing twice as
            # many samples and trimming, then forcing extreme classes
            # if absent.
            y = _pom_sample(rng, η, α)
            for k in 1:K
                if !any(==(k), y)
                    y[k] = k
                end
            end

            lp = log_density(ℓ, y, η, θ)
            @test isfinite(lp)
            @test sum(pointwise_log_density(ℓ, y, η, θ)) ≈ lp

            g = ∇_η_log_density(ℓ, y, η, θ)
            g_fd = fd_grad(h -> log_density(ℓ, y, h, θ), η)
            @test g≈g_fd atol=1.0e-5

            H = ∇²_η_log_density(ℓ, y, η, θ)
            H_fd = fd_grad(h -> sum(∇_η_log_density(ℓ, y, h, θ)), η)
            @test H≈H_fd atol=1.0e-5

            # Log-concavity of cumulative logit: H ≤ 0 elementwise.
            @test all(H .<= 0)

            # ∇³ via finite-difference fallback.
            T = ∇³_η_log_density(ℓ, y, η, θ)
            T_fd = fd_grad(h -> sum(∇²_η_log_density(ℓ, y, h, θ)), η)
            @test T≈T_fd atol=1.0e-3
        end
    end

    @testset "binary special case (K = 2)" begin
        # K = 2 reduces to a logistic regression with cut point α_1.
        n = 20
        η = randn(rng, n) .* 0.5
        ℓ = POMLikelihood(2)
        @test nhyperparameters(ℓ) == 1

        θ = [0.3]
        α1 = θ[1]
        # Sample y ∈ {1, 2} via the cumulative-logit construction.
        y = [(rand(rng) < 1 / (1 + exp(η[i] - α1)) ? 1 : 2) for i in 1:n]

        # log p(y_i = 1) = log F(α_1 − η_i) = −log1p(exp(η_i − α_1))
        # log p(y_i = 2) = log(1 − F(α_1 − η_i)) = −log1p(exp(α_1 − η_i))
        lp_direct = sum(y[i] == 1 ?
                        -log1p(exp(η[i] - α1)) :
                        -log1p(exp(α1 - η[i])) for i in 1:n)
        @test log_density(ℓ, y, η, θ) ≈ lp_direct

        # Boundary-only case: every y at class 1 reduces to logistic
        # regression with intercept α_1 and slope -1 on η.
        y1 = ones(Int, n)
        g1 = ∇_η_log_density(ℓ, y1, η, θ)
        # ∂η log F(α − η) = -f(α − η)/F(α − η) = -(1 − F(α − η))
        @test g1 ≈ [-(1 - 1 / (1 + exp(η[i] - α1))) for i in 1:n]
    end

    @testset "boundary classes match limiting forms" begin
        # When y is uniformly the lowest class, the gradient is the
        # logistic-CDF tail derivative; when uniformly the highest, the
        # complementary form. This is the closed-form check on the
        # branches in `∇_η_log_density`.
        K = 3
        n = 8
        η = collect(range(-1.0, 1.0, length=n))
        ℓ = POMLikelihood(K)
        θ = [-0.2, log(0.8)]
        α = [θ[1], θ[1] + exp(θ[2])]

        y_low = ones(Int, n)
        g_low = ∇_η_log_density(ℓ, y_low, η, θ)
        g_low_direct = [-(1 - 1 / (1 + exp(η[i] - α[1]))) for i in 1:n]
        @test g_low ≈ g_low_direct

        y_high = fill(K, n)
        g_high = ∇_η_log_density(ℓ, y_high, η, θ)
        g_high_direct = [1 / (1 + exp(η[i] - α[end])) for i in 1:n]
        @test g_high ≈ g_high_direct
    end

    @testset "out-of-range y rejected" begin
        ℓ = POMLikelihood(3)
        η = zeros(4)
        θ = [0.0, 0.0]
        @test_throws ArgumentError log_density(ℓ, [0, 1, 2, 3], η, θ)
        @test_throws ArgumentError log_density(ℓ, [1, 2, 3, 4], η, θ)
    end

    @testset "non-LogitLink rejected" begin
        @test_throws ArgumentError POMLikelihood(3, link=IdentityLink())
        @test_throws ArgumentError POMLikelihood(3, link=LogLink())
    end

    @testset "constructor validation" begin
        @test_throws ArgumentError POMLikelihood(1)
        @test_throws ArgumentError POMLikelihood(0)
        @test_throws ArgumentError POMLikelihood(-2)
    end

    @testset "log_hyperprior — Dirichlet on cut-point class probabilities" begin
        # R-INLA's pom prior is Dirichlet(γ, …, γ) on the implied
        # class probabilities π_k(α) = F(α_k) − F(α_{k−1}) at η = 0,
        # pushed back to θ via α and the chain π = π(α(θ)).
        K = 4
        γ = 3.0
        ℓ = POMLikelihood(K; dirichlet_concentration=γ)
        θ = [0.4, log(0.7), log(1.3)]

        # Cut points: α[1] = θ[1], α[k] = α[k−1] + exp(θ[k]).
        α = [θ[1], θ[1] + exp(θ[2]), θ[1] + exp(θ[2]) + exp(θ[3])]
        # F(α_k) at η = 0.
        sig(t) = 1 / (1 + exp(-t))
        g = sig.(α)
        # π_k = F(α_k) − F(α_{k−1}); π_K = 1 − F(α_{K−1}).
        π = [g[1], g[2] - g[1], g[3] - g[2], 1 - g[3]]
        # f(α_k) = F(α_k) (1 − F(α_k)) is the logistic pdf.
        log_jac_α = sum(log(gk * (1 - gk)) for gk in g)
        log_jac_θ = θ[2] + θ[3]
        log_dir = SpecialFunctions.loggamma(K * γ) -
                  K * SpecialFunctions.loggamma(γ) +
                  (γ - 1) * sum(log.(π))
        expected = log_dir + log_jac_α + log_jac_θ
        @test log_hyperprior(ℓ, θ) ≈ expected

        # Symmetric cut points (α = (-Δ, 0, Δ)) → π = (p, ½−p, ½−p, p)
        # with p = F(−Δ); double-check the formula at a tidy value.
        Δ = 1.0
        θ_sym = [-Δ, log(Δ), log(Δ)]
        α_sym = [-Δ, 0.0, Δ]
        g_sym = sig.(α_sym)
        π_sym = [g_sym[1], g_sym[2] - g_sym[1],
            g_sym[3] - g_sym[2], 1 - g_sym[3]]
        log_jac_α_sym = sum(log(gk * (1 - gk)) for gk in g_sym)
        log_jac_θ_sym = θ_sym[2] + θ_sym[3]
        log_dir_sym = SpecialFunctions.loggamma(K * γ) -
                      K * SpecialFunctions.loggamma(γ) +
                      (γ - 1) * sum(log.(π_sym))
        @test log_hyperprior(ℓ, θ_sym) ≈
              log_dir_sym + log_jac_α_sym + log_jac_θ_sym
    end
end

using LatentGaussianModels: GaussianLikelihood, PoissonLikelihood,
    BinomialLikelihood, IdentityLink, LogLink, LogitLink,
    log_density, ∇_η_log_density, ∇²_η_log_density

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

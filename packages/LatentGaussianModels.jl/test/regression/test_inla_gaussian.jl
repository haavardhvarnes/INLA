using LatentGaussianModels: GaussianLikelihood, Intercept, LatentGaussianModel,
    INLA, Grid, GaussHermite, CCD, fit, inla, empirical_bayes
using SparseArrays
using Random
using Statistics: mean

@testset "INLA — Gaussian + Intercept (Grid)" begin
    rng = Random.Xoshiro(20260423)
    n = 200
    α_true = 1.0
    σ = 0.5
    τ_true = 1 / σ^2
    y = α_true .+ σ .* randn(rng, n)

    c = Intercept()
    A = sparse(ones(n, 1))
    ℓ = GaussianLikelihood()
    model = LatentGaussianModel(ℓ, (c,), A)

    res = inla(model, y; int_strategy = :grid)
    @test length(res.θ_points) == 5           # Grid(5) on 1D θ
    @test isapprox(sum(res.θ_weights), 1.0; atol = 1.0e-12)

    # Posterior mean of α should match EB/MLE to close tolerance for Gaussian.
    @test isapprox(res.x_mean[1], mean(y); rtol = 1.0e-2)
    # Variance around the posterior mean should be positive.
    @test res.x_var[1] > 0
    # θ̂ ≈ log τ_true on internal scale.
    @test isapprox(exp(res.θ̂[1]), τ_true; rtol = 0.2)
    # Σθ > 0.
    @test res.Σθ[1, 1] > 0
    # Log-marginal is finite.
    @test isfinite(res.log_marginal)
end

@testset "INLA — auto picks Grid for dim(θ) ≤ 2" begin
    rng = Random.Xoshiro(42)
    n = 100
    y = randn(rng, n)

    c = Intercept()
    A = sparse(ones(n, 1))
    ℓ = GaussianLikelihood()
    model = LatentGaussianModel(ℓ, (c,), A)

    res = inla(model, y)   # :auto → Grid for 1D θ
    @test length(res.θ_points) == 5
    @test isfinite(res.log_marginal)
end

@testset "INLA — GaussHermite scheme" begin
    rng = Random.Xoshiro(7)
    n = 150
    y = 0.5 .+ 0.8 .* randn(rng, n)

    c = Intercept()
    A = sparse(ones(n, 1))
    ℓ = GaussianLikelihood()
    model = LatentGaussianModel(ℓ, (c,), A)

    res_gh = inla(model, y; int_strategy = GaussHermite(n_per_dim = 7))
    res_gr = inla(model, y; int_strategy = Grid(n_per_dim = 21, span = 4.0))

    # Posterior mean of the intercept should match across schemes.
    @test isapprox(res_gh.x_mean[1], res_gr.x_mean[1]; rtol = 1.0e-3)
    @test isapprox(res_gh.θ_mean[1], res_gr.θ_mean[1]; rtol = 1.0e-2)
end

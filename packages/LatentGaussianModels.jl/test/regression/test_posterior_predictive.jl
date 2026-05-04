using LatentGaussianModels
using LatentGaussianModels: GaussianLikelihood, PoissonLikelihood,
                            Intercept, IID, FixedEffects,
                            LatentGaussianModel, inla, posterior_predictive,
                            LinearProjector, IdentityMapping,
                            n_latent
using Distributions: Poisson
using SparseArrays
using LinearAlgebra: I
using Random
using Statistics: mean

@testset "posterior_predictive — shape + statistics on Gaussian + IID" begin
    rng = Random.Xoshiro(20260504)
    n = 30
    y = 0.4 .+ 0.5 .* randn(rng, n)
    ℓ = GaussianLikelihood()
    A = sparse([ones(n) Matrix{Float64}(I, n, n)])
    model = LatentGaussianModel(ℓ, (Intercept(), IID(n)), A)
    res = inla(model, y; int_strategy=:grid)

    S = 800
    draws = posterior_predictive(rng, res, model, A; n_samples=S)
    @test size(draws.x) == (n_latent(model), S)
    @test size(draws.θ) == (2, S)              # GaussianLikelihood τ + IID log-prec
    @test size(draws.η) == (n, S)

    # MC mean of η should match A * x_mean within MC error.
    sample_η_mean = vec(mean(draws.η; dims=2))
    analytic_η_mean = A * res.x_mean
    # η_i = α + u_i, so sd(η_i) ≤ sd(α) + sd(u_i) by Cauchy-Schwarz.
    α_sd = sqrt(res.x_var[1])
    u_max_sd = sqrt(maximum(res.x_var[2:end]))
    band = 5 * (α_sd + u_max_sd) / sqrt(S) + 1.0e-6
    @test all(abs.(sample_η_mean .- analytic_η_mean) .< band)
end

@testset "posterior_predictive — matrix wraps to LinearProjector" begin
    rng = Random.Xoshiro(0)
    n = 20
    y = randn(rng, n)
    ℓ = GaussianLikelihood()
    A = sparse(ones(n, 1))
    model = LatentGaussianModel(ℓ, (Intercept(),), A)
    res = inla(model, y; int_strategy=:grid)

    # Pass an AbstractMatrix instead of a mapping — should accept and
    # produce same-shape output.
    A_new = ones(5, 1)            # 5 new observations on same intercept
    draws = posterior_predictive(rng, res, model, A_new; n_samples=100)
    @test size(draws.η) == (5, 100)
    # All rows of η should be equal across rows for a constant design
    # (every new observation reads the same intercept).
    @test all(draws.η[1, :] .≈ draws.η[5, :])
end

@testset "posterior_predictive — IdentityMapping" begin
    rng = Random.Xoshiro(20260504)
    n = 25
    y = randn(rng, n)
    ℓ = GaussianLikelihood()
    A = sparse(Matrix{Float64}(I, n, n))
    model = LatentGaussianModel(ℓ, (IID(n),), A)
    res = inla(model, y; int_strategy=:grid)

    mapping = IdentityMapping(n)
    draws = posterior_predictive(rng, res, model, mapping; n_samples=100)
    @test size(draws.η) == (n, 100)
    # IdentityMapping: η == x exactly.
    @test draws.η == draws.x
end

@testset "posterior_predictive — n_hyperparameters == 0 fast path" begin
    # Poisson + FixedEffects: dim(θ) = 0. Posterior predictive should
    # still produce η samples with a 0-row θ matrix.
    rng = Random.Xoshiro(7)
    n = 60
    X = [ones(n) randn(rng, n)]
    β_true = [0.3, -0.5]
    y = [rand(rng, Poisson(exp(X[i, :] ⋅ β_true))) for i in 1:n]
    ℓ = PoissonLikelihood()
    A = sparse(X)
    model = LatentGaussianModel(ℓ, (FixedEffects(2),), A)
    res = inla(model, y)

    # New covariate rows: original X works as A_new for the predictive.
    draws = posterior_predictive(rng, res, model, A; n_samples=200)
    @test size(draws.θ) == (0, 200)
    @test size(draws.η) == (n, 200)
    sample_η_mean = vec(mean(draws.η; dims=2))
    @test all(isapprox.(sample_η_mean, A * res.x_mean; atol=0.1))
end

@testset "posterior_predictive — argument validation" begin
    rng = Random.Xoshiro(0)
    n = 30
    y = randn(rng, n)
    ℓ = GaussianLikelihood()
    A = sparse(ones(n, 1))
    model = LatentGaussianModel(ℓ, (Intercept(),), A)
    res = inla(model, y; int_strategy=:grid)

    # n_samples = 0 throws.
    @test_throws ArgumentError posterior_predictive(rng, res, model, A;
        n_samples=0)
    # mapping with wrong ncols throws.
    A_bad = ones(5, 7)            # 7 cols, but model.n_x == 1
    @test_throws DimensionMismatch posterior_predictive(rng, res, model, A_bad;
        n_samples=10)
end

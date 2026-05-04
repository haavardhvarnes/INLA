using LatentGaussianModels
using LatentGaussianModels: GaussianLikelihood, PoissonLikelihood,
                            BinomialLikelihood, NegativeBinomialLikelihood,
                            GammaLikelihood,
                            Intercept, IID, FixedEffects,
                            LatentGaussianModel, inla, posterior_predictive_y,
                            sample_y, n_latent
using Distributions: Poisson, Binomial, NegativeBinomial, Gamma
using SparseArrays
using LinearAlgebra: I
using Random
using Statistics: mean, var

@testset "posterior_predictive_y — Gaussian + IID structure + statistics" begin
    rng = Random.Xoshiro(20260504)
    n = 30
    y = 0.4 .+ 0.5 .* randn(rng, n)
    ℓ = GaussianLikelihood()
    A = sparse([ones(n) Matrix{Float64}(I, n, n)])
    model = LatentGaussianModel(ℓ, (Intercept(), IID(n)), A)
    res = inla(model, y; int_strategy=:grid)

    S = 800
    draws = posterior_predictive_y(rng, res, model; n_samples=S)
    @test size(draws.x) == (n_latent(model), S)
    @test size(draws.θ) == (2, S)              # GaussianLikelihood τ + IID log-prec
    @test size(draws.η) == (n, S)
    @test size(draws.y_rep) == (n, S)
    @test eltype(draws.y_rep) == Float64
    @test all(isfinite, draws.y_rep)

    # MC mean of y_rep should track the posterior predictive mean ≈ A * x_mean
    # (Gaussian: E[y_rep] = E[η] = A x_mean). Wider band than η because y_rep
    # adds observation noise on top of η-uncertainty.
    sample_y_mean = vec(mean(draws.y_rep; dims=2))
    analytic_η_mean = A * res.x_mean
    σ_obs = 1 / sqrt(exp(res.θ_mean[1]))
    α_sd = sqrt(res.x_var[1])
    u_max_sd = sqrt(maximum(res.x_var[2:end]))
    band = 5 * (σ_obs + α_sd + u_max_sd) / sqrt(S) + 1.0e-6
    @test all(abs.(sample_y_mean .- analytic_η_mean) .< band)
end

@testset "posterior_predictive_y — Poisson dim(θ) = 0 fast path" begin
    rng = Random.Xoshiro(7)
    n = 60
    X = [ones(n) randn(rng, n)]
    β_true = [0.3, -0.5]
    y = [rand(rng, Poisson(exp(X[i, :] ⋅ β_true))) for i in 1:n]
    ℓ = PoissonLikelihood()
    A = sparse(X)
    model = LatentGaussianModel(ℓ, (FixedEffects(2),), A)
    res = inla(model, y)

    S = 500
    draws = posterior_predictive_y(rng, res, model; n_samples=S)
    @test size(draws.θ) == (0, S)
    @test size(draws.y_rep) == (n, S)
    # All y_rep entries are non-negative integers stored as Float64.
    @test all(>=(0), draws.y_rep)
    @test all(==(0), mod.(draws.y_rep, 1))

    # Sample mean of y_rep tracks E[y] = exp(η) ≈ exp(A * x_mean).
    sample_mean = vec(mean(draws.y_rep; dims=2))
    analytic_mean = exp.(A * res.x_mean)
    # Use sqrt(λ̂) as a per-row noise scale; broad envelope (5σ / √S + 0.05).
    band = 5 .* sqrt.(analytic_mean) ./ sqrt(S) .+ 0.05
    @test all(abs.(sample_mean .- analytic_mean) .< band)
end

@testset "posterior_predictive_y — Binomial reads n_trials from likelihood" begin
    rng = Random.Xoshiro(2)
    n = 40
    X = [ones(n) randn(rng, n)]
    β_true = [0.0, 0.7]
    n_trials = fill(20, n)
    p_true = 1 ./ (1 .+ exp.(.-(X * β_true)))
    y = [rand(rng, Binomial(n_trials[i], p_true[i])) for i in 1:n]
    ℓ = BinomialLikelihood(n_trials)
    A = sparse(X)
    model = LatentGaussianModel(ℓ, (FixedEffects(2),), A)
    res = inla(model, y)

    S = 400
    draws = posterior_predictive_y(rng, res, model; n_samples=S)
    @test size(draws.y_rep) == (n, S)
    # Each y_rep entry must lie in 0:n_trials[i].
    @test all(0 .<= draws.y_rep .<= reshape(n_trials, n, 1))
    # Counts are integer-valued.
    @test all(==(0), mod.(draws.y_rep, 1))
end

@testset "sample_y — direct dispatch on Gaussian / Poisson / Gamma" begin
    rng = Random.Xoshiro(11)
    η = collect(range(-1.0, 1.0; length=20))

    # Gaussian: σ = 1 / √τ, mean = η.
    ℓ_g = GaussianLikelihood()
    y_g = sample_y(rng, ℓ_g, η, [0.0])    # τ = 1
    @test length(y_g) == length(η)
    @test eltype(y_g) == Float64

    # Poisson: λ = exp(η).
    ℓ_p = PoissonLikelihood()
    y_p = sample_y(rng, ℓ_p, η, Float64[])
    @test length(y_p) == length(η)
    @test all(>=(0), y_p)

    # Gamma: μ = exp(η), shape φ = exp(θ[1]).
    ℓ_γ = GammaLikelihood()
    y_γ = sample_y(rng, ℓ_γ, η, [0.0])
    @test length(y_γ) == length(η)
    @test all(>(0), y_γ)
end

@testset "posterior_predictive_y — convenience method without rng" begin
    rng = Random.Xoshiro(0)
    n = 25
    y = randn(rng, n)
    ℓ = GaussianLikelihood()
    A = sparse([ones(n) Matrix{Float64}(I, n, n)])
    model = LatentGaussianModel(ℓ, (Intercept(), IID(n)), A)
    res = inla(model, y; int_strategy=:grid)

    draws = posterior_predictive_y(res, model; n_samples=50)
    @test size(draws.y_rep) == (n, 50)
end

@testset "posterior_predictive_y — argument validation" begin
    rng = Random.Xoshiro(0)
    n = 20
    y = randn(rng, n)
    ℓ = GaussianLikelihood()
    A = sparse([ones(n) Matrix{Float64}(I, n, n)])
    model = LatentGaussianModel(ℓ, (Intercept(), IID(n)), A)
    res = inla(model, y; int_strategy=:grid)

    @test_throws ArgumentError posterior_predictive_y(rng, res, model;
        n_samples=0)
end

@testset "sample_y — unsupported likelihood raises ArgumentError" begin
    # ExponentialLikelihood ships a `log_density` and `pointwise_log_density`
    # but no `sample_y` (censoring patterns make response-scale sampling
    # ambiguous; defer to a future PR).
    rng = Random.Xoshiro(0)
    ℓ = ExponentialLikelihood()
    @test_throws ArgumentError sample_y(rng, ℓ, [0.0, 0.5], Float64[])
end

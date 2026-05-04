using LatentGaussianModels
using LatentGaussianModels: GaussianLikelihood, PoissonLikelihood,
                            Intercept, IID, FixedEffects,
                            LatentGaussianModel, inla, psis_loo, cpo
using Distributions: Poisson
using SparseArrays
using LinearAlgebra: I
using Random
using Statistics: mean
using PSIS

@testset "psis_loo — Gaussian fixed-effects structure + diagnostics" begin
    rng = Random.Xoshiro(20260504)
    n = 60
    X = [ones(n) randn(rng, n)]
    β_true = [0.4, 0.5]
    y = X * β_true .+ 0.3 .* randn(rng, n)
    ℓ = GaussianLikelihood()
    A = sparse(X)
    # Pure fixed-effects model — non-saturated, so LOO importance ratios
    # are well-behaved and pareto_k should be small.
    model = LatentGaussianModel(ℓ, (FixedEffects(2),), A)
    res = inla(model, y)

    S = 1500
    loo = psis_loo(rng, res, model, y; n_samples=S)
    @test isfinite(loo.elpd_loo)
    @test loo.looic ≈ -2 * loo.elpd_loo
    @test length(loo.pointwise_elpd_loo) == n
    @test length(loo.pointwise_p_loo) == n
    @test length(loo.pareto_k) == n
    @test loo.p_loo ≈ sum(loo.pointwise_p_loo)
    @test loo.elpd_loo ≈ sum(loo.pointwise_elpd_loo)
    @test loo.p_loo > 0
    # Non-saturated model: most k̂ should be safely below 0.7.
    @test count(loo.pareto_k .> 0.7) ≤ 3
end

@testset "psis_loo — Gaussian + IID accepts saturated model" begin
    # IID(n) is a fully-saturated random effect: removing observation i
    # changes the posterior of u_i drastically, so LOO importance ratios
    # are high-variance and pareto_k is large for most observations.
    # PSIS-LOO is documented to be unreliable in this regime — this test
    # confirms the function still produces a structurally well-formed
    # result and surfaces the diagnostic to the caller.
    rng = Random.Xoshiro(20260504)
    n = 30
    y = 0.4 .+ 0.5 .* randn(rng, n)
    ℓ = GaussianLikelihood()
    A = sparse([ones(n) Matrix{Float64}(I, n, n)])
    model = LatentGaussianModel(ℓ, (Intercept(), IID(n)), A)
    res = inla(model, y; int_strategy=:grid)

    loo = psis_loo(rng, res, model, y; n_samples=1000)
    @test isfinite(loo.elpd_loo)
    @test length(loo.pareto_k) == n
end

@testset "psis_loo — convenience method without rng" begin
    rng = Random.Xoshiro(0)
    n = 25
    y = randn(rng, n)
    ℓ = GaussianLikelihood()
    A = sparse([ones(n) Matrix{Float64}(I, n, n)])
    model = LatentGaussianModel(ℓ, (Intercept(), IID(n)), A)
    res = inla(model, y; int_strategy=:grid)

    # The default-rng method should run without error.
    loo = psis_loo(res, model, y; n_samples=400)
    @test isfinite(loo.elpd_loo)
    @test length(loo.pareto_k) == n
end

@testset "psis_loo — agrees with cpo within MC error" begin
    rng = Random.Xoshiro(2)
    n = 30
    X = [ones(n) randn(rng, n)]
    β_true = [0.5, -0.3]
    y = [rand(rng, Poisson(exp(X[i, :] ⋅ β_true))) for i in 1:n]
    ℓ = PoissonLikelihood()
    A = sparse(X)
    model = LatentGaussianModel(ℓ, (FixedEffects(2),), A)
    res = inla(model, y)

    S = 2000
    loo = psis_loo(rng, res, model, y; n_samples=S)
    cp = cpo(rng, res, model, y; n_samples=S)
    # PSIS-LOO is a smoothed/stabilised variant of the same quantity
    # `cpo` estimates by harmonic mean. Aggregate scores should agree
    # within MC error on a small well-specified Poisson model.
    diff = loo.elpd_loo - cp.log_pseudo_marginal
    @test abs(diff) < 1.5    # < 1.5 nats absolute on n=30 obs, S=2000
end

@testset "psis_loo — argument validation" begin
    rng = Random.Xoshiro(0)
    n = 20
    y = randn(rng, n)
    ℓ = GaussianLikelihood()
    A = sparse([ones(n) Matrix{Float64}(I, n, n)])
    model = LatentGaussianModel(ℓ, (Intercept(), IID(n)), A)
    res = inla(model, y; int_strategy=:grid)

    @test_throws ArgumentError psis_loo(rng, res, model, y; n_samples=0)
end

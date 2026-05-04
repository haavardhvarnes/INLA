using LatentGaussianModels
using LatentGaussianModels: PoissonLikelihood, GaussianLikelihood,
                            Intercept, IID, FixedEffects,
                            LatentGaussianModel, inla, refine_hyperposterior,
                            n_hyperparameters
using Distributions: Poisson
using SparseArrays
using LinearAlgebra: I
using Random

@testset "refine_hyperposterior — round-trip same n_grid" begin
    # Refining at the same resolution as the original 5×5 grid should
    # reproduce the log-marginal and posterior summaries within IS noise.
    rng = Random.Xoshiro(20260504)
    n = 30
    y = 0.4 .+ 0.5 .* randn(rng, n)
    ℓ = GaussianLikelihood()
    A = sparse([ones(n) Matrix{Float64}(I, n, n)])
    model = LatentGaussianModel(ℓ, (Intercept(), IID(n)), A)
    res = inla(model, y; int_strategy=:grid)

    res2 = refine_hyperposterior(res, model, y;
        n_grid=5, span=3.0, skewness_correction=false)

    @test length(res2.θ_points) == 25                    # 5×5
    @test isapprox(sum(res2.θ_weights), 1.0; atol=1.0e-12)
    @test res2.θ̂ == res.θ̂                                # mode reused
    @test res2.Σθ == res.Σθ                              # Σθ reused
    # Log-marginal and posterior summaries within IS tolerance.
    @test isapprox(res2.log_marginal, res.log_marginal; atol=0.1)
    @test isapprox(res2.x_mean[1], res.x_mean[1]; atol=1.0e-3)
    @test isapprox(res2.θ_mean[1], res.θ_mean[1]; atol=0.05)
end

@testset "refine_hyperposterior — denser grid is consistent" begin
    # Refining 5×5 → 11×11 should improve resolution and stay close to
    # the original on a well-conditioned posterior.
    rng = Random.Xoshiro(20260504)
    n = 30
    y = 0.4 .+ 0.5 .* randn(rng, n)
    ℓ = GaussianLikelihood()
    A = sparse([ones(n) Matrix{Float64}(I, n, n)])
    model = LatentGaussianModel(ℓ, (Intercept(), IID(n)), A)
    res = inla(model, y; int_strategy=:grid)
    res2 = refine_hyperposterior(res, model, y; n_grid=11, span=4.0)

    @test length(res2.θ_points) == 121                   # 11×11
    @test isapprox(sum(res2.θ_weights), 1.0; atol=1.0e-12)
    @test isfinite(res2.log_marginal)
    @test all(res2.x_var .≥ 0)
    # Refined log-marginal should be within ~1 nat of the original on a
    # benign posterior (denser grid catches more probability mass).
    @test isapprox(res2.log_marginal, res.log_marginal; atol=1.0)
    # Posterior mean of the latent stays close.
    @test isapprox(res2.x_mean[1], res.x_mean[1]; atol=5.0e-3)
end

@testset "refine_hyperposterior — skewness_correction toggles" begin
    rng = Random.Xoshiro(20260504)
    n = 30
    y = 0.4 .+ 0.5 .* randn(rng, n)
    ℓ = GaussianLikelihood()
    A = sparse([ones(n) Matrix{Float64}(I, n, n)])
    model = LatentGaussianModel(ℓ, (Intercept(), IID(n)), A)
    res = inla(model, y; int_strategy=:grid)

    res_off = refine_hyperposterior(res, model, y;
        n_grid=11, span=4.0, skewness_correction=false)
    res_on = refine_hyperposterior(res, model, y;
        n_grid=11, span=4.0, skewness_correction=true)
    @test isfinite(res_off.log_marginal)
    @test isfinite(res_on.log_marginal)
    # Both produce sensible posterior summaries.
    @test isapprox(res_off.x_mean[1], res_on.x_mean[1]; atol=5.0e-3)
end

@testset "refine_hyperposterior — n_hyperparameters == 0 throws" begin
    # Poisson + FixedEffects has dim(θ) = 0 — nothing to refine.
    rng = Random.Xoshiro(7)
    n = 40
    X = [ones(n) randn(rng, n)]
    β_true = [0.3, -0.5]
    y = [rand(rng, Poisson(exp(X[i, :] ⋅ β_true))) for i in 1:n]
    ℓ = PoissonLikelihood()
    A = sparse(X)
    model = LatentGaussianModel(ℓ, (FixedEffects(2),), A)
    res = inla(model, y)
    @test n_hyperparameters(model) == 0
    @test_throws ArgumentError refine_hyperposterior(res, model, y; n_grid=11)
end

@testset "refine_hyperposterior — argument validation" begin
    rng = Random.Xoshiro(0)
    n = 30
    y = randn(rng, n)
    ℓ = GaussianLikelihood()
    A = sparse(ones(n, 1))
    model = LatentGaussianModel(ℓ, (Intercept(),), A)
    res = inla(model, y; int_strategy=:grid)

    @test_throws ArgumentError refine_hyperposterior(res, model, y; n_grid=0)
    @test_throws ArgumentError refine_hyperposterior(res, model, y; span=-1.0)
    @test_throws ArgumentError refine_hyperposterior(res, model, y;
        latent_strategy=:foo)
end

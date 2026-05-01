using LatentGaussianModels: GaussianLikelihood, Intercept, IID, LatentGaussianModel,
                            inla, initial_hyperparameters, n_hyperparameters
using LGMTuring: nuts_sample
using LinearAlgebra: I
using MCMCChains: Chains
using Random
using SparseArrays: sparse
using Test

@testset "nuts_sample — Gaussian + Intercept (1-D θ)" begin
    rng = Random.Xoshiro(20260426)
    n = 200
    σ_ε = 0.5
    y = 0.3 .+ σ_ε .* randn(rng, n)
    model = LatentGaussianModel(GaussianLikelihood(), (Intercept(),),
        sparse(ones(n, 1)))

    chain = nuts_sample(model, y, 200;
        n_adapts=100,
        rng=Random.Xoshiro(1),
        progress=false)
    @test chain isa Chains
    @test size(chain, 1) == 200
    @test size(chain, 2) == n_hyperparameters(model) == 1

    # Posterior mean of log-precision should be near log(1/σ_ε²) within
    # a generous band — small n so wide tolerance.
    samples = vec(Array(chain[Symbol("likelihood[1]")]))
    @test abs(sum(samples) / length(samples) - log(1 / σ_ε^2)) < 0.5
end

@testset "nuts_sample — init_θ override and dim mismatch" begin
    rng = Random.Xoshiro(20260426)
    n = 60
    y = randn(rng, n)
    model = LatentGaussianModel(GaussianLikelihood(), (Intercept(),),
        sparse(ones(n, 1)))

    # init_θ wrong length errors loudly.
    @test_throws DimensionMismatch nuts_sample(model, y, 5;
        init_θ=[0.0, 0.0],
        n_adapts=0,
        rng=Random.Xoshiro(1))

    # Argument validation.
    @test_throws ArgumentError nuts_sample(model, y, 0;
        rng=Random.Xoshiro(1))
    @test_throws ArgumentError nuts_sample(model, y, 5;
        target_acceptance=1.5,
        rng=Random.Xoshiro(1))
end

@testset "nuts_sample — init_from_inla uses INLAResult.θ̂" begin
    # Build a 2-D θ model so we exercise both init paths.
    rng = Random.Xoshiro(20260426)
    n = 80
    α_true = 0.2
    σ_ε = 0.5
    σ_x = 0.7
    x_true = σ_x .* randn(rng, n)
    y = α_true .+ x_true .+ σ_ε .* randn(rng, n)

    A = sparse([ones(n) Matrix{Float64}(I, n,n)])
    model = LatentGaussianModel(GaussianLikelihood(), (Intercept(), IID(n)), A)

    inla_fit = inla(model, y; int_strategy=:grid)
    @test n_hyperparameters(model) == 2

    chain = nuts_sample(model, y, 100;
        n_adapts=50,
        init_from_inla=inla_fit,
        rng=Random.Xoshiro(2),
        progress=false)
    @test chain isa Chains
    @test size(chain, 1) == 100
    @test size(chain, 2) == 2
end

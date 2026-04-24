using LatentGaussianModels: NegativeBinomialLikelihood, GammaLikelihood,
    Intercept, IID, LatentGaussianModel, inla, PCPrecision,
    fixed_effects, hyperparameters
using Distributions: NegativeBinomial, Gamma
using SparseArrays
using LinearAlgebra: I
using Random

@testset "INLA — NegativeBinomial + Intercept + IID (synthetic recovery)" begin
    rng = Random.Xoshiro(20260424)
    n = 150
    α_true = 0.2
    τ_true = 4.0
    size_true = 2.5

    u = randn(rng, n) ./ sqrt(τ_true)
    η_true = α_true .+ u
    y = [rand(rng, NegativeBinomial(size_true, size_true / (size_true + exp(η_true[i]))))
         for i in 1:n]

    c_int = Intercept()
    c_iid = IID(n; hyperprior = PCPrecision(1.0, 0.01))
    A = sparse([ones(n) Matrix{Float64}(I, n, n)])
    ℓ = NegativeBinomialLikelihood()
    model = LatentGaussianModel(ℓ, (c_int, c_iid), A)

    res = inla(model, y; int_strategy = :grid)

    # Wiring: hyperparameter layout [log(size), log(τ_iid)].
    @test length(res.θ̂) == 2
    @test all(isfinite, res.θ̂)

    fe = fixed_effects(model, res)
    @test length(fe) == 1
    # Intercept recovery within 4σ — loose because the per-obs IID soaks
    # most of the residual variation at this n.
    @test abs(fe[1].mean - α_true) < 4 * fe[1].sd

    hp = hyperparameters(model, res)
    @test length(hp) == 2
    @test all(isfinite(r.mean) && r.sd > 0 for r in hp)

    @test isfinite(res.log_marginal)
end

@testset "INLA — Gamma + Intercept + IID (synthetic recovery)" begin
    rng = Random.Xoshiro(20260424)
    n = 150
    α_true = 0.2
    τ_true = 10.0
    φ_true = 5.0

    u = randn(rng, n) ./ sqrt(τ_true)
    η_true = α_true .+ u
    y = [rand(rng, Gamma(φ_true, exp(η_true[i]) / φ_true)) for i in 1:n]

    c_int = Intercept()
    c_iid = IID(n; hyperprior = PCPrecision(1.0, 0.01))
    A = sparse([ones(n) Matrix{Float64}(I, n, n)])
    ℓ = GammaLikelihood()
    model = LatentGaussianModel(ℓ, (c_int, c_iid), A)

    res = inla(model, y; int_strategy = :grid)

    # Wiring: hyperparameter layout [log(φ), log(τ_iid)].
    @test length(res.θ̂) == 2
    @test all(isfinite, res.θ̂)

    fe = fixed_effects(model, res)
    @test length(fe) == 1
    @test abs(fe[1].mean - α_true) < 4 * fe[1].sd

    hp = hyperparameters(model, res)
    @test length(hp) == 2
    @test all(isfinite(r.mean) && r.sd > 0 for r in hp)

    @test isfinite(res.log_marginal)
end

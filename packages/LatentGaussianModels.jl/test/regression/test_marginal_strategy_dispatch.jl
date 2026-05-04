using LatentGaussianModels
using LatentGaussianModels: GaussianLikelihood, PoissonLikelihood,
                            Intercept, IID, LatentGaussianModel, inla,
                            posterior_marginal_x, refine_hyperposterior,
                            AbstractMarginalStrategy, Gaussian,
                            SimplifiedLaplace, INLA
using Distributions: Poisson
using SparseArrays
using LinearAlgebra: I
using Random

# Phase L PR-1 (ADR-026): the symbol-keyed marginal strategy API is
# now a backwards-compat shim over an `AbstractMarginalStrategy` type
# hierarchy. These tests pin the contract: the two forms must produce
# bit-for-bit identical results, and unknown symbols must error loudly.

@testset "AbstractMarginalStrategy — type hierarchy" begin
    @test Gaussian <: AbstractMarginalStrategy
    @test SimplifiedLaplace <: AbstractMarginalStrategy
    @test Gaussian() isa AbstractMarginalStrategy
    @test SimplifiedLaplace() isa AbstractMarginalStrategy
end

@testset "INLA(latent_strategy = …) — symbol vs type parity" begin
    rng = Random.Xoshiro(20260504)
    n = 25
    y = 0.4 .+ randn(rng, n)
    ℓ = GaussianLikelihood()
    A = sparse(reshape(ones(n), n, 1))
    model = LatentGaussianModel(ℓ, (Intercept(),), A)

    # Gaussian (default) — symbol vs type.
    res_sym = inla(model, y; int_strategy=:grid, latent_strategy=:gaussian)
    res_typ = inla(model, y; int_strategy=:grid, latent_strategy=Gaussian())
    @test res_sym.x_mean == res_typ.x_mean
    @test res_sym.x_var == res_typ.x_var
    @test res_sym.θ_mean == res_typ.θ_mean
    @test res_sym.log_marginal == res_typ.log_marginal

    # SimplifiedLaplace — symbol vs type.
    res_sym_sl = inla(model, y; int_strategy=:grid,
        latent_strategy=:simplified_laplace)
    res_typ_sl = inla(model, y; int_strategy=:grid,
        latent_strategy=SimplifiedLaplace())
    @test res_sym_sl.x_mean == res_typ_sl.x_mean
    @test res_sym_sl.x_var == res_typ_sl.x_var
    @test res_sym_sl.θ_mean == res_typ_sl.θ_mean
    @test res_sym_sl.log_marginal == res_typ_sl.log_marginal
end

@testset "posterior_marginal_x(strategy = …) — symbol vs type parity" begin
    # Low-count Poisson so SimplifiedLaplace is non-trivial (skewness ≠ 0).
    rng = Random.Xoshiro(20260504)
    n = 30
    y = rand(rng, Poisson(0.4), n)
    E = fill(1.0, n)
    ℓ = PoissonLikelihood(; E=E)
    A = sparse([ones(n) Matrix{Float64}(I, n, n)])
    model = LatentGaussianModel(ℓ, (Intercept(), IID(n)), A)
    res = inla(model, y; int_strategy=:grid)

    # Gaussian — symbol vs type.
    g_sym = posterior_marginal_x(res, 1; strategy=:gaussian, grid_size=80)
    g_typ = posterior_marginal_x(res, 1; strategy=Gaussian(), grid_size=80)
    @test g_sym.x == g_typ.x
    @test g_sym.pdf == g_typ.pdf

    # SimplifiedLaplace — symbol vs type.
    sl_sym = posterior_marginal_x(res, 1; strategy=:simplified_laplace,
        model=model, y=y, grid_size=80)
    sl_typ = posterior_marginal_x(res, 1; strategy=SimplifiedLaplace(),
        model=model, y=y, grid_size=80)
    @test sl_sym.x == sl_typ.x
    @test sl_sym.pdf == sl_typ.pdf
end

@testset "refine_hyperposterior(latent_strategy = …) — symbol vs type parity" begin
    rng = Random.Xoshiro(20260504)
    n = 25
    y = 0.4 .+ randn(rng, n)
    ℓ = GaussianLikelihood()
    A = sparse(reshape(ones(n), n, 1))
    model = LatentGaussianModel(ℓ, (Intercept(),), A)
    res = inla(model, y; int_strategy=:grid)

    r_sym = refine_hyperposterior(res, model, y;
        n_grid=7, latent_strategy=:gaussian, skewness_correction=false)
    r_typ = refine_hyperposterior(res, model, y;
        n_grid=7, latent_strategy=Gaussian(), skewness_correction=false)
    @test r_sym.x_mean == r_typ.x_mean
    @test r_sym.x_var == r_typ.x_var
    @test r_sym.log_marginal == r_typ.log_marginal
end

@testset "Unknown symbol → ArgumentError" begin
    rng = Random.Xoshiro(20260504)
    n = 20
    y = randn(rng, n)
    ℓ = GaussianLikelihood()
    A = sparse(reshape(ones(n), n, 1))
    model = LatentGaussianModel(ℓ, (Intercept(),), A)

    @test_throws ArgumentError INLA(latent_strategy=:not_a_strategy)
    @test_throws ArgumentError inla(model, y; latent_strategy=:not_a_strategy)
end

@testset "SimplifiedLaplace() requires model + y for posterior_marginal_x" begin
    rng = Random.Xoshiro(20260504)
    n = 20
    y = 0.1 .+ randn(rng, n)
    ℓ = GaussianLikelihood()
    A = sparse(reshape(ones(n), n, 1))
    model = LatentGaussianModel(ℓ, (Intercept(),), A)
    res = inla(model, y; int_strategy=:grid)

    @test_throws ArgumentError posterior_marginal_x(res, 1;
        strategy=SimplifiedLaplace())
    @test_throws ArgumentError posterior_marginal_x(res, 1;
        strategy=:simplified_laplace)
end

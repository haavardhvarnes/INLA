using LatentGaussianModels
using LatentGaussianModels: GaussianLikelihood, PoissonLikelihood,
                            Intercept, IID, Generic0, UserComponent,
                            LatentGaussianModel, inla, log_prior_density,
                            PCPrecision, GammaPrecision,
                            log_normalizing_constant, log_hyperprior,
                            precision_matrix, nhyperparameters,
                            initial_hyperparameters
using GMRFs: NoConstraint, LinearConstraint, constraints, constraint_matrix
using Distributions: Poisson
using SparseArrays
using LinearAlgebra: I
using Random

# Phase L PR-2 (ADR-025): `UserComponent` is the R-INLA `rgeneric`
# equivalent — a one-line callable wrapper around the
# `AbstractLatentComponent` contract. The acceptance criterion is that
# a `UserComponent` whose callable mirrors `Generic0`'s precision and
# log-NC reproduces the full `Generic0` posterior bit-for-bit (or to
# 1e-10) under `inla()`. We cover both proper (full-rank) and
# intrinsic (rank-deficient + sum-to-zero constraint) variants.

# Helper: mimic Generic0's behaviour in a single closure. The
# callable's view of θ is this component's slice — `log τ` here.
function _generic0_callable(R::AbstractMatrix; rd::Integer=0,
        hyperprior=PCPrecision())
    n = size(R, 1)
    Rs = SparseMatrixCSC{Float64, Int}(R)
    return θ -> (; Q = exp(θ[1]) .* Rs,
                   log_prior = log_prior_density(hyperprior, θ[1]),
                   log_normc = -0.5 * (n - rd) * log(2π) +
                               0.5 * (n - rd) * θ[1])
end

@testset "UserComponent — contract" begin
    n = 4
    R = sparse(2.0 * I, n, n)
    c = UserComponent(_generic0_callable(R); n=n, θ0=[0.0])

    @test length(c) == n
    @test nhyperparameters(c) == 1
    @test initial_hyperparameters(c) == [0.0]

    Q = precision_matrix(c, [log(3.0)])
    @test Matrix(Q) ≈ 3.0 * Matrix(R)
    @test log_hyperprior(c, [0.0]) ≈ log_prior_density(PCPrecision(), 0.0)
    @test constraints(c) isa NoConstraint

    # log NC: -½ n log(2π) + ½ n log τ for full-rank R-INLA F_GENERIC0.
    @test log_normalizing_constant(c, [0.5]) ≈ -0.5 * n * log(2π) + 0.5 * n * 0.5
end

@testset "UserComponent — input validation" begin
    n = 3
    Rs = sparse(I, n, n)
    callable = θ -> (; Q = exp(θ[1]) .* Rs)

    @test_throws ArgumentError UserComponent(callable; n=0)
    # Wrong-size Q.
    @test_throws DimensionMismatch UserComponent(callable; n=4, θ0=[0.0])
    # Missing :Q.
    @test_throws ArgumentError UserComponent(_θ -> (; foo=1); n=n, θ0=Float64[])
    # Non-NamedTuple return.
    @test_throws ArgumentError UserComponent(_θ -> 42; n=n, θ0=Float64[])
end

@testset "UserComponent reproduces Generic0 — proper, full rank" begin
    rng = Random.Xoshiro(20260504)
    n = 30

    # Tridiagonal SPD structure matrix: 2 on diagonal, -0.5 on off-diag.
    di = fill(2.0, n)
    od = fill(-0.5, n - 1)
    R = sparse(SymTridiagonal(di, od))

    # Generate data from the corresponding GMRF + Gaussian likelihood.
    y = randn(rng, n) .+ 0.3
    A = sparse(I, n, n)

    g0 = Generic0(R; rankdef=0, hyperprior=PCPrecision())
    uc = UserComponent(_generic0_callable(R; rd=0, hyperprior=PCPrecision());
        n=n, θ0=[0.0])

    model_g0 = LatentGaussianModel(GaussianLikelihood(), (g0,), A)
    model_uc = LatentGaussianModel(GaussianLikelihood(), (uc,), A)

    res_g0 = inla(model_g0, y; int_strategy=:grid)
    res_uc = inla(model_uc, y; int_strategy=:grid)

    @test isapprox(res_g0.x_mean, res_uc.x_mean; atol=1.0e-10, rtol=1.0e-10)
    @test isapprox(res_g0.x_var, res_uc.x_var; atol=1.0e-10, rtol=1.0e-10)
    @test isapprox(res_g0.θ_mean, res_uc.θ_mean; atol=1.0e-10, rtol=1.0e-10)
    @test isapprox(res_g0.log_marginal, res_uc.log_marginal;
        atol=1.0e-10, rtol=1.0e-10)
end

@testset "UserComponent reproduces Generic0 — intrinsic with constraint" begin
    rng = Random.Xoshiro(20260504)
    n = 25

    # RW1 Laplacian: rank n-1, sum-to-zero null space.
    R = sparse(SymTridiagonal(
        vcat(1.0, fill(2.0, n - 2), 1.0),
        fill(-1.0, n - 1)))
    Aeq = ones(1, n)
    e = zeros(1)
    constraint = LinearConstraint(Aeq, e)

    y = 0.2 .+ randn(rng, n)
    A = sparse(I, n, n)

    g0 = Generic0(R; rankdef=1, constraint=constraint,
        hyperprior=GammaPrecision(1.0, 5.0e-5))
    # Mirror the constraint in the UserComponent callable's NamedTuple.
    g0_cb = _generic0_callable(R; rd=1, hyperprior=GammaPrecision(1.0, 5.0e-5))
    uc_callable = θ -> (; Q = g0_cb(θ).Q,
                          log_prior = g0_cb(θ).log_prior,
                          log_normc = g0_cb(θ).log_normc,
                          constraint = constraint)
    uc = UserComponent(uc_callable; n=n, θ0=[0.0])

    @test constraints(uc) isa LinearConstraint
    @test constraint_matrix(constraints(uc)) ≈ Aeq

    model_g0 = LatentGaussianModel(GaussianLikelihood(), (g0,), A)
    model_uc = LatentGaussianModel(GaussianLikelihood(), (uc,), A)

    res_g0 = inla(model_g0, y; int_strategy=:grid)
    res_uc = inla(model_uc, y; int_strategy=:grid)

    @test isapprox(res_g0.x_mean, res_uc.x_mean; atol=1.0e-10, rtol=1.0e-10)
    @test isapprox(res_g0.x_var, res_uc.x_var; atol=1.0e-10, rtol=1.0e-10)
    @test isapprox(res_g0.θ_mean, res_uc.θ_mean; atol=1.0e-10, rtol=1.0e-10)
    @test isapprox(res_g0.log_marginal, res_uc.log_marginal;
        atol=1.0e-10, rtol=1.0e-10)
end

@testset "UserComponent — Poisson + IID composite" begin
    # Stack a UserComponent (mimicking IID) under an Intercept + Poisson
    # likelihood. Exercises the joint pipeline (block-diagonal precision,
    # multi-component log_normalizing_constant sum, hyperprior contributions).
    rng = Random.Xoshiro(20260504)
    n = 20
    y = rand(rng, Poisson(1.5), n)
    E = fill(1.0, n)
    A = sparse([ones(n) Matrix{Float64}(I, n, n)])

    iid_callable = function (θ)
        τ = exp(θ[1])
        return (; Q = sparse(τ * I, n, n),
                  log_prior = log_prior_density(PCPrecision(), θ[1]),
                  log_normc = -0.5 * n * log(2π) + 0.5 * n * θ[1])
    end
    uc_iid = UserComponent(iid_callable; n=n, θ0=[0.0])

    model_uc = LatentGaussianModel(PoissonLikelihood(; E=E),
        (Intercept(), uc_iid), A)
    model_native = LatentGaussianModel(PoissonLikelihood(; E=E),
        (Intercept(), IID(n)), A)

    res_uc = inla(model_uc, y; int_strategy=:grid)
    res_native = inla(model_native, y; int_strategy=:grid)

    @test isapprox(res_uc.x_mean, res_native.x_mean; atol=1.0e-8, rtol=1.0e-8)
    @test isapprox(res_uc.x_var, res_native.x_var; atol=1.0e-8, rtol=1.0e-8)
    @test isapprox(res_uc.log_marginal, res_native.log_marginal;
        atol=1.0e-8, rtol=1.0e-8)
end

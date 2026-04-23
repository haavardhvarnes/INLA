using Test
using GMRFs
using LinearAlgebra
using SparseArrays

@testset "FactorCache: build, solve, logdet" begin
    Q = sparse([2.0 -1 0; -1 2 -1; 0 -1 2])
    cache = FactorCache(Q)
    b = [1.0, 2.0, 3.0]
    x = cache \ b
    @test Q * x ≈ b rtol = 1e-12
    @test logdet(cache) ≈ logdet(Matrix(Q)) rtol = 1e-10
    @test factor(cache) !== nothing
end

@testset "FactorCache: update! preserves symbolic pattern and solves correctly" begin
    # Same sparsity, different values
    Q1 = sparse([4.0 -1 0; -1 4 -1; 0 -1 4])
    Q2 = sparse([3.0 -0.5 0; -0.5 3 -0.5; 0 -0.5 3])
    cache = FactorCache(Q1)
    update!(cache, Q2)
    b = [1.0, 0.5, -1.0]
    x = cache \ b
    @test Q2 * x ≈ b rtol = 1e-12
    @test logdet(cache) ≈ logdet(Matrix(Q2)) rtol = 1e-10
end

@testset "FactorCache: AR1 precision sweep (τ-reuse)" begin
    n = 8
    g = AR1GMRF(n; ρ = 0.5, τ = 1.0)
    cache = FactorCache(precision_matrix(g))
    b = randn(n)
    for τ in (0.5, 1.0, 2.0, 4.0)
        gτ = AR1GMRF(n; ρ = 0.5, τ = τ)
        update!(cache, precision_matrix(gτ))
        x = cache \ b
        @test precision_matrix(gτ) * x ≈ b rtol = 1e-10
    end
end

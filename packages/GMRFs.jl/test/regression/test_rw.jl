using Test
using GMRFs
using LinearAlgebra
using SparseArrays
using Random

@testset "RW1GMRF open structure" begin
    n = 5
    g = RW1GMRF(n; τ = 1.0)
    @test rankdef(g) == 1
    Q = Matrix(precision_matrix(g))
    # Expected:  [1 -1  0  0  0
    #            -1  2 -1  0  0
    #             0 -1  2 -1  0
    #             0  0 -1  2 -1
    #             0  0  0 -1  1]
    expected = [1.0 -1 0 0 0; -1 2 -1 0 0; 0 -1 2 -1 0; 0 0 -1 2 -1; 0 0 0 -1 1]
    @test Q ≈ expected
end

@testset "RW1GMRF cyclic structure" begin
    n = 4
    g = RW1GMRF(n; τ = 1.0, cyclic = true)
    Q = Matrix(precision_matrix(g))
    # Cyclic RW1: diag all 2, -1 on each cyclic neighbor
    expected = [2.0 -1 0 -1; -1 2 -1 0; 0 -1 2 -1; -1 0 -1 2]
    @test Q ≈ expected
    # Null space is still span{1}
    @test abs(sum(Q * ones(n))) < 1e-12
end

@testset "RW1GMRF sample sums to zero" begin
    rng = MersenneTwister(7)
    g = RW1GMRF(10; τ = 1.0)
    for _ in 1:20
        x = rand(rng, g)
        @test abs(sum(x)) < 1e-9
    end
end

@testset "RW2GMRF open" begin
    n = 5
    g = RW2GMRF(n; τ = 1.0)
    @test rankdef(g) == 2
    Q = Matrix(precision_matrix(g))
    # RW2 open 5-node R = D'D with D the (n-2)×n second diff.
    # Row of D at index i corresponds to interior node i+1, pattern
    # (1, -2, 1). Precision pentadiagonal.
    D = zeros(n - 2, n)
    for (row, k) in enumerate(2:(n - 1))
        D[row, k - 1] = 1
        D[row, k] = -2
        D[row, k + 1] = 1
    end
    expected = D' * D
    @test Q ≈ expected
    # Null space: 1 and 1:n
    @test maximum(abs, Q * ones(n)) < 1e-12
    @test maximum(abs, Q * Float64.(1:n)) < 1e-12
end

@testset "RW2GMRF cyclic" begin
    n = 6
    g = RW2GMRF(n; τ = 1.0, cyclic = true)
    @test rankdef(g) == 1
    Q = Matrix(precision_matrix(g))
    # Null space span{1}
    @test maximum(abs, Q * ones(n)) < 1e-12
end

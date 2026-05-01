using Test
using GMRFs
using LinearAlgebra
using SparseArrays

@testset "Generic0GMRF: proper wrapper agrees with IID/AR1" begin
    n = 5
    τ = 3.0
    R = sparse(1.0 * I, n, n)
    g = Generic0GMRF(R; τ=τ, rankdef=0, scale_model=false)
    @test precision_matrix(g) ≈ τ * sparse(I, n, n)
    @test rankdef(g) == 0
end

@testset "Generic0GMRF: rankdef validation" begin
    # Claim a higher rankdef than the true nullity — should throw.
    R = sparse(Matrix(1.0I, 4, 4))
    @test_throws ArgumentError Generic0GMRF(R; rankdef=1, scale_model=true)
end

@testset "Generic0GMRF: intrinsic structure, rankdef honored" begin
    # Structure = path-graph Laplacian (rankdef 1)
    L = Matrix([1.0 -1 0 0; -1 2 -1 0; 0 -1 2 -1; 0 0 -1 1])
    g = Generic0GMRF(sparse(L); τ=1.0, rankdef=1, scale_model=false)
    @test rankdef(g) == 1
    V = null_space_basis(g)
    @test size(V) == (4, 1)
    # Column should be (approximately) proportional to 1_n
    @test abs(dot(V[:, 1], fill(1 / 2, 4)))≈1.0 atol=1e-8
end

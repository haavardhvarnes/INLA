using Test
using GMRFs
using Graphs
using SparseArrays
using LinearAlgebra

@testset "GMRFGraph basic" begin
    g = GMRFGraph(path_graph(4))
    @test num_nodes(g) == 4
    @test nconnected_components(g) == 1
    A = adjacency_matrix(g)
    @test size(A) == (4, 4)
    @test issymmetric(A)
    @test A[1, 2] == true
    @test A[1, 3] == false
    @test all(iszero, diag(A))

    L = laplacian_matrix(g)
    @test L == [1 -1 0 0; -1 2 -1 0; 0 -1 2 -1; 0 0 -1 1]
end

@testset "GMRFGraph from adjacency matrix" begin
    W = sparse([0 1 0; 1 0 1; 0 1 0])
    g = GMRFGraph(W)
    @test num_nodes(g) == 3
    @test nconnected_components(g) == 1
    @test Matrix(adjacency_matrix(g)) == W

    # non-symmetric
    W2 = sparse([0 1 0; 0 0 1; 0 1 0])
    @test_throws ArgumentError GMRFGraph(W2)

    # nonzero diagonal
    W3 = sparse([1 1 0; 1 0 1; 0 1 0])
    @test_throws ArgumentError GMRFGraph(W3)

    # non-square
    W4 = sparse([0 1 0; 1 0 1])
    @test_throws DimensionMismatch GMRFGraph(W4)
end

@testset "Disconnected graph" begin
    # Two components: {1,2,3} path and {4,5} edge
    W = spzeros(Int, 5, 5)
    W[1, 2] = W[2, 1] = 1
    W[2, 3] = W[3, 2] = 1
    W[4, 5] = W[5, 4] = 1
    g = GMRFGraph(W)
    @test nconnected_components(g) == 2
    labels = connected_component_labels(g)
    @test labels[1] == labels[2] == labels[3]
    @test labels[4] == labels[5]
    @test labels[1] != labels[4]
end

using Test
using GMRFs
using Graphs
using LinearAlgebra
using SparseArrays
using Random
using Statistics

@testset "BesagGMRF structure = Laplacian (unscaled)" begin
    g = GMRFGraph(path_graph(5))
    b = BesagGMRF(g; τ=1.0, scale_model=false)
    Q = Matrix(precision_matrix(b))
    @test Q == Matrix(laplacian_matrix(g))
    @test rankdef(b) == 1
end

@testset "BesagGMRF scaled geometric-mean-variance ≈ 1" begin
    # Sørbye-Rue (2014): under scaling, geomean of non-null marginal
    # variances of τ⁻¹ Q⁻¹ should equal τ⁻¹. Check on a small cycle.
    g = GMRFGraph(cycle_graph(6))
    b = BesagGMRF(g; τ=1.0, scale_model=true)
    Q = Matrix(precision_matrix(b))
    V = null_space_basis(b)
    Σ = inv(Q + V * V') - V * V'
    # Geometric mean of diagonal on non-null subspace
    geomean_var = exp(mean(log.(diag(Σ))))
    @test geomean_var≈1.0 rtol=1e-8
end

@testset "BesagGMRF disconnected components" begin
    # Two disconnected cycles
    W = spzeros(Int, 6, 6)
    # Triangle on {1,2,3}
    W[1, 2] = W[2, 1] = 1
    W[2, 3] = W[3, 2] = 1
    W[3, 1] = W[1, 3] = 1
    # Edge on {4,5} + isolated {6}? — keep 4-5-6 as a path
    W[4, 5] = W[5, 4] = 1
    W[5, 6] = W[6, 5] = 1
    g = GMRFGraph(W)
    b = BesagGMRF(g; τ=1.0, scale_model=true)
    @test rankdef(b) == 2
    # Sum-to-zero per component
    cons = constraints(b)
    @test nconstraints(cons) == 2
    A = constraint_matrix(cons)
    @test A[1, 1:3] == [1, 1, 1]
    @test A[1, 4:6] == [0, 0, 0]
    @test A[2, 1:3] == [0, 0, 0]
    @test A[2, 4:6] == [1, 1, 1]
end

@testset "per_component_scale_factors — Freni-Sterrantino per-CC" begin
    # Triangle ⊔ path-{4,5,6}: K=2 components, both with K_k = 1 null vector.
    W = spzeros(Int, 6, 6)
    W[1, 2] = W[2, 1] = 1
    W[2, 3] = W[3, 2] = 1
    W[3, 1] = W[1, 3] = 1
    W[4, 5] = W[5, 4] = 1
    W[5, 6] = W[6, 5] = 1
    g = GMRFGraph(W)
    c = per_component_scale_factors(g)
    @test length(c) == 2
    @test all(c .> 0)

    # Geomean of marginal variance on each component's non-null subspace
    # should equal 1 — that's the defining property of the Sørbye-Rue
    # scaling, applied per-component.
    L = Matrix{Float64}(laplacian_matrix(g))
    labels = connected_component_labels(g)
    for k in 1:2
        idx = findall(==(k), labels)
        L_k = L[idx, idx]
        n_k = length(idx)
        v = fill(1 / sqrt(n_k), n_k)
        Q_k = c[k] .* L_k
        Σ_k = inv(Q_k + v * v') - v * v'
        @test exp(mean(log.(diag(Σ_k))))≈1.0 rtol=1.0e-8
    end

    # Singleton components get c_k = 1 (no non-null subspace to scale).
    W2 = spzeros(Int, 4, 4)
    W2[1, 2] = W2[2, 1] = 1
    W2[2, 3] = W2[3, 2] = 1
    g2 = GMRFGraph(W2)            # main {1,2,3} + singleton {4}
    c2 = per_component_scale_factors(g2)
    @test length(c2) == 2
    @test c2[2] == 1.0            # singleton convention
end

@testset "BesagGMRF sample respects per-component sum-to-zero" begin
    rng = MersenneTwister(11)
    W = spzeros(Int, 6, 6)
    W[1, 2] = W[2, 1] = 1
    W[2, 3] = W[3, 2] = 1
    W[4, 5] = W[5, 4] = 1
    W[5, 6] = W[6, 5] = 1
    g = GMRFGraph(W)
    b = BesagGMRF(g; τ=1.0, scale_model=true)
    for _ in 1:20
        x = rand(rng, b)
        @test abs(sum(x[1:3])) < 1e-9
        @test abs(sum(x[4:6])) < 1e-9
    end
end

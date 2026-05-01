# SPDE2 component — the AbstractLatentComponent contract, the
# (log τ, log κ) ↔ (ρ, σ) mapping per ADR-013, and interop with the
# M1 FEM assembly.

using GMRFs
using LatentGaussianModels
using LatentGaussianModels: initial_hyperparameters, nhyperparameters,
                            precision_matrix, log_hyperprior

@testset "SPDE2 — construction + AbstractLatentComponent contract" begin
    pc = PCMatern(range_U=0.5, range_α=0.05,
        sigma_U=1.0, sigma_α=0.01)
    spde = SPDE2(POINTS_SQ, TRIS_SQ; pc=pc)

    @test spde isa AbstractLatentComponent
    @test length(spde) == 4
    @test nhyperparameters(spde) == 2
    @test initial_hyperparameters(spde) == [0.0, 0.0]

    # NoConstraint — SPDE precision is SPD for all positive (τ, κ).
    @test GMRFs.constraints(spde) isa GMRFs.NoConstraint
end

@testset "SPDE2 — graph matches mesh-vertex adjacency" begin
    spde = SPDE2(POINTS_SQ, TRIS_SQ)
    g = spde.graph
    @test g isa GMRFs.AbstractGMRFGraph
    @test GMRFs.num_nodes(g) == 4

    # On the 2-triangle unit square, every pair of vertices shares a
    # triangle except (2, 4). So the graph has 5 edges.
    A = GMRFs.adjacency_matrix(g)
    @test A == A'
    @test sum(A) == 2 * 5            # symmetric bool matrix → twice #edges
    @test A[2, 4] == false
    @test A[1, 2] == true && A[1, 3] == true && A[1, 4] == true
    @test A[2, 3] == true && A[3, 4] == true
end

@testset "SPDE2 — precision_matrix agrees with spde_precision(fem, …)" begin
    spde = SPDE2(POINTS_SQ, TRIS_SQ)
    for (log_τ, log_κ) in ((0.0, 0.0), (-1.0, 0.7), (2.0, -0.5))
        Q_via_component = precision_matrix(spde, [log_τ, log_κ])
        Q_via_fem = spde_precision(spde.fem, 2, exp(log_τ), exp(log_κ))
        @test Matrix(Q_via_component)≈Matrix(Q_via_fem) rtol=1.0e-12
    end
end

@testset "SPDE2 — (log τ, log κ) ↔ (ρ, σ) mapping (ADR-013)" begin
    spde = SPDE2(POINTS_SQ, TRIS_SQ)
    # At (τ, κ) = (1, 1): ρ = √8, σ = 1/√(4π) for α = 2 in 2D.
    ρ, σ = spde_user_scale(spde, [0.0, 0.0])
    @test ρ≈sqrt(8) rtol=1.0e-12
    @test σ≈inv(sqrt(4π)) rtol=1.0e-12

    # Round-trip: user → internal → user.
    for (ρ_target, σ_target) in ((0.5, 0.3), (2.7, 1.9), (1.0, 1.0))
        log_τ, log_κ = spde_internal_scale(spde, ρ_target, σ_target)
        ρ_back, σ_back = spde_user_scale(spde, [log_τ, log_κ])
        @test ρ_back≈ρ_target rtol=1.0e-12
        @test σ_back≈σ_target rtol=1.0e-12
    end
end

@testset "SPDE2 — log_hyperprior matches PC-Matern evaluator" begin
    pc = PCMatern(range_U=0.5, range_α=0.05,
        sigma_U=1.0, sigma_α=0.01)
    spde = SPDE2(POINTS_SQ, TRIS_SQ; pc=pc)

    for θ in ([0.0, 0.0], [-1.3, 0.7], [2.1, -0.4])
        log_τ, log_κ = θ
        log_ρ = 0.5 * log(8.0) - log_κ
        log_σ = -0.5 * log(4π) - log_κ - log_τ
        expected = pc_matern_log_density(pc, log_ρ, log_σ)
        @test log_hyperprior(spde, θ)≈expected rtol=1.0e-12
    end
end

@testset "SPDE2 — α = 1 rejected (PC-Matern invalid, v0.1)" begin
    @test_throws ArgumentError SPDE2(POINTS_SQ, TRIS_SQ; α=1)
    @test_throws ArgumentError SPDE2(POINTS_SQ, TRIS_SQ; α=3)
end

@testset "SPDE2 — precision on fine mesh is SPD and sparse" begin
    # Reuse the 3×3 regular grid from the lumping test.
    nx, ny = 3, 3
    xs = range(0.0, 1.0; length=nx + 1)
    ys = range(0.0, 1.0; length=ny + 1)
    npts = (nx + 1) * (ny + 1)
    points = Matrix{Float64}(undef, npts, 2)
    idx(i, j) = (j - 1) * (nx + 1) + i
    for j in 1:(ny + 1), i in 1:(nx + 1)
        points[idx(i, j), 1] = xs[i]
        points[idx(i, j), 2] = ys[j]
    end
    tris = Int[]
    for j in 1:ny, i in 1:nx
        push!(tris, idx(i, j), idx(i + 1, j), idx(i + 1, j + 1))
        push!(tris, idx(i, j), idx(i + 1, j + 1), idx(i, j + 1))
    end
    triangles = permutedims(reshape(tris, 3, :))

    spde = SPDE2(points, triangles)
    Q = precision_matrix(spde, [0.0, 0.0])
    @test issymmetric(Q)
    @test isposdef(Symmetric(Matrix(Q)))
    @test size(Q) == (npts, npts)
    # Sparsity: not every entry is nonzero on a reasonable mesh.
    @test nnz(Q) < npts^2
end

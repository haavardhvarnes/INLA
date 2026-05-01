# MeshProjector — sparse barycentric projection from mesh vertices to
# arbitrary observation locations.

using Random: MersenneTwister
using LinearAlgebra: norm

@testset "MeshProjector — row sums to 1, at most 3 nonzeros per row" begin
    sq = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    mesh = inla_mesh_2d(; boundary=sq, max_edge=0.25, min_angle=25.0)

    rng = MersenneTwister(7)
    locs = 0.05 .+ 0.9 .* rand(rng, 40, 2)           # strictly interior
    P = MeshProjector(mesh, locs)

    @test size(P) == (40, num_vertices(mesh))
    # Each row sums to exactly 1 (barycentric partition of unity).
    row_sums = vec(sum(P.A, dims=2))
    @test all(row_sums .≈ 1.0)
    # Each row has ≤ 3 nonzeros.
    for i in 1:size(P, 1)
        nz = count(!iszero, @view P.A[i, :])
        @test nz <= 3
    end
    # Barycentric weights are non-negative for interior points.
    @test all(P.A.nzval .>= -1.0e-12)
end

@testset "MeshProjector — reproduces linear fields exactly" begin
    # A P1 projector is exact on linear fields. Pick u(x, y) = a + b·x + c·y
    # on the mesh; projected values at arbitrary interior locations
    # must equal u evaluated at those locations.
    sq = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    mesh = inla_mesh_2d(; boundary=sq, max_edge=0.3, min_angle=25.0)

    a, b, c = 2.0, -0.7, 1.3
    u_mesh = a .+ b .* mesh.points[:, 1] .+ c .* mesh.points[:, 2]

    rng = MersenneTwister(314)
    locs = 0.05 .+ 0.9 .* rand(rng, 50, 2)
    P = MeshProjector(mesh, locs)

    u_interp = P * u_mesh
    u_exact = a .+ b .* locs[:, 1] .+ c .* locs[:, 2]
    @test norm(u_interp - u_exact, Inf) < 1.0e-12
end

@testset "MeshProjector — vertex locations give identity-like rows" begin
    sq = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    mesh = inla_mesh_2d(; boundary=sq, max_edge=0.3, min_angle=25.0)

    # Query the first 5 mesh vertices as "observations". Each row should
    # place weight 1 on its own vertex (barycentric λ_i = 1 at vertex i).
    locs = mesh.points[1:5, :]
    P = MeshProjector(mesh, locs)

    # Row i: projected value equals u[mesh_vertex(i)] for any u.
    u = randn(MersenneTwister(1), num_vertices(mesh))
    for i in 1:5
        # There can be multiple triangles meeting at a vertex; any is
        # valid, as long as the weight on vertex i is 1 and the other
        # two are 0.
        @test (P * u)[i]≈u[i] rtol=1.0e-12
    end
end

@testset "MeshProjector — outside locations policy" begin
    sq = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    mesh = inla_mesh_2d(; boundary=sq, max_edge=0.5, min_angle=25.0)

    bad = [0.5 0.5;   # inside
           2.0 2.0;   # outside
           -0.5 0.5]  # outside

    @test_throws ArgumentError MeshProjector(mesh, bad)
    @test_throws ArgumentError MeshProjector(mesh, bad; outside=:error)

    P = MeshProjector(mesh, bad; outside=:zero)
    @test size(P) == (3, num_vertices(mesh))
    # Inside row has nonzero weights; outside rows are empty. The
    # interior point `(0.5, 0.5)` can land exactly on an edge of the
    # refined Delaunay triangulation, in which case one of the three
    # barycentric weights is zero — so accept 2 or 3 nonzeros, and
    # require the row to sum to 1.
    @test count(!iszero, @view P.A[1, :]) ∈ (2, 3)
    @test sum(@view P.A[1, :]) ≈ 1.0
    @test iszero(@view P.A[2, :])
    @test iszero(@view P.A[3, :])
end

@testset "MeshProjector — argument validation" begin
    sq = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    mesh = inla_mesh_2d(; boundary=sq, max_edge=0.5)

    @test_throws ArgumentError MeshProjector(mesh, rand(3, 3))               # 3D locs
    @test_throws ArgumentError MeshProjector(mesh, [0.5 0.5]; outside=:clamp)
    @test_throws ArgumentError MeshProjector(mesh, [0.5 0.5]; atol=-1.0)
end

@testset "MeshProjector — SciMLOperators wrapper" begin
    sq = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    mesh = inla_mesh_2d(; boundary=sq, max_edge=0.4, min_angle=25.0)
    P = MeshProjector(mesh, [0.25 0.25; 0.75 0.6])

    op = scimloperator(P)
    u = randn(MersenneTwister(2), num_vertices(mesh))
    @test op * u≈P * u rtol=1.0e-12
    @test size(op) == size(P)
end

@testset "MeshProjector — integrates with LatentGaussianModel" begin
    using LatentGaussianModels: LatentGaussianModel, GaussianLikelihood
    sq = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    mesh = inla_mesh_2d(; boundary=sq, max_edge=0.4, min_angle=25.0)

    spde = SPDE2(mesh)
    locs = [0.25 0.25; 0.5 0.5; 0.75 0.75]
    P = MeshProjector(mesh, locs)

    # The sparse A is directly accepted by LatentGaussianModel; the
    # v0.1 constructor wraps it in a LinearProjector internally
    # (ADR-017).
    like = GaussianLikelihood()
    model = LatentGaussianModel(like, spde, P.A)
    @test size(model.mapping) == (3, length(spde))
    @test model.mapping isa LinearProjector
end

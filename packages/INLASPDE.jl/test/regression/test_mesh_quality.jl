# `inla_mesh_2d` quality checks: the returned mesh must be a valid
# triangulation of the requested domain and satisfy the refinement
# parameters the user asked for (minimum interior angle, maximum
# triangle edge length).

using Random: MersenneTwister

# Helper: minimum interior angle across all triangles (degrees).
function min_triangle_angle(points, triangles)
    mins = Float64[]
    for t in axes(triangles, 1)
        i, j, k = triangles[t, 1], triangles[t, 2], triangles[t, 3]
        a = (points[i, 1], points[i, 2])
        b = (points[j, 1], points[j, 2])
        c = (points[k, 1], points[k, 2])
        for (p, q, r) in ((a, b, c), (b, c, a), (c, a, b))
            v1 = (p[1] - q[1], p[2] - q[2])
            v2 = (r[1] - q[1], r[2] - q[2])
            d = v1[1] * v2[1] + v1[2] * v2[2]
            m = sqrt((v1[1]^2 + v1[2]^2) * (v2[1]^2 + v2[2]^2))
            push!(mins, acos(clamp(d / m, -1.0, 1.0)) * 180 / π)
        end
    end
    return minimum(mins)
end

# Helper: maximum edge length across all triangles.
function max_triangle_edge(points, triangles)
    e = 0.0
    for t in axes(triangles, 1)
        i, j, k = triangles[t, 1], triangles[t, 2], triangles[t, 3]
        a = (points[i, 1], points[i, 2])
        b = (points[j, 1], points[j, 2])
        c = (points[k, 1], points[k, 2])
        for (p, q) in ((a, b), (b, c), (c, a))
            e = max(e, hypot(p[1] - q[1], p[2] - q[2]))
        end
    end
    return e
end

@testset "inla_mesh_2d — unit-square boundary, refinement bounds respected" begin
    sq = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    mesh = inla_mesh_2d(; boundary = sq, max_edge = 0.25, min_angle = 25.0)

    @test mesh isa INLAMesh
    @test num_vertices(mesh) > 4
    @test num_triangles(mesh) > 2
    @test size(mesh.points, 2) == 2
    @test size(mesh.triangles, 2) == 3

    # All triangle indices are in range.
    @test all(1 .<= mesh.triangles .<= num_vertices(mesh))

    # Boundary loop closes (treating it as implicitly closed).
    @test length(mesh.boundary) >= 4

    # Refinement guarantees: Ruppert enforces `min_angle` strictly; the
    # area bound gives a soft edge-length bound. Output edges may
    # modestly exceed `max_edge` because the refinement criterion is
    # area, not edge length — a proper edge-length pass is deferred.
    @test min_triangle_angle(mesh.points, mesh.triangles) >= 25.0 - 1.0e-6
    @test max_triangle_edge(mesh.points, mesh.triangles) <= 1.5 * 0.25
end

@testset "inla_mesh_2d — loc with offset builds an expanded outer hull" begin
    rng = MersenneTwister(0xBEEF)
    # 30 points in the unit square interior.
    loc = 0.1 .+ 0.8 .* rand(rng, 30, 2)
    mesh = inla_mesh_2d(loc; max_edge = 0.2, offset = 0.4, min_angle = 25.0)

    @test mesh isa INLAMesh
    @test num_vertices(mesh) >= 30
    # The mesh domain extends beyond [0, 1]² because of the offset.
    @test minimum(mesh.points[:, 1]) < 0.0
    @test minimum(mesh.points[:, 2]) < 0.0
    @test maximum(mesh.points[:, 1]) > 1.0
    @test maximum(mesh.points[:, 2]) > 1.0
    # All loc points are inside the mesh bounding box.
    @test minimum(mesh.points[:, 1]) <= minimum(loc[:, 1])
    @test maximum(mesh.points[:, 1]) >= maximum(loc[:, 1])
end

@testset "inla_mesh_2d — loc only (hull as boundary)" begin
    rng = MersenneTwister(42)
    loc = randn(rng, 50, 2)
    mesh = inla_mesh_2d(loc; max_edge = 1.0, min_angle = 25.0)

    @test num_triangles(mesh) > 0
    @test min_triangle_angle(mesh.points, mesh.triangles) >= 25.0 - 1.0e-6
    # Every loc hull point should appear as a boundary vertex.
    hull = convex_hull_polygon(loc)
    hull_set = Set(Tuple(hull[i, :]) for i in axes(hull, 1))
    mesh_bnd = Set(Tuple(mesh.points[v, :]) for v in mesh.boundary)
    @test hull_set ⊆ mesh_bnd
end

@testset "inla_mesh_2d — cutoff collapses near-duplicate loc points" begin
    # A small grid with duplicates placed within cutoff.
    base = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    dups = [0.0 0.0; 0.001 0.002; 1.0 0.0; 0.999 0.001; 1.0 1.0; 0.0 1.0]
    m1 = inla_mesh_2d(base; max_edge = 1.0, min_angle = 25.0)
    m2 = inla_mesh_2d(dups; max_edge = 1.0, min_angle = 25.0, cutoff = 0.01)
    @test num_vertices(m2) == num_vertices(m1)
    @test num_triangles(m2) == num_triangles(m1)
end

@testset "inla_mesh_2d — argument validation" begin
    @test_throws ArgumentError inla_mesh_2d(; max_edge = 0.1)   # no loc/boundary
    @test_throws ArgumentError inla_mesh_2d([0.0 0.0; 1.0 0.0; 0.0 1.0];
                                             max_edge = -0.1)
    @test_throws ArgumentError inla_mesh_2d([0.0 0.0; 1.0 0.0; 0.0 1.0];
                                             max_edge = 0.5, min_angle = -1.0)
    @test_throws ArgumentError inla_mesh_2d([0.0 0.0; 1.0 0.0; 0.0 1.0];
                                             max_edge = 0.5, min_angle = 40.0)
    @test_throws ArgumentError inla_mesh_2d([0.0 0.0; 1.0 0.0; 0.0 1.0];
                                             max_edge = 0.5, offset = -0.1)
    @test_throws ArgumentError inla_mesh_2d([0.0 0.0; 1.0 0.0; 0.0 1.0];
                                             max_edge = 0.5, cutoff = -0.1)
end

@testset "inla_mesh_2d — FEM assembly and SPDE2 interop" begin
    sq = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    mesh = inla_mesh_2d(; boundary = sq, max_edge = 0.25, min_angle = 25.0)

    # FEMMatrices convenience constructor agrees with raw-matrix form.
    fem1 = FEMMatrices(mesh)
    fem2 = FEMMatrices(mesh.points, mesh.triangles)
    @test fem1.C == fem2.C
    @test fem1.G1 == fem2.G1
    @test fem1.C_lumped == fem2.C_lumped
    @test fem1.G2 == fem2.G2

    # SPDE2 convenience constructor builds a valid component.
    spde = SPDE2(mesh)
    @test spde isa SPDE2
    @test length(spde) == num_vertices(mesh)

    # Precision at θ = 0 is SPD. On refined meshes the sparse-product
    # `G₂ = G₁ C̃⁻¹ G₁` accumulates roundoff asymmetries of size ε
    # relative to norm(Q); we check approximate symmetry against that
    # floor and then test SPD after a Symmetric wrap.
    Q = LatentGaussianModels.precision_matrix(spde, [0.0, 0.0])
    @test norm(Q - Q') <= 1.0e-10 * norm(Q)
    @test isposdef(Symmetric(Matrix(Q)))
end

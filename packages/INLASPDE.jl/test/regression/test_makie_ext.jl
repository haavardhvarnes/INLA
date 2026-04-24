# MakieExt — mesh plot-recipe glue. We exercise `convert_arguments`
# directly (without a Makie backend) so that the ext loads cleanly in
# headless CI and the returned layout matches what Makie expects for
# each primitive.

using MakieCore

@testset "MakieExt — loads when MakieCore is present" begin
    ext = Base.get_extension(INLASPDE, :INLASPDEMakieExt)
    @test ext !== nothing
end

@testset "MakieExt — scatter(mesh) yields a Vector of 2-tuples" begin
    sq = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    mesh = inla_mesh_2d(; boundary = sq, max_edge = 0.4, min_angle = 25.0)

    (pts,) = MakieCore.convert_arguments(MakieCore.Scatter, mesh)
    @test length(pts) == num_vertices(mesh)
    @test eltype(pts) == NTuple{2, Float64}
    for i in 1:num_vertices(mesh)
        @test pts[i] == (mesh.points[i, 1], mesh.points[i, 2])
    end
end

@testset "MakieExt — linesegments(mesh) emits 3 edges per triangle" begin
    sq = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    mesh = inla_mesh_2d(; boundary = sq, max_edge = 0.4, min_angle = 25.0)

    (segs,) = MakieCore.convert_arguments(MakieCore.LineSegments, mesh)
    @test length(segs) == 6 * num_triangles(mesh)       # 3 edges × 2 endpoints
    # First triangle's endpoints must appear as the first six entries.
    t1 = mesh.triangles[1, :]
    p = [(mesh.points[v, 1], mesh.points[v, 2]) for v in t1]
    @test segs[1:6] == [p[1], p[2], p[2], p[3], p[3], p[1]]
end

@testset "MakieExt — mesh(mesh) forwards (points, triangles)" begin
    sq = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    mesh = inla_mesh_2d(; boundary = sq, max_edge = 0.5, min_angle = 25.0)

    (pts, tri) = MakieCore.convert_arguments(MakieCore.Mesh, mesh)
    @test pts === mesh.points
    @test tri === mesh.triangles
end

@testset "MakieExt — wireframe(mesh) forwards (points, triangles)" begin
    sq = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    mesh = inla_mesh_2d(; boundary = sq, max_edge = 0.5, min_angle = 25.0)

    (pts, tri) = MakieCore.convert_arguments(MakieCore.Wireframe, mesh)
    @test pts === mesh.points
    @test tri === mesh.triangles
end

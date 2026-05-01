using Test
using INLASPDE: inla_mesh_2d, num_vertices
using INLASPDERasters: extract_at_mesh
using Rasters: Raster, X, Y

# Build a raster on [0, 1] × [0, 1] with a known affine field, then
# check that bilinear extraction reproduces the field exactly at mesh
# vertices. The mesh interior stays well inside the raster extent.
@testset "extract_at_mesh — bilinear reproduces affine fields exactly" begin
    xs = 0.0:0.05:1.0
    ys = 0.0:0.05:1.0
    f = (x, y) -> 2.0 * x + 3.0 * y + 1.0
    vals = [f(x, y) for x in xs, y in ys]
    r = Raster(vals, (X(collect(xs)), Y(collect(ys))))

    sq = [0.1 0.1; 0.9 0.1; 0.9 0.9; 0.1 0.9]
    mesh = inla_mesh_2d(; boundary=sq, max_edge=0.15, min_angle=25.0)

    u = extract_at_mesh(r, mesh; method=:bilinear)
    @test length(u) == num_vertices(mesh)

    expected = [f(mesh.points[k, 1], mesh.points[k, 2]) for k in 1:num_vertices(mesh)]
    @test maximum(abs, u .- expected) < 1.0e-12
end

@testset "extract_at_mesh — bilinear is exact at cell corners" begin
    xs = 0.0:0.25:1.0
    ys = 0.0:0.25:1.0
    vals = reshape(Float64.(1:(length(xs) * length(ys))), length(xs), length(ys))
    r = Raster(vals, (X(collect(xs)), Y(collect(ys))))

    # Boundary matches a raster cell — vertex coords land on cell corners.
    sq = [0.25 0.25; 0.75 0.25; 0.75 0.75; 0.25 0.75]
    mesh = inla_mesh_2d(; boundary=sq, max_edge=0.3, min_angle=25.0)

    u = extract_at_mesh(r, mesh; method=:bilinear)
    @test length(u) == num_vertices(mesh)

    # Corners of the mesh boundary are cell centres; the raster values
    # there must come through unchanged.
    for (k, corner) in enumerate(eachrow(sq))
        # Find the mesh vertex that matches each corner.
        idx = findfirst(
            i -> mesh.points[i, 1] ≈ corner[1] && mesh.points[i, 2] ≈ corner[2],
            1:num_vertices(mesh))
        @test idx !== nothing
        cx = findfirst(≈(corner[1]), xs)
        cy = findfirst(≈(corner[2]), ys)
        @test u[idx] ≈ vals[cx, cy]
    end
end

@testset "extract_at_mesh — nearest snaps to closest cell" begin
    xs = 0.0:1.0:4.0
    ys = 0.0:1.0:4.0
    vals = reshape(Float64.(1:25), 5, 5)
    r = Raster(vals, (X(collect(xs)), Y(collect(ys))))

    sq = [0.5 0.5; 3.5 0.5; 3.5 3.5; 0.5 3.5]
    mesh = inla_mesh_2d(; boundary=sq, max_edge=1.0, min_angle=25.0)

    u_near = extract_at_mesh(r, mesh; method=:nearest)
    @test length(u_near) == num_vertices(mesh)

    # Every returned value must exist somewhere in the raster cells.
    for v in u_near
        @test any(≈(v), vals)
    end

    # A known-centred vertex: find a mesh vertex inside cell (2,2) but
    # strictly closer to corner (2,2) than to any neighbour. The mesh
    # includes the square corners, so (0.5, 0.5) lies dead-centre
    # between the four corner cells (1..2, 1..2) and will snap
    # deterministically to one of them — just check the value is in
    # that neighbourhood.
    idx = findfirst(k -> mesh.points[k, 1] ≈ 0.5 && mesh.points[k, 2] ≈ 0.5,
        1:num_vertices(mesh))
    @test idx !== nothing
    # The snapped value must be one of the four surrounding cell values.
    neighbourhood = (vals[1, 1], vals[2, 1], vals[1, 2], vals[2, 2])
    @test u_near[idx] ∈ neighbourhood
end

@testset "extract_at_mesh — descending coordinates work" begin
    # Rasters commonly have Y descending (top-to-bottom). Both axes must
    # be handled.
    xs = 0.0:0.1:1.0
    ys = collect(1.0:-0.1:0.0)
    f = (x, y) -> 5.0 * x - 2.0 * y
    vals = [f(x, y) for x in xs, y in ys]
    r = Raster(vals, (X(collect(xs)), Y(ys)))

    sq = [0.2 0.2; 0.8 0.2; 0.8 0.8; 0.2 0.8]
    mesh = inla_mesh_2d(; boundary=sq, max_edge=0.2, min_angle=25.0)

    u = extract_at_mesh(r, mesh; method=:bilinear)
    expected = [f(mesh.points[k, 1], mesh.points[k, 2]) for k in 1:num_vertices(mesh)]
    @test maximum(abs, u .- expected) < 1.0e-12
end

@testset "extract_at_mesh — outside-domain policy" begin
    xs = 0.0:0.1:1.0
    ys = 0.0:0.1:1.0
    vals = zeros(length(xs), length(ys))
    r = Raster(vals, (X(collect(xs)), Y(collect(ys))))

    # A boundary that pokes outside the raster extent.
    sq = [-0.5 -0.5; 1.5 -0.5; 1.5 1.5; -0.5 1.5]
    mesh = inla_mesh_2d(; boundary=sq, max_edge=0.5, min_angle=25.0)

    @test_throws ArgumentError extract_at_mesh(r, mesh)
    @test_throws ArgumentError extract_at_mesh(r, mesh; outside=:error)

    u = extract_at_mesh(r, mesh; outside=:missing, missingval=-99.0)
    @test length(u) == num_vertices(mesh)
    # At least one vertex (the corners of `sq`) is outside the raster.
    @test any(==(-99.0), u)
end

@testset "extract_at_mesh — argument validation" begin
    xs = 0.0:0.25:1.0
    ys = 0.0:0.25:1.0
    r = Raster(zeros(length(xs), length(ys)), (X(collect(xs)), Y(collect(ys))))
    sq = [0.1 0.1; 0.9 0.1; 0.9 0.9; 0.1 0.9]
    mesh = inla_mesh_2d(; boundary=sq, max_edge=0.3, min_angle=25.0)

    @test_throws ArgumentError extract_at_mesh(r, mesh; method=:spline)
    @test_throws ArgumentError extract_at_mesh(r, mesh; outside=:clamp)
end

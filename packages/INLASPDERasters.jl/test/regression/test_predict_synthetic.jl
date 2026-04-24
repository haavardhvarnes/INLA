using Test
using INLASPDE: inla_mesh_2d, num_vertices
using INLASPDERasters: predict_raster
using Rasters: Rasters, Raster, X, Y, dims

# A P1 projector is exact on linear fields. Pick u(x, y) = a + b·x + c·y
# on the mesh; each interior pixel must equal u at its cell centre.
@testset "predict_raster — reproduces linear fields exactly inside mesh" begin
    sq = [0.1 0.1; 0.9 0.1; 0.9 0.9; 0.1 0.9]
    mesh = inla_mesh_2d(; boundary = sq, max_edge = 0.15, min_angle = 25.0)

    a, b, c = 2.0, -0.7, 1.3
    u = a .+ b .* mesh.points[:, 1] .+ c .* mesh.points[:, 2]

    # Template strictly interior to the mesh.
    xs = 0.2:0.05:0.8
    ys = 0.2:0.05:0.8
    template = Raster(zeros(length(xs), length(ys)), (X(collect(xs)), Y(collect(ys))))

    r = predict_raster(u, mesh, template)
    @test size(r) == size(template)

    # Every cell centre must equal u exactly.
    err = 0.0
    for (i, x) in enumerate(xs), (j, y) in enumerate(ys)
        expected = a + b * x + c * y
        err = max(err, abs(r[X = i, Y = j] - expected))
    end
    @test err < 1.0e-12
end

@testset "predict_raster — preserves template dims and extent" begin
    sq = [0.1 0.1; 0.9 0.1; 0.9 0.9; 0.1 0.9]
    mesh = inla_mesh_2d(; boundary = sq, max_edge = 0.25, min_angle = 25.0)
    u = ones(num_vertices(mesh))

    xs = 0.2:0.1:0.8
    ys = 0.2:0.1:0.8
    template = Raster(zeros(length(xs), length(ys)), (X(collect(xs)), Y(collect(ys))))

    r = predict_raster(u, mesh, template)
    @test size(r) == size(template)
    # Dims are the same object type in the same order.
    @test typeof(dims(r)) === typeof(dims(template))
    # Lookups match (same X and Y coordinates).
    @test collect(Rasters.lookup(r, X)) == collect(Rasters.lookup(template, X))
    @test collect(Rasters.lookup(r, Y)) == collect(Rasters.lookup(template, Y))
end

@testset "predict_raster — constant field projects to constant raster" begin
    sq = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    mesh = inla_mesh_2d(; boundary = sq, max_edge = 0.25, min_angle = 25.0)

    # Template strictly inside the mesh so no cells are masked.
    xs = 0.1:0.1:0.9
    ys = 0.1:0.1:0.9
    template = Raster(zeros(length(xs), length(ys)), (X(collect(xs)), Y(collect(ys))))

    r = predict_raster(fill(7.5, num_vertices(mesh)), mesh, template)
    for v in parent(r)
        @test v ≈ 7.5 rtol = 1.0e-12
    end
end

@testset "predict_raster — outside cells get missingval" begin
    # Small mesh, large raster — cells outside get masked.
    sq = [0.3 0.3; 0.7 0.3; 0.7 0.7; 0.3 0.7]
    mesh = inla_mesh_2d(; boundary = sq, max_edge = 0.15, min_angle = 25.0)

    xs = 0.0:0.1:1.0
    ys = 0.0:0.1:1.0
    template = Raster(zeros(length(xs), length(ys)), (X(collect(xs)), Y(collect(ys))))

    u = ones(num_vertices(mesh))
    r = predict_raster(u, mesh, template; outside = :missing, missingval = -99.0)

    # Cell (0.0, 0.0) is outside the [0.3, 0.7]² mesh.
    @test r[X = 1, Y = 1] == -99.0
    # Cell (0.5, 0.5) is inside — value interpolated from u ≡ 1 is 1.0.
    mid_x = findfirst(≈(0.5), xs)
    mid_y = findfirst(≈(0.5), ys)
    @test r[X = mid_x, Y = mid_y] ≈ 1.0
end

@testset "predict_raster — :error throws when a cell is outside" begin
    sq = [0.3 0.3; 0.7 0.3; 0.7 0.7; 0.3 0.7]
    mesh = inla_mesh_2d(; boundary = sq, max_edge = 0.15, min_angle = 25.0)

    xs = 0.0:0.1:1.0
    ys = 0.0:0.1:1.0
    template = Raster(zeros(length(xs), length(ys)), (X(collect(xs)), Y(collect(ys))))
    u = ones(num_vertices(mesh))

    @test_throws ArgumentError predict_raster(u, mesh, template; outside = :error)
end

@testset "predict_raster — works with reversed dim order (Y, X)" begin
    sq = [0.1 0.1; 0.9 0.1; 0.9 0.9; 0.1 0.9]
    mesh = inla_mesh_2d(; boundary = sq, max_edge = 0.25, min_angle = 25.0)

    a, b, c = 1.5, 2.0, -1.0
    u = a .+ b .* mesh.points[:, 1] .+ c .* mesh.points[:, 2]

    xs = 0.2:0.1:0.8
    ys = 0.2:0.1:0.8
    # Dim order (Y, X) means the storage is (ny, nx).
    template = Raster(zeros(length(ys), length(xs)), (Y(collect(ys)), X(collect(xs))))

    r = predict_raster(u, mesh, template)
    @test size(r) == size(template)

    for (i, x) in enumerate(xs), (j, y) in enumerate(ys)
        expected = a + b * x + c * y
        @test r[X = i, Y = j] ≈ expected atol = 1.0e-12
    end
end

@testset "predict_raster — argument validation" begin
    sq = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    mesh = inla_mesh_2d(; boundary = sq, max_edge = 0.3, min_angle = 25.0)
    template = Raster(zeros(5, 5), (X(0.1:0.2:0.9), Y(0.1:0.2:0.9)))

    # Wrong length.
    @test_throws ArgumentError predict_raster(ones(2), mesh, template)
    # Unknown policy.
    @test_throws ArgumentError predict_raster(
        ones(num_vertices(mesh)), mesh, template; outside = :clamp,
    )
end

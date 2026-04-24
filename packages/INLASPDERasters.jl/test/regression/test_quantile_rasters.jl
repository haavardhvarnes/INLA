using Test
using INLASPDE: inla_mesh_2d, num_vertices
using INLASPDERasters: quantile_rasters, predict_raster
using Rasters: Raster, X, Y

@testset "quantile_rasters — mean/sd/lower/upper layout" begin
    sq = [0.1 0.1; 0.9 0.1; 0.9 0.9; 0.1 0.9]
    mesh = inla_mesh_2d(; boundary = sq, max_edge = 0.25, min_angle = 25.0)
    n = num_vertices(mesh)

    mean = fill(3.0, n)
    sd = fill(0.5, n)

    xs = 0.2:0.1:0.8
    ys = 0.2:0.1:0.8
    template = Raster(zeros(length(xs), length(ys)), (X(collect(xs)), Y(collect(ys))))

    q = quantile_rasters(mean, sd, mesh, template)
    @test keys(q) == (:mean, :sd, :lower, :upper)

    # Constant mean projects to constant raster.
    for v in parent(q.mean)
        @test v ≈ 3.0 rtol = 1.0e-12
    end
    for v in parent(q.sd)
        @test v ≈ 0.5 rtol = 1.0e-12
    end
    # Default z ≈ 1.96: lower ≈ 3 - 0.98, upper ≈ 3 + 0.98.
    for v in parent(q.lower)
        @test v ≈ 3.0 - 1.959963984540054 * 0.5 rtol = 1.0e-12
    end
    for v in parent(q.upper)
        @test v ≈ 3.0 + 1.959963984540054 * 0.5 rtol = 1.0e-12
    end
end

@testset "quantile_rasters — matches predict_raster on mean ± z*sd" begin
    # The contract is that lower/upper are linear projections of the
    # vertex-level intervals, not a separate computation. Confirm this
    # explicitly.
    sq = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    mesh = inla_mesh_2d(; boundary = sq, max_edge = 0.3, min_angle = 25.0)
    n = num_vertices(mesh)

    mean = [2.0 * p[1] + 1.0 for p in eachrow(mesh.points)]
    sd = [0.1 + 0.3 * p[2] for p in eachrow(mesh.points)]

    xs = 0.1:0.1:0.9
    ys = 0.1:0.1:0.9
    template = Raster(zeros(length(xs), length(ys)), (X(collect(xs)), Y(collect(ys))))

    z = 1.5
    q = quantile_rasters(mean, sd, mesh, template; z = z)

    lower_ref = predict_raster(mean .- z .* sd, mesh, template)
    upper_ref = predict_raster(mean .+ z .* sd, mesh, template)
    @test parent(q.lower) ≈ parent(lower_ref) rtol = 1.0e-12
    @test parent(q.upper) ≈ parent(upper_ref) rtol = 1.0e-12
    # Upper > lower everywhere (sd > 0, z > 0).
    @test all(parent(q.upper) .>= parent(q.lower))
end

@testset "quantile_rasters — outside cells masked consistently" begin
    # Mesh smaller than template; masked cells get missingval in all four
    # rasters.
    sq = [0.3 0.3; 0.7 0.3; 0.7 0.7; 0.3 0.7]
    mesh = inla_mesh_2d(; boundary = sq, max_edge = 0.15, min_angle = 25.0)
    n = num_vertices(mesh)

    xs = 0.0:0.1:1.0
    ys = 0.0:0.1:1.0
    template = Raster(zeros(length(xs), length(ys)), (X(collect(xs)), Y(collect(ys))))

    q = quantile_rasters(
        fill(1.0, n), fill(0.25, n), mesh, template;
        outside = :missing, missingval = -77.0,
    )
    @test q.mean[X = 1, Y = 1] == -77.0
    @test q.sd[X = 1, Y = 1] == -77.0
    @test q.lower[X = 1, Y = 1] == -77.0
    @test q.upper[X = 1, Y = 1] == -77.0
end

@testset "quantile_rasters — argument validation" begin
    sq = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    mesh = inla_mesh_2d(; boundary = sq, max_edge = 0.4, min_angle = 25.0)
    n = num_vertices(mesh)
    template = Raster(zeros(3, 3), (X(0.2:0.3:0.8), Y(0.2:0.3:0.8)))

    @test_throws ArgumentError quantile_rasters(ones(n - 1), ones(n), mesh, template)
    @test_throws ArgumentError quantile_rasters(ones(n), ones(n - 1), mesh, template)
    @test_throws ArgumentError quantile_rasters(ones(n), -ones(n), mesh, template)
    @test_throws ArgumentError quantile_rasters(ones(n), ones(n), mesh, template; z = -1.0)
end

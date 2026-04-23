# Boundary helpers used by `inla_mesh_2d`: convex hull extraction,
# polygon outward expansion, and point-cloud cutoff dedup.

@testset "convex_hull_polygon — unit square point set" begin
    pts = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0; 0.5 0.5; 0.3 0.7]
    hull = convex_hull_polygon(pts)
    @test size(hull) == (4, 2)
    # The four corners of the unit square are the hull vertices.
    hull_rows = Set(Tuple(hull[i, :]) for i in 1:4)
    expected = Set([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    @test hull_rows == expected
    # CCW orientation → signed area is positive.
    signed_area = 0.0
    for i in 1:4
        j = i == 4 ? 1 : i + 1
        signed_area += hull[i, 1] * hull[j, 2] - hull[j, 1] * hull[i, 2]
    end
    @test signed_area > 0
end

@testset "convex_hull_polygon — too few points rejected" begin
    @test_throws ArgumentError convex_hull_polygon([0.0 0.0; 1.0 1.0])
    # Wrong dimensionality also rejected.
    @test_throws ArgumentError convex_hull_polygon(rand(5, 3))
end

@testset "expand_polygon — unit square grows by perpendicular offset" begin
    sq = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    for δ in (0.1, 0.5, 1.25)
        expanded = expand_polygon(sq, δ)
        @test size(expanded) == size(sq)
        # Each input edge sits δ inside the corresponding expanded edge.
        # Expanded square: corners at (-δ, -δ), (1+δ, -δ), (1+δ, 1+δ), (-δ, 1+δ).
        expected = [-δ -δ; 1 + δ -δ; 1 + δ 1 + δ; -δ 1 + δ]
        @test expanded ≈ expected rtol = 1.0e-12
    end
end

@testset "expand_polygon — offset 0 is a copy" begin
    sq = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    @test expand_polygon(sq, 0.0) == Matrix{Float64}(sq)
end

@testset "expand_polygon — regular triangle expands radially" begin
    # Equilateral triangle centred at origin, circumradius 1.
    θs = (π / 2, π / 2 + 2π / 3, π / 2 + 4π / 3)
    tri = [cos(θs[1]) sin(θs[1]); cos(θs[2]) sin(θs[2]); cos(θs[3]) sin(θs[3])]
    δ = 0.3
    expanded = expand_polygon(tri, δ)
    # Each expanded edge must be exactly δ from its parent edge.
    for i in 1:3
        j = i == 3 ? 1 : i + 1
        # Edge direction and outward normal in the original triangle.
        ex, ey = tri[j, 1] - tri[i, 1], tri[j, 2] - tri[i, 2]
        L = hypot(ex, ey)
        nx, ny = ey / L, -ex / L
        # Perpendicular distance from any point on the expanded edge to
        # the original edge: signed projection of (expanded - original)
        # onto the outward normal.
        dx = expanded[i, 1] - tri[i, 1]
        dy = expanded[i, 2] - tri[i, 2]
        @test dx * nx + dy * ny ≈ δ rtol = 1.0e-12
    end
end

@testset "expand_polygon — argument validation" begin
    sq = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    @test_throws ArgumentError expand_polygon(sq, -0.1)
    @test_throws ArgumentError expand_polygon([0.0 0.0; 1.0 1.0], 0.1)  # < 3 verts
end

@testset "cutoff_dedup — first-wins greedy with cutoff" begin
    pts = [0.0 0.0;   # keep
           0.1 0.1;   # dropped (within cutoff of 1)
           1.0 0.0;   # keep
           1.05 0.0;  # dropped
           0.0 1.0]   # keep
    kept = cutoff_dedup(pts, 0.2)
    @test size(kept, 1) == 3
    @test kept[1, :] == [0.0, 0.0]
    @test kept[2, :] == [1.0, 0.0]
    @test kept[3, :] == [0.0, 1.0]
end

@testset "cutoff_dedup — cutoff ≤ 0 is identity" begin
    pts = [0.0 0.0; 0.5 0.5; 1.0 1.0]
    @test cutoff_dedup(pts, 0.0) == Matrix{Float64}(pts)
    @test cutoff_dedup(pts, -1.0) == Matrix{Float64}(pts)
end

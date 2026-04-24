# INLASPDEGeoInterfaceExt — accept GeoInterface geometries as `boundary`
# and `locations` inputs to `inla_mesh_2d` and `MeshProjector`. The
# matrix path stays canonical; these tests establish that the ext routes
# GeoInterface inputs through the same coercion hooks and produces
# identical meshes / projectors.

using GeoInterface
using Random: Random
const GI = GeoInterface

@testset "INLASPDEGeoInterfaceExt" begin
    # Unit square polygon, CCW, with a repeating closing vertex.
    square_pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
    square_poly = GI.Wrappers.Polygon([square_pts])
    square_mat = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]

    @testset "_as_boundary_matrix — polygon drops closing duplicate" begin
        M = INLASPDE._as_boundary_matrix(square_poly)
        @test size(M) == (4, 2)
        @test M ≈ square_mat
    end

    @testset "_as_boundary_matrix — closed LineString treated as ring" begin
        ls = GI.Wrappers.LineString(square_pts)
        M = INLASPDE._as_boundary_matrix(ls)
        @test size(M) == (4, 2)
        @test M ≈ square_mat
    end

    @testset "_as_boundary_matrix — polygon without closing duplicate" begin
        open_pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        poly = GI.Wrappers.Polygon([open_pts])
        M = INLASPDE._as_boundary_matrix(poly)
        @test size(M) == (4, 2)
        @test M ≈ square_mat
    end

    @testset "inla_mesh_2d — polygon boundary equivalent to matrix" begin
        # DelaunayTriangulation uses the default RNG for point-insertion
        # ordering; seed before each call so the two meshes are
        # byte-identical rather than just equivalent.
        Random.seed!(42)
        mesh_from_poly = inla_mesh_2d(; boundary = square_poly, max_edge = 0.3)
        Random.seed!(42)
        mesh_from_mat  = inla_mesh_2d(; boundary = square_mat,  max_edge = 0.3)
        @test num_vertices(mesh_from_poly) == num_vertices(mesh_from_mat)
        @test num_triangles(mesh_from_poly) == num_triangles(mesh_from_mat)
        @test mesh_from_poly.points ≈ mesh_from_mat.points
        @test mesh_from_poly.triangles == mesh_from_mat.triangles
    end

    @testset "_as_boundary_matrix — errors on non-geometry input" begin
        # With the ext loaded, unsupported inputs go through the Any
        # method and raise ArgumentError (no geomtrait).
        @test_throws ArgumentError INLASPDE._as_boundary_matrix(:not_a_geometry)
        # Point is a geometry but not a polygon / ring.
        pt = GI.Wrappers.Point(0.0, 0.0)
        @test_throws ArgumentError INLASPDE._as_boundary_matrix(pt)
    end

    # --- locations ---------------------------------------------------
    pts = [(0.2, 0.3), (0.5, 0.5), (0.9, 0.1)]
    pts_mat = [0.2 0.3; 0.5 0.5; 0.9 0.1]

    @testset "_as_location_matrix — MultiPoint" begin
        mp = GI.Wrappers.MultiPoint(pts)
        @test INLASPDE._as_location_matrix(mp) ≈ pts_mat
    end

    @testset "_as_location_matrix — Vector of Points" begin
        v = [GI.Wrappers.Point(x, y) for (x, y) in pts]
        @test INLASPDE._as_location_matrix(v) ≈ pts_mat
    end

    @testset "_as_location_matrix — single Point returns 1×2" begin
        p = GI.Wrappers.Point(0.4, 0.6)
        M = INLASPDE._as_location_matrix(p)
        @test size(M) == (1, 2)
        @test M ≈ [0.4 0.6]
    end

    @testset "_as_location_matrix — empty vector" begin
        v = GI.Wrappers.Point[]
        M = INLASPDE._as_location_matrix(v)
        @test size(M) == (0, 2)
    end

    @testset "MeshProjector — MultiPoint equivalent to matrix" begin
        mesh = inla_mesh_2d(; boundary = square_mat, max_edge = 0.3)
        mp = GI.Wrappers.MultiPoint(pts)
        P_geo = MeshProjector(mesh, mp)
        P_mat = MeshProjector(mesh, pts_mat)
        @test sparse(P_geo) ≈ sparse(P_mat)
        @test P_geo.locations ≈ P_mat.locations
    end

    @testset "_as_location_matrix — errors on non-point vector element" begin
        bad = Any[GI.Wrappers.Point(0.0, 0.0), :oops]
        @test_throws ArgumentError INLASPDE._as_location_matrix(bad)
    end
end

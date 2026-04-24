"""
    INLAMesh{T, TRI}

A 2D triangular mesh in INLA-SPDE form: raw `points` and `triangles`
matrices suitable for [`FEMMatrices`](@ref) / [`SPDE2`](@ref), plus the
underlying `DelaunayTriangulation.Triangulation` for advanced use
(further refinement, point-in-triangle queries, etc.).

# Fields

- `points::Matrix{T}` — `n × 2` vertex coordinates.
- `triangles::Matrix{Int}` — `m × 3` one-based vertex indices.
- `boundary::Vector{Int}` — ordered boundary vertex indices (open loop
  in the CCW convention; the loop is closed implicitly).
- `triangulation::TRI` — the raw DT object.

Construct via [`inla_mesh_2d`](@ref).
"""
struct INLAMesh{T <: Real, TRI}
    points::Matrix{T}
    triangles::Matrix{Int}
    boundary::Vector{Int}
    triangulation::TRI
    dt_to_mesh::Dict{Int, Int}
end

Base.show(io::IO, mesh::INLAMesh) = print(
    io,
    "INLAMesh(",
    size(mesh.points, 1),
    " vertices, ",
    size(mesh.triangles, 1),
    " triangles, ",
    length(mesh.boundary),
    " boundary)",
)

num_vertices(mesh::INLAMesh) = size(mesh.points, 1)
num_triangles(mesh::INLAMesh) = size(mesh.triangles, 1)

"""
    inla_mesh_2d([loc]; boundary, max_edge, offset, cutoff, min_angle) -> INLAMesh

Julia-native analogue of R-INLA's `inla.mesh.2d`: builds a constrained
Delaunay triangulation refined to satisfy a maximum edge length and a
minimum triangle angle.

# Arguments

- `loc::AbstractMatrix{<:Real}` (optional, `n × 2`) — observation
  locations. If `boundary` is not supplied, the mesh domain defaults to
  the convex hull of `loc`, optionally expanded by `offset`.
- `boundary::AbstractMatrix{<:Real}` — `k × 2` CCW polygon defining the
  mesh domain (no repeating closing vertex). If omitted, the domain is
  derived from `loc`.
- `max_edge::Real` — upper bound on triangle edge length. Implemented as
  a triangle-area constraint `max_area = √3 / 4 · max_edge²`
  (equilateral reference), which Ruppert refinement enforces strictly.
- `offset::Real = 0` — extension buffer applied to the `loc` convex
  hull when `boundary` is not given. Matches R-INLA's `offset`.
- `cutoff::Real = 0` — minimum pairwise distance between interior
  points; closer points are collapsed (first-wins) before
  triangulation. Has no effect on boundary vertices.
- `min_angle::Real = 21.0` — minimum interior triangle angle in
  degrees. Must be below ≈33.8° for Ruppert's algorithm to terminate
  (Shewchuk 2002); values much below 20° give fewer refinement steps
  but coarser meshes near acute corners.

# Returns

An [`INLAMesh`](@ref) with `points`, `triangles`, `boundary`, and the
underlying `DelaunayTriangulation` object.

# Example

```julia
loc = randn(50, 2)
mesh = inla_mesh_2d(loc; max_edge = 0.3, offset = 0.5, min_angle = 25.0)
fem  = FEMMatrices(mesh)
spde = SPDE2(mesh)
```
"""
function inla_mesh_2d(
        loc = nothing;
        boundary = nothing,
        max_edge::Real,
        offset::Real = 0.0,
        cutoff::Real = 0.0,
        min_angle::Real = 21.0,
    )
    max_edge > 0 ||
        throw(ArgumentError("max_edge must be positive; got $max_edge"))
    min_angle > 0 ||
        throw(ArgumentError("min_angle must be positive; got $min_angle"))
    min_angle < 33.8 ||
        throw(ArgumentError("min_angle must be < 33.8° for Ruppert termination; got $min_angle"))
    offset >= 0 ||
        throw(ArgumentError("offset must be non-negative; got $offset"))
    cutoff >= 0 ||
        throw(ArgumentError("cutoff must be non-negative; got $cutoff"))

    loc === nothing && boundary === nothing &&
        throw(ArgumentError("inla_mesh_2d requires either `loc` or `boundary`"))

    loc_m = _as_location_matrix(loc)
    bnd_m = _as_boundary_matrix(boundary)

    bnd_poly, interior_pts = _mesh_domain(loc_m, bnd_m, offset, cutoff)

    n_b = size(bnd_poly, 1)
    n_i = size(interior_pts, 1)
    all_pts = Vector{NTuple{2, Float64}}(undef, n_b + n_i)
    @inbounds for i in 1:n_b
        all_pts[i] = (bnd_poly[i, 1], bnd_poly[i, 2])
    end
    @inbounds for i in 1:n_i
        all_pts[n_b + i] = (interior_pts[i, 1], interior_pts[i, 2])
    end

    boundary_nodes = [[vcat(collect(1:n_b), 1)]]

    # Equilateral-triangle area as the Ruppert area bound for a target
    # edge length.  Not tight for right triangles — they'd need
    # `max_edge²/2` — but the refinement keeps a generous safety factor
    # since edges in the output are typically below the bound.
    max_area = sqrt(3.0) / 4.0 * max_edge^2

    tri = DelaunayTriangulation.triangulate(all_pts; boundary_nodes = boundary_nodes)
    DelaunayTriangulation.refine!(tri; min_angle = Float64(min_angle), max_area = max_area)

    return _build_inla_mesh(tri)
end

# ------------------------------------------------------------------
# Domain assembly: produce a CCW boundary polygon + any extra interior
# vertices to seed the triangulation with.
# ------------------------------------------------------------------
function _mesh_domain(loc, boundary, offset, cutoff)
    if boundary !== nothing
        size(boundary, 2) == 2 ||
            throw(ArgumentError("boundary must be k × 2; got size $(size(boundary))"))
        bnd = Matrix{Float64}(boundary)
        interior = if loc === nothing
            Matrix{Float64}(undef, 0, 2)
        else
            # Keep only loc points strictly inside the boundary polygon
            # (duplicates of boundary vertices would create degenerate
            # triangulations). Cutoff is also applied to interior points.
            loc_d = cutoff_dedup(loc, cutoff)
            _interior_only(loc_d, bnd)
        end
        return bnd, interior
    end

    @assert loc !== nothing
    loc_d = cutoff_dedup(loc, cutoff)
    hull = convex_hull_polygon(loc_d)
    if offset > 0
        bnd = expand_polygon(hull, offset)
        # All dedup'd loc points fall strictly inside the expanded hull.
        return bnd, loc_d
    else
        # Hull vertices become boundary; the remaining loc points are
        # interior. Boundary vertices must not appear twice, so strip
        # them from the interior set.
        return hull, _strip_hull_points(loc_d, hull)
    end
end

"""
    _interior_only(pts, polygon) -> Matrix{Float64}

Rows of `pts` strictly inside the CCW `polygon` (not on the boundary).
"""
function _interior_only(pts::AbstractMatrix{<:Real}, polygon::AbstractMatrix{<:Real})
    n = size(pts, 1)
    keep = Int[]
    for i in 1:n
        x = Float64(pts[i, 1])
        y = Float64(pts[i, 2])
        if _point_strictly_inside(x, y, polygon)
            push!(keep, i)
        end
    end
    P = Matrix{Float64}(undef, length(keep), 2)
    for (k, i) in enumerate(keep)
        P[k, 1] = pts[i, 1]
        P[k, 2] = pts[i, 2]
    end
    return P
end

"""
    _strip_hull_points(pts, hull) -> Matrix{Float64}

Rows of `pts` strictly inside `hull`; rows on the hull are dropped.
"""
_strip_hull_points(pts::AbstractMatrix{<:Real}, hull::AbstractMatrix{<:Real}) =
    _interior_only(pts, hull)

# Standard ray-casting / half-plane test for a CCW polygon. Returns
# true iff (x, y) is strictly inside.
function _point_strictly_inside(x::Real, y::Real, polygon::AbstractMatrix{<:Real})
    n = size(polygon, 1)
    for i in 1:n
        j = i == n ? 1 : i + 1
        ax = polygon[i, 1]; ay = polygon[i, 2]
        bx = polygon[j, 1]; by = polygon[j, 2]
        cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
        cross > 0 || return false
    end
    return true
end

# ------------------------------------------------------------------
# DT → INLAMesh conversion.
# ------------------------------------------------------------------
function _build_inla_mesh(tri)
    # Build a dense vertex index map: the triangulation may have added
    # Steiner points whose index exceeds num_solid_vertices; we compact
    # them to 1:n.
    solid_verts = sort!(collect(DelaunayTriangulation.each_solid_vertex(tri)))
    idxmap = Dict{Int, Int}()
    for (k, v) in enumerate(solid_verts)
        idxmap[v] = k
    end
    n = length(solid_verts)

    points = Matrix{Float64}(undef, n, 2)
    for (k, v) in enumerate(solid_verts)
        p = DelaunayTriangulation.get_point(tri, v)
        points[k, 1] = p[1]
        points[k, 2] = p[2]
    end

    solid_tris = collect(DelaunayTriangulation.each_solid_triangle(tri))
    m = length(solid_tris)
    triangles = Matrix{Int}(undef, m, 3)
    for (k, T) in enumerate(solid_tris)
        triangles[k, 1] = idxmap[T[1]]
        triangles[k, 2] = idxmap[T[2]]
        triangles[k, 3] = idxmap[T[3]]
    end

    # Boundary indices (remapped, open loop — strip the closing duplicate).
    bnd_raw = DelaunayTriangulation.get_boundary_nodes(tri)
    bnd_loop = _extract_boundary_loop(bnd_raw)
    boundary = [idxmap[v] for v in bnd_loop]

    return INLAMesh(points, triangles, boundary, tri, idxmap)
end

# Strip the outer nesting of boundary_nodes down to the first curve's
# closed-loop Int vector, then drop the repeating closing vertex.
function _extract_boundary_loop(bn)
    bn isa AbstractVector{<:Integer} && return _strip_closing_duplicate(bn)
    inner = first(bn)
    inner isa AbstractVector{<:Integer} && return _strip_closing_duplicate(inner)
    return _strip_closing_duplicate(first(inner))
end

function _strip_closing_duplicate(loop)
    if length(loop) > 1 && first(loop) == last(loop)
        return collect(@view loop[1:(end - 1)])
    end
    return collect(loop)
end

# ------------------------------------------------------------------
# Convenience constructors bridging M3 mesh → M1/M2 assemblies.
# ------------------------------------------------------------------

"""
    FEMMatrices(mesh::INLAMesh)

Assemble the FEM matrices `(C, G₁, C̃, G₂)` directly from an INLA mesh.
Forwards to [`FEMMatrices(points, triangles)`](@ref).
"""
FEMMatrices(mesh::INLAMesh) = FEMMatrices(mesh.points, mesh.triangles)

"""
    SPDE2(mesh::INLAMesh; α = 2, pc = PCMatern())

Assemble an [`SPDE2`](@ref) component directly on an INLA mesh.
"""
SPDE2(mesh::INLAMesh; α::Integer = 2, pc::PCMatern = PCMatern()) =
    SPDE2(mesh.points, mesh.triangles; α = α, pc = pc)

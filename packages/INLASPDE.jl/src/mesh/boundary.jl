"""
    convex_hull_polygon(points) -> Matrix{Float64}

Return the convex-hull polygon of `points` (`n × 2` matrix) as a
`k × 2` matrix in counter-clockwise order, without repeating the first
vertex at the end. Uses `DelaunayTriangulation.convex_hull`, which
emits CCW-ordered vertices.

Collinear points are excluded from the hull vertex set. Throws
`ArgumentError` if the hull is degenerate (fewer than three distinct
vertices).
"""
function convex_hull_polygon(points::AbstractMatrix{<:Real})
    size(points, 2) == 2 ||
        throw(ArgumentError("points must be n × 2; got size $(size(points))"))
    size(points, 1) >= 3 ||
        throw(ArgumentError("need at least 3 points for a convex hull; got $(size(points, 1))"))

    pts_tup = [(Float64(points[i, 1]), Float64(points[i, 2])) for i in axes(points, 1)]
    ch = DelaunayTriangulation.convex_hull(pts_tup)
    verts = DelaunayTriangulation.get_vertices(ch)
    # DT returns a closed loop with verts[1] == verts[end]; strip the duplicate.
    idx = @view verts[1:(end - 1)]
    length(idx) >= 3 ||
        throw(ArgumentError("convex hull is degenerate (collinear or duplicate points)"))

    P = Matrix{Float64}(undef, length(idx), 2)
    for (k, v) in enumerate(idx)
        P[k, 1], P[k, 2] = pts_tup[v]
    end
    return P
end

"""
    expand_polygon(polygon, offset) -> Matrix{Float64}

Expand a simple convex polygon outward by a perpendicular distance
`offset`. `polygon` is a `k × 2` matrix of CCW-ordered vertices (no
closing duplicate). Each vertex is shifted along the outward bisector
such that each edge of the output polygon lies `offset` away from the
corresponding input edge.

Used to construct the outer mesh domain from the convex hull of the
observation locations (R-INLA's `offset` argument to `inla.mesh.2d`).
"""
function expand_polygon(polygon::AbstractMatrix{<:Real}, offset::Real)
    size(polygon, 2) == 2 ||
        throw(ArgumentError("polygon must be k × 2; got size $(size(polygon))"))
    n = size(polygon, 1)
    n >= 3 || throw(ArgumentError("polygon needs ≥ 3 vertices; got $n"))
    offset >= 0 || throw(ArgumentError("offset must be non-negative; got $offset"))

    iszero(offset) && return Matrix{Float64}(polygon)

    P = Matrix{Float64}(undef, n, 2)
    for i in 1:n
        prev = i == 1 ? n : i - 1
        nxt = i == n ? 1 : i + 1
        pc = (Float64(polygon[i, 1]),    Float64(polygon[i, 2]))
        pp = (Float64(polygon[prev, 1]), Float64(polygon[prev, 2]))
        pn = (Float64(polygon[nxt, 1]),  Float64(polygon[nxt, 2]))

        # Outward normals of edges (pp → pc) and (pc → pn) on a CCW polygon
        # are (dy, -dx) of each edge direction.
        e_in  = (pc[1] - pp[1], pc[2] - pp[2])
        e_out = (pn[1] - pc[1], pn[2] - pc[2])
        len_in  = hypot(e_in[1],  e_in[2])
        len_out = hypot(e_out[1], e_out[2])
        (len_in > 0 && len_out > 0) ||
            throw(ArgumentError("polygon has coincident consecutive vertices"))
        n_in  = (e_in[2]  / len_in,  -e_in[1]  / len_in)
        n_out = (e_out[2] / len_out, -e_out[1] / len_out)

        # Shift along the sum-of-normals direction by the magnitude that
        # pushes each edge outward by exactly `offset`. Derivation:
        # |n_in + n_out|² = 2 (1 + n_in · n_out) = 4 cos²(α), with α the
        # half exterior angle. Correct shift is 2 · offset · (n_in + n_out)
        # / |n_in + n_out|².
        bx = n_in[1] + n_out[1]
        by = n_in[2] + n_out[2]
        bn2 = bx^2 + by^2
        bn2 > 0 ||
            throw(ArgumentError("polygon bisector is degenerate (spike or reflex corner)"))
        s = 2 * offset / bn2
        P[i, 1] = pc[1] + s * bx
        P[i, 2] = pc[2] + s * by
    end
    return P
end

"""
    cutoff_dedup(points, cutoff) -> Matrix{Float64}

Remove near-duplicate rows of `points` so that the returned set has
pairwise distance ≥ `cutoff`. First occurrence is kept (greedy). Runs
in `O(n²)` — adequate for mesh vertex counts; for 10⁶-point clouds a
KD-tree version would replace this.

`cutoff ≤ 0` returns `Matrix{Float64}(points)` unchanged.
"""
function cutoff_dedup(points::AbstractMatrix{<:Real}, cutoff::Real)
    size(points, 2) == 2 ||
        throw(ArgumentError("points must be n × 2; got size $(size(points))"))
    cutoff <= 0 && return Matrix{Float64}(points)

    kept = Int[]
    n = size(points, 1)
    for i in 1:n
        xi = Float64(points[i, 1]); yi = Float64(points[i, 2])
        ok = true
        for j in kept
            if hypot(xi - points[j, 1], yi - points[j, 2]) < cutoff
                ok = false
                break
            end
        end
        ok && push!(kept, i)
    end
    P = Matrix{Float64}(undef, length(kept), 2)
    for (k, i) in enumerate(kept)
        P[k, 1] = points[i, 1]
        P[k, 2] = points[i, 2]
    end
    return P
end

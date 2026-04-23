"""
    assemble_fem_matrices(points, triangles) -> (C, G1)

Assemble the P1 finite-element mass matrix `C` and stiffness matrix `G₁`
on a 2D triangular mesh.

The mass matrix has entries `C[i,j] = ∫_Ω φ_i φ_j dx` and the stiffness
matrix `G₁[i,j] = ∫_Ω ∇φ_i · ∇φ_j dx`, where `φ_i` is the piecewise-linear
hat basis function at vertex `i` on the mesh defined by `points` and
`triangles`.

# Arguments
- `points::AbstractMatrix{<:Real}` — `n × 2` vertex coordinates.
- `triangles::AbstractMatrix{<:Integer}` — `m × 3` one-based vertex
  indices, one row per triangle. Orientation (CW / CCW) is irrelevant;
  the unsigned triangle area is used.

# Returns
- `C::SparseMatrixCSC` — `n × n` symmetric positive-definite mass matrix.
- `G1::SparseMatrixCSC` — `n × n` symmetric positive-semidefinite
  stiffness matrix. `G₁ · 1 = 0` (constant-preserving).

# Notes
A Meshes.jl overload accepting `Meshes.SimpleMesh` is introduced in M3
alongside `inla_mesh_2d`. This low-level method is what that overload
ultimately calls.
"""
function assemble_fem_matrices(
        points::AbstractMatrix{<:Real},
        triangles::AbstractMatrix{<:Integer},
    )
    size(points, 2) == 2 ||
        throw(ArgumentError("points must be n × 2 for 2D meshes; got size $(size(points))"))
    size(triangles, 2) == 3 ||
        throw(ArgumentError("triangles must be m × 3 for P1 elements; got size $(size(triangles))"))

    n_vertices = size(points, 1)
    n_triangles = size(triangles, 1)
    T = float(eltype(points))

    nnz_upper_bound = 9 * n_triangles
    Is = Vector{Int}(undef, nnz_upper_bound)
    Js = Vector{Int}(undef, nnz_upper_bound)
    Vc = Vector{T}(undef, nnz_upper_bound)
    Vg = Vector{T}(undef, nnz_upper_bound)

    k = 0
    for t in axes(triangles, 1)
        i1 = Int(triangles[t, 1])
        i2 = Int(triangles[t, 2])
        i3 = Int(triangles[t, 3])
        (1 <= i1 <= n_vertices && 1 <= i2 <= n_vertices && 1 <= i3 <= n_vertices) ||
            throw(ArgumentError("triangle $t references vertex out of range 1:$n_vertices"))

        x1, y1 = T(points[i1, 1]), T(points[i1, 2])
        x2, y2 = T(points[i2, 1]), T(points[i2, 2])
        x3, y3 = T(points[i3, 1]), T(points[i3, 2])

        two_area_signed = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        area = abs(two_area_signed) / 2
        area > 0 ||
            throw(ArgumentError("triangle $t is degenerate (zero area)"))

        # Gradient components scaled by 2·area: ∇φ_k = (1/(2A)) · [b_k; c_k].
        b1, c1 = y2 - y3, x3 - x2
        b2, c2 = y3 - y1, x1 - x3
        b3, c3 = y1 - y2, x2 - x1
        inv_four_area = inv(4 * area)
        bs = (b1, b2, b3)
        cs = (c1, c2, c3)
        idx = (i1, i2, i3)

        mass_diag = area / 6
        mass_off = area / 12

        for a in 1:3, b in 1:3
            k += 1
            Is[k] = idx[a]
            Js[k] = idx[b]
            Vg[k] = (bs[a] * bs[b] + cs[a] * cs[b]) * inv_four_area
            Vc[k] = a == b ? mass_diag : mass_off
        end
    end

    C = sparse(Is, Js, Vc, n_vertices, n_vertices)
    G1 = sparse(Is, Js, Vg, n_vertices, n_vertices)
    return C, G1
end

"""
    INLASPDEMakieExt

Weakdep extension providing minimal Makie plot-recipe glue for
[`INLAMesh`](@ref). The approach is conservative: we register
[`MakieCore.convert_arguments`](@ref) for a handful of standard plot
primitives so that `scatter(mesh)`, `linesegments(mesh)`, and
`mesh(mesh)` work when any Makie backend (CairoMakie, GLMakie,
WGLMakie) is loaded.

Recipes deliberately live at the primitive level rather than as a
named `@recipe` block so that users may compose them with
per-primitive attributes (`color`, `linewidth`, `transparency`, …)
without having to learn a separate recipe surface.

```julia
using GLMakie, INLASPDE
mesh = inla_mesh_2d(; boundary = bnd, max_edge = 0.1, min_angle = 25.0)

# Vertices only.
scatter(mesh; markersize = 4)

# Edge wireframe.
linesegments(mesh; color = :gray)

# Filled mesh coloured by a per-vertex field.
mesh(mesh; color = field_values, colormap = :viridis)
```
"""
module INLASPDEMakieExt

using INLASPDE: INLAMesh
import MakieCore

# Vertices → `scatter(mesh)`.
#
# Returned as a Vector of 2-tuples so Makie picks up the 2D layout; in
# 3D scenes the points are lifted to Z = 0 by Makie's standard promotion.
function MakieCore.convert_arguments(::Type{<:MakieCore.Scatter}, mesh::INLAMesh)
    n = size(mesh.points, 1)
    pts = Vector{NTuple{2, Float64}}(undef, n)
    @inbounds for i in 1:n
        pts[i] = (mesh.points[i, 1], mesh.points[i, 2])
    end
    return (pts,)
end

# Triangle edges → `linesegments(mesh)` as a wireframe.
function MakieCore.convert_arguments(::Type{<:MakieCore.LineSegments}, mesh::INLAMesh)
    m = size(mesh.triangles, 1)
    segs = Vector{NTuple{2, Float64}}(undef, 6 * m)
    k = 0
    @inbounds for t in 1:m
        a = mesh.triangles[t, 1]
        b = mesh.triangles[t, 2]
        c = mesh.triangles[t, 3]
        pa = (mesh.points[a, 1], mesh.points[a, 2])
        pb = (mesh.points[b, 1], mesh.points[b, 2])
        pc = (mesh.points[c, 1], mesh.points[c, 2])
        segs[k + 1] = pa
        segs[k + 2] = pb
        segs[k + 3] = pb
        segs[k + 4] = pc
        segs[k + 5] = pc
        segs[k + 6] = pa
        k += 6
    end
    return (segs,)
end

# Filled mesh → `mesh(mesh; color = per_vertex_values)`.
#
# Makie's `Mesh` primitive accepts `(points, faces)` with points as an
# `n × D` matrix (D ∈ {2, 3}) and faces as an `m × 3` integer matrix.
# We forward the raw fields unchanged — the user-facing keyword
# `color = ...` is handled by Makie and may carry a per-vertex scalar
# field for a smooth-shaded posterior surface.
function MakieCore.convert_arguments(::Type{<:MakieCore.Mesh}, mesh::INLAMesh)
    return (mesh.points, mesh.triangles)
end

# Same primitive as `Mesh`, exposed under `Wireframe` for the
# attribute-set typical of wireframe renders.
function MakieCore.convert_arguments(::Type{<:MakieCore.Wireframe}, mesh::INLAMesh)
    return (mesh.points, mesh.triangles)
end

end # module

# Input coercion for boundary polygons and point-cloud locations.
#
# The core package handles the `Nothing` and `AbstractMatrix{<:Real}`
# cases. GeoInterface-compatible polygons, MultiPoints, and vectors of
# Points are picked up by `INLASPDEGeoInterfaceExt` (weakdep extension),
# which adds methods on the untyped `Any` slot. Keeping the core surface
# matrix-only avoids pulling GeoInterface into the mandatory dep graph.
#
# Callers (`inla_mesh_2d`, `MeshProjector`) route all inputs through
# these helpers so the public API is uniform whether or not the ext is
# loaded: a matrix works bare, a geometry works once the weakdep
# activates, and anything else surfaces as a `MethodError` naming this
# function — the idiomatic "unsupported input" signal in Julia.

"""
    _as_boundary_matrix(b) -> Matrix{Float64} | Nothing

Coerce the public `boundary` argument of [`inla_mesh_2d`](@ref) to the
internal `k × 2` matrix form.

- `nothing` passes through (boundary defaults to the convex hull of `loc`).
- `AbstractMatrix{<:Real}` is copied to `Matrix{Float64}`.
- GeoInterface `PolygonTrait` / `LineStringTrait` / `LinearRingTrait`
  geometries are handled by `INLASPDEGeoInterfaceExt` once
  `GeoInterface` is loaded.
"""
_as_boundary_matrix(::Nothing) = nothing
_as_boundary_matrix(b::AbstractMatrix{<:Real}) = Matrix{Float64}(b)

"""
    _as_location_matrix(l) -> Matrix{Float64} | Nothing

Coerce the public `loc` / `locations` arguments of [`inla_mesh_2d`](@ref)
and [`MeshProjector`](@ref) to the internal `n × 2` matrix form.

- `nothing` passes through.
- `AbstractMatrix{<:Real}` is copied to `Matrix{Float64}`.
- GeoInterface `MultiPointTrait` / `PointTrait` geometries and vectors
  of `PointTrait` geometries are handled by `INLASPDEGeoInterfaceExt`
  once `GeoInterface` is loaded.
"""
_as_location_matrix(::Nothing) = nothing
_as_location_matrix(l::AbstractMatrix{<:Real}) = Matrix{Float64}(l)

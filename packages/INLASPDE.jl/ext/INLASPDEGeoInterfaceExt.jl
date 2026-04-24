module INLASPDEGeoInterfaceExt

# Accept GeoInterface-compatible geometries as boundary / location inputs
# for `inla_mesh_2d` and `MeshProjector`. The matrix path in the core
# package is untouched; these methods route geometry inputs through the
# same `_as_*_matrix` coercion hooks. See `src/coercion.jl` for the
# core contract.

using INLASPDE: INLASPDE
using GeoInterface: GeoInterface

# --- boundary ------------------------------------------------------

function INLASPDE._as_boundary_matrix(g)
    trait = GeoInterface.geomtrait(g)
    trait === nothing && throw(ArgumentError(
        "boundary must be a k × 2 matrix or a GeoInterface geometry; " *
        "got $(typeof(g))"
    ))
    return _boundary_from_trait(trait, g)
end

# Single polygon → exterior ring. Holes (interior rings) are ignored
# with a documented warning: the single-domain Ruppert refinement this
# package exposes does not model voids.
function _boundary_from_trait(::GeoInterface.PolygonTrait, g)
    if GeoInterface.nring(g) > 1
        @warn "inla_mesh_2d: polygon has holes; interior rings are ignored"
    end
    ring = GeoInterface.getexterior(g)
    return _ring_to_matrix(ring)
end

# Closed LineString / LinearRing — strip the repeating closing vertex.
_boundary_from_trait(::GeoInterface.LineStringTrait, g)  = _ring_to_matrix(g)
_boundary_from_trait(::GeoInterface.LinearRingTrait, g)  = _ring_to_matrix(g)

_boundary_from_trait(trait, g) = throw(ArgumentError(
    "boundary geometry must have PolygonTrait or LineStringTrait; got $trait"
))

function _ring_to_matrix(ring)
    n = GeoInterface.npoint(ring)
    n ≥ 3 ||
        throw(ArgumentError("boundary ring must have at least 3 points; got $n"))
    # Drop the repeating closing vertex if present.
    p1 = GeoInterface.getpoint(ring, 1)
    pn = GeoInterface.getpoint(ring, n)
    m = (GeoInterface.x(p1) == GeoInterface.x(pn) &&
         GeoInterface.y(p1) == GeoInterface.y(pn)) ? n - 1 : n
    M = Matrix{Float64}(undef, m, 2)
    @inbounds for i in 1:m
        p = GeoInterface.getpoint(ring, i)
        M[i, 1] = GeoInterface.x(p)
        M[i, 2] = GeoInterface.y(p)
    end
    return M
end

# --- locations -----------------------------------------------------

function INLASPDE._as_location_matrix(l)
    if l isa AbstractVector
        return _points_vec_to_matrix(l)
    end
    trait = GeoInterface.geomtrait(l)
    trait === nothing && throw(ArgumentError(
        "locations must be a matrix, GeoInterface geometry, or vector of " *
        "GeoInterface points; got $(typeof(l))"
    ))
    return _locations_from_trait(trait, l)
end

function _locations_from_trait(::GeoInterface.MultiPointTrait, g)
    n = GeoInterface.ngeom(g)
    M = Matrix{Float64}(undef, n, 2)
    @inbounds for i in 1:n
        p = GeoInterface.getgeom(g, i)
        M[i, 1] = GeoInterface.x(p)
        M[i, 2] = GeoInterface.y(p)
    end
    return M
end

function _locations_from_trait(::GeoInterface.PointTrait, g)
    return reshape(Float64[GeoInterface.x(g), GeoInterface.y(g)], 1, 2)
end

_locations_from_trait(trait, g) = throw(ArgumentError(
    "locations geometry must have MultiPointTrait or PointTrait; got $trait"
))

function _points_vec_to_matrix(v::AbstractVector)
    n = length(v)
    n == 0 && return Matrix{Float64}(undef, 0, 2)
    M = Matrix{Float64}(undef, n, 2)
    @inbounds for i in eachindex(v)
        p = v[i]
        trait = GeoInterface.geomtrait(p)
        trait isa GeoInterface.PointTrait ||
            throw(ArgumentError(
                "element $i of locations vector is not a GeoInterface Point; " *
                "got $(typeof(p)) with trait $trait"
            ))
        # Index `M` by the 1-based offset into `v` so non-1-based
        # containers (OffsetArrays) still land in the right row.
        row = i - firstindex(v) + 1
        M[row, 1] = GeoInterface.x(p)
        M[row, 2] = GeoInterface.y(p)
    end
    return M
end

end # module

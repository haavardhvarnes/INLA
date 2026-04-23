"""
    MeshProjector{T, M, S}

Sparse barycentric projection from mesh-vertex values to observation
locations. For a field `u` defined on mesh vertices, the interpolated
value at observation `i` is `(A * u)[i]`, where `A` is the
`n_obs × n_vertices` sparse matrix stored in the `A` field. Each row
of `A` has at most three nonzeros: the barycentric weights of
observation `i` in its enclosing mesh triangle, which sum to 1.

# Fields

- `mesh::INLAMesh` — the source mesh.
- `locations::Matrix{T}` — `n_obs × 2` observation coordinates.
- `A::SparseMatrixCSC{T, Int}` — the sparse projection matrix.

Construct via [`MeshProjector(mesh, locations)`](@ref).

# Example

```julia
mesh = inla_mesh_2d(; boundary = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0],
                    max_edge = 0.25)
locs = rand(50, 2)
P = MeshProjector(mesh, locs)
u  = mesh.points[:, 1]              # field defined on mesh vertices
y  = P * u                          # interpolate at locations
```
"""
struct MeshProjector{T <: Real, M <: INLAMesh, S <: AbstractSparseMatrix{T, Int}}
    mesh::M
    locations::Matrix{T}
    A::S
end

"""
    MeshProjector(mesh::INLAMesh, locations; outside = :error, atol = 0.0)

Build a sparse barycentric projector from `mesh` to the points in
`locations` (`n_obs × 2` matrix).

# Keyword arguments

- `outside::Symbol = :error` — policy for locations falling outside the
  mesh domain:
    - `:error` (default) — throw `ArgumentError` on the first outside
      point.
    - `:zero` — emit an empty row (zero interpolation weight); the
      location still occupies an output row.
- `atol::Real = 0.0` — extra tolerance when classifying a location as
  outside. A point with signed barycentric coordinates ≥ `-atol` in
  its nearest triangle is accepted. Useful when observation
  coordinates lie slightly outside the numerical mesh boundary due to
  rounding.
"""
function MeshProjector(
        mesh::INLAMesh,
        locations::AbstractMatrix{<:Real};
        outside::Symbol = :error,
        atol::Real = 0.0,
    )
    size(locations, 2) == 2 ||
        throw(ArgumentError("locations must be n × 2; got size $(size(locations))"))
    outside in (:error, :zero) ||
        throw(ArgumentError("outside must be :error or :zero; got $outside"))
    atol >= 0 ||
        throw(ArgumentError("atol must be non-negative; got $atol"))

    T = promote_type(eltype(mesh.points), float(eltype(locations)))
    n_obs = size(locations, 1)
    n_v = size(mesh.points, 1)
    locs = Matrix{T}(locations)

    Is = Int[]; Js = Int[]; Vs = T[]
    sizehint!(Is, 3 * n_obs)
    sizehint!(Js, 3 * n_obs)
    sizehint!(Vs, 3 * n_obs)

    tri = mesh.triangulation
    dt2m = mesh.dt_to_mesh
    for i in 1:n_obs
        x = locs[i, 1]; y = locs[i, 2]
        Tri = DelaunayTriangulation.find_triangle(tri, (Float64(x), Float64(y)))
        if DelaunayTriangulation.is_ghost_triangle(Tri)
            outside === :error &&
                throw(ArgumentError("location $i at ($x, $y) is outside the mesh domain"))
            continue  # :zero — leave row empty
        end

        v1 = dt2m[Tri[1]]; v2 = dt2m[Tri[2]]; v3 = dt2m[Tri[3]]
        λ1, λ2, λ3 = _barycentric(
            x, y,
            mesh.points[v1, 1], mesh.points[v1, 2],
            mesh.points[v2, 1], mesh.points[v2, 2],
            mesh.points[v3, 1], mesh.points[v3, 2],
        )

        if atol > 0 && (λ1 < -atol || λ2 < -atol || λ3 < -atol)
            outside === :error &&
                throw(ArgumentError("location $i at ($x, $y) is outside the mesh domain"))
            continue
        end

        push!(Is, i); push!(Js, v1); push!(Vs, λ1)
        push!(Is, i); push!(Js, v2); push!(Vs, λ2)
        push!(Is, i); push!(Js, v3); push!(Vs, λ3)
    end

    A = sparse(Is, Js, Vs, n_obs, n_v)
    return MeshProjector{T, typeof(mesh), typeof(A)}(mesh, locs, A)
end

"""
    _barycentric(x, y, x1, y1, x2, y2, x3, y3) -> (λ1, λ2, λ3)

Barycentric coordinates of `(x, y)` with respect to the triangle
`(p1, p2, p3)`. Sums to 1 exactly by construction.
"""
function _barycentric(x, y, x1, y1, x2, y2, x3, y3)
    det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    # DT orients triangles CCW, so det > 0 for non-degenerate input.
    inv_det = inv(det)
    λ1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) * inv_det
    λ2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) * inv_det
    λ3 = 1 - λ1 - λ2
    return λ1, λ2, λ3
end

# --- matrix-like interface ----------------------------------------

Base.size(P::MeshProjector) = size(P.A)
Base.size(P::MeshProjector, d::Integer) = size(P.A, d)
Base.eltype(::Type{MeshProjector{T, M, S}}) where {T, M, S} = T
SparseArrays.sparse(P::MeshProjector) = P.A

Base.:*(P::MeshProjector, x::AbstractVector) = P.A * x
Base.:*(P::MeshProjector, X::AbstractMatrix) = P.A * X

LinearAlgebra.mul!(y::AbstractVecOrMat, P::MeshProjector, x::AbstractVecOrMat) =
    mul!(y, P.A, x)
LinearAlgebra.mul!(
    y::AbstractVecOrMat, P::MeshProjector, x::AbstractVecOrMat,
    α::Number, β::Number,
) = mul!(y, P.A, x, α, β)

function Base.show(io::IO, P::MeshProjector)
    n_obs, n_v = size(P.A)
    return print(io, "MeshProjector(", n_obs, " locations → ", n_v, " vertices)")
end

"""
    scimloperator(P::MeshProjector)

Wrap the sparse projection matrix as a `SciMLOperators.MatrixOperator`
for use with the SciMLOperators algebra (lazy adjoint / composition
with other operators).
"""
scimloperator(P::MeshProjector) = SciMLOperators.MatrixOperator(P.A)

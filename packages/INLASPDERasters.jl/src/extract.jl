"""
    extract_at_mesh(raster::Raster, mesh::INLAMesh;
                    method::Symbol = :bilinear,
                    outside::Symbol = :error,
                    missingval = NaN) -> Vector{Float64}

Sample `raster` at each vertex of `mesh` and return a length-`num_vertices`
vector of values.

# Arguments

- `raster::Raster` — a 2D raster with `X` and `Y` dimensions. The raster
  must be defined on regular, monotonically-ordered coordinates along
  both axes (ascending or descending — both are supported).
- `mesh::INLAMesh` — the SPDE mesh. Mesh vertex coordinates are assumed
  to be in the same CRS as `raster`. Mesh types currently carry no CRS
  metadata; it is the caller's responsibility to pre-project the mesh
  if needed. When CRS-aware meshes land this function will assert
  equality (see `CLAUDE.md`).

# Keywords

- `method = :bilinear` — one of `:bilinear` or `:nearest`. Bilinear
  interpolation reproduces affine raster fields exactly at mesh
  vertices. Nearest-neighbour is useful for categorical covariates.
- `outside = :error` — policy for vertices outside the raster extent:
  `:error` throws, `:missing` substitutes `missingval`.
- `missingval = NaN` — the sentinel inserted for outside-domain
  vertices when `outside = :missing`.

# Returns

`Vector{Float64}` of length `num_vertices(mesh)` with one extracted
value per vertex, ordered by vertex index.

# Notes

- For bilinear sampling, a vertex at the raster edge still has a
  well-defined value (the bracketing cell collapses to the edge).
- If the raster itself contains missing values and a bracketing cell
  has any missing corner, the returned value is `NaN` for that vertex
  under `:bilinear`, and `missingval` under `:nearest` if the nearest
  cell itself is missing.
"""
function extract_at_mesh(
        raster::Raster,
        mesh::INLAMesh;
        method::Symbol=:bilinear,
        outside::Symbol=:error,
        missingval::Real=NaN
)
    method ∈ (:bilinear, :nearest) ||
        throw(ArgumentError("method must be :bilinear or :nearest; got $method"))
    outside ∈ (:error, :missing) ||
        throw(ArgumentError("outside must be :error or :missing; got $outside"))

    xs = collect(Rasters.lookup(raster, X))
    ys = collect(Rasters.lookup(raster, Y))

    length(xs) >= 2 ||
        throw(ArgumentError("raster X dimension must have ≥ 2 points; got $(length(xs))"))
    length(ys) >= 2 ||
        throw(ArgumentError("raster Y dimension must have ≥ 2 points; got $(length(ys))"))

    n = num_vertices(mesh)
    out = Vector{Float64}(undef, n)

    for k in 1:n
        x = mesh.points[k, 1]
        y = mesh.points[k, 2]

        bx = _bracket(xs, x)
        by = _bracket(ys, y)

        if bx === nothing || by === nothing
            if outside === :error
                throw(ArgumentError(
                    "mesh vertex $k at ($x, $y) is outside the raster extent; " *
                    "pass `outside = :missing` to substitute a sentinel instead",
                ))
            else
                out[k] = missingval
                continue
            end
        end

        i, tx = bx
        j, ty = by
        out[k] = if method === :bilinear
            _bilinear_sample(raster, i, j, tx, ty)
        else
            _nearest_sample(raster, i, j, tx, ty)
        end
    end

    return out
end

# Locate the bracketing cell index `i` such that xs[i] ≤ x ≤ xs[i+1]
# (for ascending xs) and return `(i, t)` with t = (x - xs[i]) / (xs[i+1]
# - xs[i]) ∈ [0, 1]. For descending xs the same invariant holds with
# the sign flipped. Returns `nothing` when x is strictly outside the
# range of xs.
function _bracket(xs::AbstractVector{<:Real}, x::Real)
    n = length(xs)
    if xs[1] <= xs[end]
        (x < xs[1] || x > xs[end]) && return nothing
        # Ascending.
        for i in 1:(n - 1)
            a = xs[i]
            b = xs[i + 1]
            if a <= x <= b
                t = a == b ? 0.0 : (x - a) / (b - a)
                return (i, Float64(t))
            end
        end
    else
        (x > xs[1] || x < xs[end]) && return nothing
        # Descending.
        for i in 1:(n - 1)
            a = xs[i]
            b = xs[i + 1]
            if b <= x <= a
                t = a == b ? 0.0 : (a - x) / (a - b)
                return (i, Float64(t))
            end
        end
    end
    return nothing
end

# Bilinear interpolation inside the cell (i, j) ↔ (i+1, j+1).
# Keyword-dim indexing makes this independent of raster dim order.
function _bilinear_sample(raster, i, j, tx, ty)
    v00 = Float64(raster[X=i, Y=j])
    v10 = Float64(raster[X=i + 1, Y=j])
    v01 = Float64(raster[X=i, Y=j + 1])
    v11 = Float64(raster[X=i + 1, Y=j + 1])
    return (1 - tx) * (1 - ty) * v00 +
           tx * (1 - ty) * v10 +
           (1 - tx) * ty * v01 +
           tx * ty * v11
end

function _nearest_sample(raster, i, j, tx, ty)
    ii = tx < 0.5 ? i : i + 1
    jj = ty < 0.5 ? j : j + 1
    return Float64(raster[X=ii, Y=jj])
end

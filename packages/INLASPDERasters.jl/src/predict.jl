"""
    predict_raster(values::AbstractVector{<:Real},
                   mesh::INLAMesh,
                   template::Raster;
                   outside::Symbol = :missing,
                   missingval::Real = NaN) -> Raster

Project a vertex-valued field onto a raster grid matching `template`.

Builds a barycentric [`MeshProjector`](@ref) from `mesh` to the cell
centres of `template` and applies it to `values`. The returned raster
has the same dimensions, extent, resolution, and dim order as
`template`; only the underlying array is replaced.

# Arguments

- `values` — length-`num_vertices(mesh)` vector: the per-vertex field
  to project. Typically the posterior mean (or any linear functional
  of it) of an SPDE component.
- `mesh` — the SPDE mesh `values` live on.
- `template` — a 2D `Raster` with `X` and `Y` dimensions that defines
  the target grid.

# Keywords

- `outside = :missing` — policy for raster cells that fall outside the
  mesh domain: `:error` throws, `:missing` substitutes `missingval`.
- `missingval = NaN` — sentinel used when `outside = :missing`.

# Returns

A `Raster` with the same dims as `template` whose data is the
projected field. Cells outside the mesh domain carry `missingval`.
"""
function predict_raster(
        values::AbstractVector{<:Real},
        mesh::INLAMesh,
        template::Raster;
        outside::Symbol=:missing,
        missingval::Real=NaN
)
    length(values) == num_vertices(mesh) || throw(ArgumentError(
        "values has length $(length(values)) but mesh has $(num_vertices(mesh)) vertices",
    ))
    outside ∈ (:error, :missing) ||
        throw(ArgumentError("outside must be :error or :missing; got $outside"))

    xs = collect(Rasters.lookup(template, X))
    ys = collect(Rasters.lookup(template, Y))
    nx = length(xs)
    ny = length(ys)

    # Stack cell centres in a fixed (i, j) scan so we can post-index.
    n_cells = nx * ny
    locs = Matrix{Float64}(undef, n_cells, 2)
    cell_ij = Vector{Tuple{Int, Int}}(undef, n_cells)
    k = 0
    for j in 1:ny
        for i in 1:nx
            k += 1
            locs[k, 1] = xs[i]
            locs[k, 2] = ys[j]
            cell_ij[k] = (i, j)
        end
    end

    # :error wants the projector to throw on first outside cell; :missing
    # lets us mask zero rows afterwards (barycentric rows sum to 1 for
    # interior points, so an all-zero row is an unambiguous outside
    # marker under :zero).
    proj_outside = outside === :error ? :error : :zero
    P = INLASPDE.MeshProjector(mesh, locs; outside=proj_outside)
    projected = P.A * Vector{Float64}(values)

    out = similar(template, Float64)
    fill!(out, missingval)
    @inbounds for k in 1:n_cells
        i, j = cell_ij[k]
        row = @view P.A[k, :]
        if !iszero(row)
            out[X=i, Y=j] = projected[k]
        end
    end
    return out
end

"""
    quantile_rasters(mean::AbstractVector{<:Real},
                     sd::AbstractVector{<:Real},
                     mesh::INLAMesh,
                     template::Raster;
                     z::Real = 1.959963984540054,
                     outside::Symbol = :missing,
                     missingval::Real = NaN) -> NamedTuple

Project per-vertex posterior mean and standard deviation onto a raster
grid, together with symmetric Gaussian credible-interval rasters.

# Semantics

Each output raster is a linear projection of the corresponding vertex
quantity via the barycentric mesh-to-pixel projector `P`:

- `mean  = P * mean_v`
- `sd    = P * sd_v`
- `lower = P * (mean_v - z * sd_v)`
- `upper = P * (mean_v + z * sd_v)`

Projecting `sd` linearly is a convenient but approximate
"vertex-level-quantile" view: the SPDE posterior at vertices is not
diagonal, so a cell-level exact standard deviation would require
`diag(P Σ P')`. The per-vertex interval is sharp at the vertex and
interpolated linearly inside each triangle; this matches the Gaussian
summary fields callers plot on R-INLA outputs.

# Keywords

- `z = 1.96` — the Gaussian quantile half-width (default = 97.5th
  percentile, giving a 95% interval). Pass `z = 1.645` for 90%, etc.
- `outside`, `missingval` — forwarded to [`predict_raster`](@ref).

# Returns

A `NamedTuple` `(; mean, sd, lower, upper)` of four `Raster`s, all
sharing the dims of `template`.
"""
function quantile_rasters(
        mean::AbstractVector{<:Real},
        sd::AbstractVector{<:Real},
        mesh::INLAMesh,
        template::Raster;
        z::Real=1.959963984540054,
        outside::Symbol=:missing,
        missingval::Real=NaN
)
    n = num_vertices(mesh)
    length(mean) == n ||
        throw(ArgumentError("mean has length $(length(mean)) but mesh has $n vertices"))
    length(sd) == n ||
        throw(ArgumentError("sd has length $(length(sd)) but mesh has $n vertices"))
    z >= 0 ||
        throw(ArgumentError("z must be non-negative; got $z"))
    all(s -> s >= 0, sd) ||
        throw(ArgumentError("sd must be non-negative"))

    m = Vector{Float64}(mean)
    s = Vector{Float64}(sd)
    lo_v = m .- z .* s
    up_v = m .+ z .* s

    mean_r = predict_raster(m, mesh, template; outside=outside, missingval=missingval)
    sd_r = predict_raster(s, mesh, template; outside=outside, missingval=missingval)
    lower_r = predict_raster(lo_v, mesh, template; outside=outside, missingval=missingval)
    upper_r = predict_raster(up_v, mesh, template; outside=outside, missingval=missingval)

    return (mean=mean_r, sd=sd_r, lower=lower_r, upper=upper_r)
end

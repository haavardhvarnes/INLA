# Convert an R-INLA JSON dump (produced by `_helpers.R`) into a JLD2
# fixture for Julia-side oracle tests.
#
# Usage:
#   julia --project=. convert_to_jld2.jl <input.json> <output.jld2>
#
# The JSON schema is produced by `write_inla_fixture` or
# `write_gmrf_fixture` in `_helpers.R`. We deserialize sparse triplets
# back into `SparseMatrixCSC`, flatten marginals, and store the whole
# thing as a `Dict{String, Any}` under the key "fixture" in the JLD2
# file so downstream loaders have a single stable handle.

using JSON3
using JLD2
using SparseArrays

"""
    triplet_to_sparse(t) -> SparseMatrixCSC{Float64, Int}

Reconstruct a sparse matrix from the triplet representation written by
`sparse_to_triplet` in `_helpers.R`.
"""
function triplet_to_sparse(t)
    I = Int.(t["i"])
    J = Int.(t["j"])
    V = Float64.(t["v"])
    nrow = Int(t["nrow"])
    ncol = Int(t["ncol"])
    return sparse(I, J, V, nrow, ncol)
end

"""
    marginal_to_pair(m) -> NamedTuple{(:x, :y), Tuple{Vector{Float64}, Vector{Float64}}}

Convert a `{x, y}` marginal dict to a named tuple of vectors. Returns
`nothing` for null marginals.
"""
function marginal_to_pair(m)
    m === nothing && return nothing
    return (x=Float64.(m["x"]), y=Float64.(m["y"]))
end

"""
    convert_summary(s) -> Dict{String, Any}

R-INLA summary frames come in as a dict of columns plus a `rownames`
entry. Left as-is; callers index by column name.
"""
function convert_summary(s)
    s === nothing && return nothing
    # JSON3.Object -> Dict for stable Julia-side access
    return Dict{String, Any}(String(k) => _materialize(v) for (k, v) in pairs(s))
end

_materialize(v::JSON3.Array) = collect(v)
function _materialize(v::JSON3.Object)
    Dict{String, Any}(String(k) => _materialize(vv) for (k, vv) in pairs(v))
end
_materialize(v) = v

function convert_fixture(doc)
    out = Dict{String, Any}()
    out["name"] = String(doc["name"])
    haskey(doc, "inla_version") && (out["inla_version"] = String(doc["inla_version"]))
    haskey(doc, "cpu_used") && (out["cpu_used"] = Float64.(collect(doc["cpu_used"])))
    haskey(doc, "mlik") && (out["mlik"] = Float64.(collect(doc["mlik"])))

    # Summary frames
    for key in ("summary_fixed", "summary_hyperpar")
        haskey(doc, key) && (out[key] = convert_summary(doc[key]))
    end
    if haskey(doc, "summary_random")
        out["summary_random"] = Dict{String, Any}(
            String(k) => convert_summary(v) for (k, v) in pairs(doc["summary_random"])
        )
    end

    # Marginals
    for key in ("marginals_fixed", "marginals_hyperpar")
        if haskey(doc, key)
            out[key] = Dict{String, Any}(
                String(k) => marginal_to_pair(v) for (k, v) in pairs(doc[key])
            )
        end
    end
    if haskey(doc, "marginals_random")
        out["marginals_random"] = Dict{String, Any}(
            String(k) => (v === nothing ? nothing :
                          Dict{String, Any}(String(kk) => marginal_to_pair(vv)
            for (kk, vv) in pairs(v)))
        for (k, v) in pairs(doc["marginals_random"])
        )
    end

    # GMRF-style fields
    haskey(doc, "Q") && (out["Q"] = triplet_to_sparse(doc["Q"]))
    if haskey(doc, "qinv_diag") && doc["qinv_diag"] !== nothing
        out["qinv_diag"] = Float64.(collect(doc["qinv_diag"]))
    end
    if haskey(doc, "log_det") && doc["log_det"] !== nothing
        out["log_det"] = Float64(doc["log_det"])
    end

    # Metadata (free-form dict)
    if haskey(doc, "meta")
        out["meta"] = _materialize(doc["meta"])
    end

    # Input data (for fixtures that append the original observations so
    # the Julia-side oracle test can re-fit the same model without
    # needing R-side packages at test time).
    if haskey(doc, "input")
        out["input"] = _convert_input(doc["input"])
    end

    # ADR-016: oracle BYM/BYM2 latent posterior mean under R-INLA's
    # `strategy = "simplified.laplace"`. Optional, present on
    # pennsylvania_bym2; downstream tests skip transparently otherwise.
    if haskey(doc, "bym_mean_sla") && doc["bym_mean_sla"] !== nothing
        out["bym_mean_sla"] = Float64.(collect(doc["bym_mean_sla"]))
    end

    # SPDE fmesher/mesh fields. `mesh` comes from `mesh_to_list` in
    # _helpers.R; `boundary` is a list of 2-vectors; `params` is a
    # flat dict of Floats; `A_field` is a sparse triplet.
    if haskey(doc, "mesh")
        out["mesh"] = _convert_mesh(doc["mesh"])
    end
    if haskey(doc, "boundary")
        out["boundary"] = _rows_to_matrix(doc["boundary"])
    end
    if haskey(doc, "params")
        out["params"] = Dict{String, Any}(
            String(k) => Float64(v) for (k, v) in pairs(doc["params"])
        )
    end
    if haskey(doc, "A_field")
        out["A_field"] = triplet_to_sparse(doc["A_field"])
    end
    if haskey(doc, "fmesher_version")
        out["fmesher_version"] = String(doc["fmesher_version"])
    end

    return out
end

"""
    _rows_to_matrix(rows) -> Matrix{Float64}

Deserialise a list of equal-length numeric vectors (as emitted by
`boundary_to_list` and `mesh_to_list` in `_helpers.R`) into a dense
matrix whose rows are the vectors.
"""
function _rows_to_matrix(rows)
    n = length(rows)
    n == 0 && return Matrix{Float64}(undef, 0, 0)
    first_row = Float64.(collect(rows[1]))
    d = length(first_row)
    M = Matrix{Float64}(undef, n, d)
    M[1, :] = first_row
    for i in 2:n
        M[i, :] = Float64.(collect(rows[i]))
    end
    return M
end

function _rows_to_int_matrix(rows)
    n = length(rows)
    n == 0 && return Matrix{Int}(undef, 0, 0)
    first_row = Int.(collect(rows[1]))
    d = length(first_row)
    M = Matrix{Int}(undef, n, d)
    M[1, :] = first_row
    for i in 2:n
        M[i, :] = Int.(collect(rows[i]))
    end
    return M
end

function _convert_mesh(m)
    return Dict{String, Any}(
        "loc" => _rows_to_matrix(m["loc"]),
        "tv" => _rows_to_int_matrix(m["tv"]),
        "n_vertices" => Int(m["n_vertices"]),
        "n_triangles" => Int(m["n_triangles"]),
        "min_angle_deg" => Float64(m["min_angle_deg"]),
        "max_edge" => Float64(m["max_edge"])
    )
end

"""
    _convert_input(d) -> Dict{String, Any}

Materialize an input-data subdict. Sparse triplet entries are
reconstructed into `SparseMatrixCSC`; arrays are collected; scalars
pass through.
"""
function _convert_input(d)
    out = Dict{String, Any}()
    for (k, v) in pairs(d)
        out[String(k)] = _convert_input_value(v)
    end
    return out
end

function _convert_input_value(v::JSON3.Array)
    # Detect a list-of-equal-length-numeric-vectors (e.g. locations =
    # lapply(..., as.numeric) on the R side) and pack into a Matrix.
    if !isempty(v) && all(x -> x isa JSON3.Array, v)
        first_len = length(v[1])
        if all(x -> length(x) == first_len, v) &&
           all(x -> all(e -> e isa Real, x), v)
            return _rows_to_matrix(v)
        end
    end
    return collect(v)
end
function _convert_input_value(v::JSON3.Object)
    # A triplet has keys {i, j, v, nrow, ncol}; otherwise keep as dict.
    if all(k -> haskey(v, k), ("i", "j", "v", "nrow", "ncol"))
        return triplet_to_sparse(v)
    end
    return Dict{String, Any}(String(k) => _convert_input_value(vv) for (k, vv) in pairs(v))
end
_convert_input_value(v) = v

function main(args)
    length(args) == 2 || error("usage: convert_to_jld2.jl <input.json> <output.jld2>")
    in_path, out_path = args
    isfile(in_path) || error("input JSON not found: $in_path")

    doc = open(JSON3.read, in_path)
    fixture = convert_fixture(doc)

    mkpath(dirname(out_path))
    jldopen(out_path, "w") do f
        f["fixture"] = fixture
    end
    @info "wrote" output=out_path keys=collect(keys(fixture))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

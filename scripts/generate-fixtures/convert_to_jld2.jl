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
    return (x = Float64.(m["x"]), y = Float64.(m["y"]))
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
_materialize(v::JSON3.Object) = Dict{String, Any}(String(k) => _materialize(vv) for (k, vv) in pairs(v))
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
                          Dict{String, Any}(String(kk) => marginal_to_pair(vv) for (kk, vv) in pairs(v)))
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

    return out
end

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

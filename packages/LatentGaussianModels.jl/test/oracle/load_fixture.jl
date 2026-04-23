# Shared loader for R-INLA oracle fixtures in LatentGaussianModels.jl.
#
# Fixtures are JLD2 files produced by `scripts/generate-fixtures/`.
# Each fixture is stored under the key "fixture" as a `Dict{String, Any}`.
#
# This loader is a no-op if the fixture file is absent — that lets the
# oracle testset degrade to a `@test_skip` so contributors without R can
# still run the suite.

using JLD2

const ORACLE_FIXTURE_DIR = joinpath(@__DIR__, "fixtures")

"""
    oracle_fixture_path(name) -> String

Absolute path to `fixtures/<name>.jld2`.
"""
oracle_fixture_path(name::AbstractString) =
    joinpath(ORACLE_FIXTURE_DIR, string(name, ".jld2"))

"""
    has_oracle_fixture(name) -> Bool
"""
has_oracle_fixture(name::AbstractString) = isfile(oracle_fixture_path(name))

"""
    load_oracle_fixture(name) -> Dict{String, Any}

Load a fixture dict written by `scripts/generate-fixtures/convert_to_jld2.jl`.
Raises if the file is missing — callers should check `has_oracle_fixture`
first and skip.
"""
function load_oracle_fixture(name::AbstractString)
    path = oracle_fixture_path(name)
    isfile(path) || error("oracle fixture not found: $path (run scripts/generate-fixtures/)")
    return jldopen(path, "r") do f
        return f["fixture"]
    end
end

"""
    fixed_summary_mean(fx, rowname) -> Float64

Lookup helper: pull the posterior mean of a fixed effect by row name.
"""
function fixed_summary_mean(fx::AbstractDict, rowname::AbstractString)
    sf = fx["summary_fixed"]
    rn = String.(sf["rownames"])
    idx = findfirst(==(rowname), rn)
    idx === nothing && error("row '$rowname' not found in summary_fixed (have: $rn)")
    return Float64(sf["mean"][idx])
end

# Tolerance comparator for R-INLA oracle fixtures.
#
# `make all` regenerates every JLD2 fixture and writes it on top of the
# committed copy in `packages/*/test/oracle/fixtures/`. This script
# loads the committed copy (from the path given by `--committed`) and
# the regenerated copy (from the path given by `--regenerated`) and
# compares them with a numeric tolerance.
#
# We can't use `git diff --exit-code` because R-INLA + BLAS + libm
# differ between the developer machine that originally generated the
# bytes and the CI runner; the tamper check nevertheless wants to fail
# loudly on *semantic* drift (a real change in the algorithm or input
# data). The tolerances below match the project-wide oracle thresholds
# in `plans/testing-strategy.md` (1% for means, 5% for hyperparameters);
# real semantic drift swings well past these and still fails loudly.
#
# Usage:
#   julia --project compare.jl <committed_root> <regenerated_root>
#
# Both arguments are directories containing matching `<pkg>/<name>.jld2`
# files. Mismatches print to stdout; exit status is 1 if any pair
# diverges beyond tolerance, 0 otherwise.

using JLD2
using SparseArrays
using Printf

# 5% relative tolerance matches `plans/testing-strategy.md`'s
# hyperparameter oracle threshold — the loosest of the three project
# tolerances and therefore the right floor for a numeric tamper check
# that has to clear sub-1% BLAS/iterative drift across runners.
const RTOL = 5.0e-2
const ATOL = 1.0e-6

# Fields that are expected to drift by definition and should be skipped
# from the comparison. R-INLA's adaptive abscissa grids for marginal
# densities are version- and BLAS-dependent (the grid points themselves
# move; the implied densities are still validated by the summary
# statistics — mean, sd, quantiles — which live alongside).
const SKIP_KEYS = Set([
    "cpu_used",
    "marginals",
    "marginals_fixed",
    "marginals_hyperpar",
    "marginals_random",
    "marginals_linear",
])

mutable struct Diff
    path::Vector{String}
    msg::String
end

Base.show(io::IO, d::Diff) = print(io, "  ", join(d.path, " → "), ": ", d.msg)

function compare_values(a, b, path::Vector{String}, diffs::Vector{Diff})
    if a isa AbstractDict && b isa AbstractDict
        return compare_dicts(a, b, path, diffs)
    elseif (a isa Number) && (b isa Number)
        if !isapprox(a, b; rtol=RTOL, atol=ATOL)
            push!(diffs, Diff(copy(path), @sprintf("scalar drift: %g vs %g (Δ=%g)",
                a, b, abs(a - b))))
        end
        return
    elseif a isa AbstractString && b isa AbstractString
        if a != b
            push!(diffs, Diff(copy(path), "string mismatch: $(repr(a)) vs $(repr(b))"))
        end
        return
    elseif a isa AbstractArray && b isa AbstractArray
        return compare_arrays(a, b, path, diffs)
    elseif a isa NamedTuple && b isa NamedTuple
        if propertynames(a) != propertynames(b)
            push!(diffs, Diff(copy(path),
                "namedtuple keys differ: $(propertynames(a)) vs $(propertynames(b))"))
            return
        end
        for k in propertynames(a)
            push!(path, String(k))
            compare_values(getproperty(a, k), getproperty(b, k), path, diffs)
            pop!(path)
        end
        return
    elseif a isa SparseMatrixCSC && b isa SparseMatrixCSC
        if size(a) != size(b) || nnz(a) != nnz(b)
            push!(diffs, Diff(copy(path),
                "sparse shape mismatch: $(size(a)),nnz=$(nnz(a)) vs $(size(b)),nnz=$(nnz(b))"))
            return
        end
        if a.colptr != b.colptr || a.rowval != b.rowval
            push!(diffs, Diff(copy(path), "sparse pattern mismatch"))
            return
        end
        compare_arrays(a.nzval, b.nzval, push!(path, "nzval"), diffs)
        pop!(path)
        return
    elseif a === nothing && b === nothing
        return
    elseif typeof(a) != typeof(b)
        push!(diffs, Diff(copy(path),
            "type mismatch: $(typeof(a)) vs $(typeof(b))"))
        return
    else
        # Boolean, Symbol, integer that didn't hit Number branch via
        # `<:`, etc. Fall back to ==.
        if a != b
            push!(diffs, Diff(copy(path), "value mismatch: $(repr(a)) vs $(repr(b))"))
        end
        return
    end
end

function compare_arrays(a::AbstractArray, b::AbstractArray, path, diffs)
    if size(a) != size(b)
        push!(diffs, Diff(copy(path),
            "array shape: $(size(a)) vs $(size(b))"))
        return
    end
    if eltype(a) <: Number && eltype(b) <: Number
        max_abs = 0.0
        max_rel = 0.0
        bad = 0
        for i in eachindex(a)
            x, y = a[i], b[i]
            if !isapprox(x, y; rtol=RTOL, atol=ATOL)
                bad += 1
                d = abs(x - y)
                max_abs = max(max_abs, d)
                if abs(y) > 0
                    max_rel = max(max_rel, d / abs(y))
                end
            end
        end
        if bad > 0
            push!(diffs, Diff(copy(path),
                @sprintf("%d/%d entries drift beyond rtol=%g atol=%g (max abs=%g, max rel=%g)",
                    bad, length(a), RTOL, ATOL, max_abs, max_rel)))
        end
        return
    end
    # Non-numeric array: element-wise recurse.
    for i in eachindex(a)
        push!(path, "[$i]")
        compare_values(a[i], b[i], path, diffs)
        pop!(path)
    end
end

function compare_dicts(a::AbstractDict, b::AbstractDict, path, diffs)
    ka = setdiff(keys(a), SKIP_KEYS)
    kb = setdiff(keys(b), SKIP_KEYS)
    only_a = setdiff(ka, kb)
    only_b = setdiff(kb, ka)
    for k in only_a
        push!(diffs, Diff(copy(path), "key only in committed: $(repr(k))"))
    end
    for k in only_b
        push!(diffs, Diff(copy(path), "key only in regenerated: $(repr(k))"))
    end
    for k in intersect(ka, kb)
        push!(path, String(k))
        compare_values(a[k], b[k], path, diffs)
        pop!(path)
    end
end

function load_fixture(path::AbstractString)
    return jldopen(path, "r") do f
        return f["fixture"]
    end
end

function compare_fixture_files(committed::AbstractString, regenerated::AbstractString)
    a = load_fixture(committed)
    b = load_fixture(regenerated)
    diffs = Diff[]
    compare_values(a, b, String[], diffs)
    return diffs
end

function find_fixtures(root::AbstractString)
    out = String[]
    for (dir, _, files) in walkdir(root)
        for f in files
            endswith(f, ".jld2") || continue
            occursin("test/oracle/fixtures", dir) || continue
            push!(out, joinpath(dir, f))
        end
    end
    return out
end

# Map an absolute fixture path under `root` to its relative form,
# so we can pair committed and regenerated copies by relative path.
function relative_to(root::AbstractString, path::AbstractString)
    return relpath(path, root)
end

function main(args)
    length(args) == 2 ||
        error("usage: compare.jl <committed_root> <regenerated_root>")
    committed_root = abspath(args[1])
    regenerated_root = abspath(args[2])
    isdir(committed_root) || error("not a dir: $committed_root")
    isdir(regenerated_root) || error("not a dir: $regenerated_root")

    committed_files = sort(find_fixtures(committed_root))
    regenerated_files = sort(find_fixtures(regenerated_root))

    committed_rel = Set(relative_to(committed_root, p) for p in committed_files)
    regenerated_rel = Set(relative_to(regenerated_root, p) for p in regenerated_files)

    only_committed = setdiff(committed_rel, regenerated_rel)
    only_regenerated = setdiff(regenerated_rel, committed_rel)

    failed = false

    # All output goes to stdout so CI's `2>&1 | tee` produces a
    # consistently-ordered log for the step summary; exit code (not
    # stream choice) signals failure.
    if !isempty(only_committed)
        println("Committed fixtures with no regenerated counterpart:")
        for p in sort!(collect(only_committed))
            println("  ", p)
        end
        failed = true
    end
    if !isempty(only_regenerated)
        println("Regenerated fixtures not committed:")
        for p in sort!(collect(only_regenerated))
            println("  ", p)
        end
        failed = true
    end

    pairs = sort!(collect(intersect(committed_rel, regenerated_rel)))
    println("Comparing $(length(pairs)) fixtures with rtol=$RTOL, atol=$ATOL")
    for rel in pairs
        committed = joinpath(committed_root, rel)
        regenerated = joinpath(regenerated_root, rel)
        diffs = compare_fixture_files(committed, regenerated)
        if isempty(diffs)
            println("  OK   $rel")
        else
            failed = true
            println("  FAIL $rel ($(length(diffs)) diff entries)")
            for d in diffs
                println(d)
            end
        end
    end

    if failed
        println("\nFixture drift exceeds rtol=$RTOL or atol=$ATOL. Investigate above.")
        exit(1)
    else
        println("All fixtures match within tolerance.")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

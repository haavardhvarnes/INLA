"""
    SymmetricQ{T,M}

Lightweight view that tags a sparse matrix as a symmetric precision
matrix. Internally stores the *upper* triangle (Julia's `Symmetric`
convention) of a `SparseMatrixCSC{T}`; algebraic operations reconstitute
the full symmetric form on demand.

This is the analogue of R-INLA's `Qfunc`-produced symmetric Q. It is
mostly a semantic wrapper — `SparseMatrixCSC` already stores symmetric
matrices fine; this type exists so that functions that require
"precision matrix" in their signature do not accept arbitrary sparse
matrices by mistake.
"""
struct SymmetricQ{T, M <: SparseMatrixCSC{T, Int}}
    data::M   # full symmetric matrix stored explicitly (both triangles)

    function SymmetricQ{T, M}(A::M) where {T, M <: SparseMatrixCSC{T, Int}}
        n, m = size(A)
        n == m || throw(DimensionMismatch("precision matrix must be square, got $n×$m"))
        issymmetric(A) || throw(ArgumentError("precision matrix must be symmetric"))
        return new{T, M}(A)
    end
end

"""
    SymmetricQ(A::SparseMatrixCSC)

Wrap a sparse matrix as a precision matrix. `A` must be square and
symmetric.
"""
SymmetricQ(A::SparseMatrixCSC{T, Int}) where {T} = SymmetricQ{T, typeof(A)}(A)

Base.size(Q::SymmetricQ) = size(Q.data)
Base.size(Q::SymmetricQ, d::Integer) = size(Q.data, d)
Base.eltype(::Type{SymmetricQ{T, M}}) where {T, M} = T
Base.getindex(Q::SymmetricQ, i::Integer, j::Integer) = Q.data[i, j]
SparseArrays.sparse(Q::SymmetricQ) = Q.data
LinearAlgebra.issymmetric(::SymmetricQ) = true
Base.convert(::Type{SparseMatrixCSC{T, Int}}, Q::SymmetricQ{T}) where {T} = Q.data

"""
    tabulated_precision(n, qfunc; pattern_edges = Tuple{Int,Int}[])

Build a `SymmetricQ` from a scalar callback `qfunc(i, j) -> value`, in
the style of gmrflib's `GMRFLib_tabulate_Qfunc`. `qfunc` is called on
`i == j` for diagonal entries and on `(i, j)` for each listed edge
(both off-diagonal positions). The resulting matrix is symmetric.

`pattern_edges` is a collection of `(i, j)` pairs with `i < j` that
describe the off-diagonal sparsity pattern. Any pair where
`qfunc(i, j) == 0` is skipped.

This is a utility for tests and for bespoke precision structures; most
concrete GMRFs in this package build their own sparse matrix directly.
"""
function tabulated_precision(n::Integer, qfunc;
                             pattern_edges = Tuple{Int, Int}[])
    Is = Int[]; Js = Int[]; Vs = Float64[]
    # diagonal
    for i in 1:n
        v = qfunc(i, i)
        if !iszero(v)
            push!(Is, i); push!(Js, i); push!(Vs, float(v))
        end
    end
    # off-diagonal: symmetric pair
    for (i, j) in pattern_edges
        i < j || throw(ArgumentError("pattern_edges must have i<j, got ($i,$j)"))
        v = qfunc(i, j)
        if !iszero(v)
            push!(Is, i); push!(Js, j); push!(Vs, float(v))
            push!(Is, j); push!(Js, i); push!(Vs, float(v))
        end
    end
    return SymmetricQ(sparse(Is, Js, Vs, n, n))
end

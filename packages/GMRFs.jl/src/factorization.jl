# FactorCache: symbolic-factorisation reuse across θ updates.
#
# The sparsity pattern of Q is fixed while its values change with the
# hyperparameters θ. Julia's SparseArrays.cholesky returns an
# incremental factor via `cholesky!(F, A; check)`: when F was built
# from a matrix with the same pattern, only the numeric values are
# recomputed. FactorCache wraps that pattern so callers don't have to
# do the bookkeeping.
#
# We intentionally build on SparseArrays / SuiteSparse directly rather
# than LinearSolve's `init(prob)` for this first implementation —
# SuiteSparse's `cholesky!` is the exact operation we need (symbolic
# once, numeric many times) and is directly supported on sparse
# SPD matrices. The LinearSolve-backed variant is deferred until the
# LGM inference loop actually needs backend swapping.

"""
    FactorCache{F}

A reusable sparse Cholesky factor for a fixed sparsity pattern.
Construct with `FactorCache(Q)` on any symmetric-positive-definite
sparse matrix; subsequent calls to `update!(cache, Q_new)` reuse the
symbolic factorisation when `Q_new` has the same sparsity pattern as
the original `Q`.

This is the primary reuse mechanism for the inner Newton loop in the
LGM Laplace step, and for the outer θ grid in INLA.
"""
mutable struct FactorCache{F}
    F::F
end

"""
    FactorCache(Q::AbstractMatrix)

Build a Cholesky cache from a symmetric-positive-definite sparse
matrix `Q`. `Q` is promoted to a Julia-native `SparseMatrixCSC`
internally.
"""
function FactorCache(Q::AbstractMatrix)
    Qs = _sparse_spd(Q)
    F = cholesky(Symmetric(Qs))
    return FactorCache{typeof(F)}(F)
end

"""
    update!(cache::FactorCache, Q_new::AbstractMatrix)

Re-factorise in place, reusing the symbolic factorisation. `Q_new`
must have the same sparsity pattern as the matrix the cache was
constructed from. Returns `cache`.

On SuiteSparse this runs the numeric-only phase of supernodal Cholesky
(`cholmod_factorize_p`) — the symbolic phase is not repeated.
"""
function update!(cache::FactorCache, Q_new::AbstractMatrix)
    Qs = _sparse_spd(Q_new)
    cache.F = cholesky!(cache.F, Symmetric(Qs))
    return cache
end

"""
    factor(cache::FactorCache)

Return the underlying Cholesky factor.
"""
factor(cache::FactorCache) = cache.F

"""
    Base.:\\(cache::FactorCache, b::AbstractVecOrMat)

Solve `Q x = b` using the cached factor.
"""
Base.:\(cache::FactorCache, b::AbstractVecOrMat) = cache.F \ b

"""
    logdet(cache::FactorCache)

Log-determinant of the factored matrix.
"""
LinearAlgebra.logdet(cache::FactorCache) = logdet(cache.F)

# --- helpers ---------------------------------------------------------

# cholesky on sparse matrices requires SparseMatrixCSC; accept any
# AbstractMatrix here for ergonomics.
_sparse_spd(Q::SparseMatrixCSC) = Q
_sparse_spd(Q::AbstractMatrix) = SparseMatrixCSC(Q)

# Log-density of a GMRF.
#
# Proper GMRF (rank deficiency r = 0):
#   log p(x) = -½ n log(2π) + ½ log det Q - ½ (x - μ)' Q (x - μ)
#
# Intrinsic GMRF (r ≥ 1):
#   log p(x) = -½ (n - r) log(2π) + ½ log det⁺ Q - ½ (x - μ)' Q (x - μ)
# where log det⁺ Q is the log pseudodeterminant = sum of log of non-zero
# eigenvalues. For intrinsic GMRFs, this density is defined on the
# non-null subspace {x : V' x = 0}; callers that pass an x with a
# nonzero null-space component get the density as if x had been
# projected. We do *not* silently project — we warn if ||V'x||₂ is
# larger than a tolerance.

"""
    Distributions.logpdf(g::AbstractGMRF, x::AbstractVector; check_constraint = true)

Log-density of a GMRF at point `x`. For intrinsic GMRFs the density is
defined on the non-null subspace `{x : V' x = 0}`; if
`check_constraint = true` (default) we check that `V'x` is near zero
and throw a `PriorConstraintError` otherwise. Pass
`check_constraint = false` to get the formal quadratic form without
the check.
"""
function Distributions.logpdf(g::AbstractGMRF, x::AbstractVector;
                              check_constraint::Bool = true)
    n = num_nodes(g)
    length(x) == n ||
        throw(DimensionMismatch("logpdf: x length $(length(x)) ≠ num_nodes $n"))
    Q = precision_matrix(g)
    μ = prior_mean(g)
    r = rankdef(g)
    xc = x .- μ

    if r == 0
        F = cholesky(Symmetric(Q))
        logdetQ = logdet(F)
        quad = dot(xc, Q * xc)
        return -0.5 * n * log(2π) + 0.5 * logdetQ - 0.5 * quad
    else
        V = null_space_basis(g)
        if check_constraint
            proj = V' * xc
            tol = sqrt(eps()) * max(one(eltype(xc)), sqrt(sum(abs2, xc)))
            if sqrt(sum(abs2, proj)) > tol
                throw(ArgumentError("logpdf: x is not in the non-null subspace of Q " *
                                    "(||V'(x-μ)||₂ = $(sqrt(sum(abs2, proj))) > $tol). " *
                                    "Pass `check_constraint=false` to suppress."))
            end
        end
        Qp = Q + SparseMatrixCSC(V * V')
        F = cholesky(Symmetric(Qp))
        # log det⁺ Q = log det(Q + V V') since V orthonormal makes
        # log det(V'V) = 0, and the non-null eigenvalues are preserved
        # by the rank-r bump.
        logdetQ_plus = logdet(F)
        quad = dot(xc, Q * xc)
        return -0.5 * (n - r) * log(2π) + 0.5 * logdetQ_plus - 0.5 * quad
    end
end

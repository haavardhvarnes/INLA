# Marginal variances interface.
#
# v0.1 ships only the interface and a correctness-only dense reference
# implementation (see plans/plan.md M4 and ADR-004). The production
# sparse (Takahashi / selected inversion) implementation lives in
# LatentGaussianModels.jl Phase 3 where it is actually needed.

"""
    marginal_variances(g::AbstractGMRF; n_dense_limit = 1000) -> Vector{Float64}

Return the vector of marginal variances `diag(Q⁻¹)` for a proper GMRF,
or `diag(Q⁺)` on the non-null subspace for an intrinsic one. This
reference implementation uses dense matrix inversion and is *slow by
design* — it exists for correctness tests, not production.

Throws a `DomainError` when `num_nodes(g) ≥ n_dense_limit` with a
pointer to the production path:

    LatentGaussianModels.marginal_variances (Phase 3 — selected inversion)

The production implementation performs Takahashi recursion on the
sparse Cholesky factor.
"""
function marginal_variances(g::AbstractGMRF; n_dense_limit::Integer = 1000)
    n = num_nodes(g)
    if n ≥ n_dense_limit
        throw(DomainError(n,
            "marginal_variances: this GMRFs.jl v0.1 reference " *
            "implementation densifies Q and only supports n < $n_dense_limit. " *
            "For production use, call LatentGaussianModels.marginal_variances, " *
            "which uses Takahashi recursion on the sparse Cholesky factor " *
            "(see plans/defaults-parity.md, ADR-004)."))
    end
    r = rankdef(g)
    Q = Matrix(precision_matrix(g))
    if r == 0
        return diag(inv(Q))
    else
        V = null_space_basis(g)
        # Generalised inverse on non-null subspace.
        Σ = inv(Q + V * V') - V * V'
        return diag(Σ)
    end
end

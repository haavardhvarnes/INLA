# Marginal variances interface.
#
# The default sparse path is Takahashi / selected inversion via
# SelectedInversion.jl (ADR-012), resolving the ADR-004 risk. A dense
# reference path is kept behind `method = :dense` as a small-n correctness
# oracle and for rank-deficient (intrinsic) GMRFs where we augment Q with
# the null-space basis — augmentation defeats sparsity, so dense is the
# right call there.

"""
    marginal_variances(Q::AbstractSparseMatrix; method = :selinv) -> Vector{Float64}

Return `diag(Q⁻¹)` for a proper, symmetric positive-definite sparse
precision `Q`.

`method`:
- `:selinv` (default) — Takahashi recursion on the sparse Cholesky
  factor via `SelectedInversion.selinv_diag`. Scales to large sparse `Q`.
- `:dense` — densify `Q` and take a straight inverse. Correctness oracle
  only; slow.
"""
function marginal_variances(Q::AbstractSparseMatrix; method::Symbol = :selinv)
    if method === :selinv
        # Route through `cholesky(Symmetric(Q))` so CHOLMOD never sees the
        # raw SparseMatrixCSC (which it inspects with `issymmetric` — this
        # fails at floating-point-level asymmetry from assembly order).
        F = cholesky(Symmetric(SparseMatrixCSC(Q)))
        Z, p = selinv(F; depermute = false)
        d = diag(Z)
        return d[invperm(p)]
    elseif method === :dense
        return diag(inv(Symmetric(Matrix(Q))))
    else
        throw(ArgumentError("marginal_variances: unknown method :$method; " *
                            "use :selinv or :dense"))
    end
end

"""
    marginal_variances(g::AbstractGMRF; method = :auto) -> Vector{Float64}

Return the vector of marginal variances for a GMRF. For a proper GMRF
this is `diag(Q⁻¹)`; for an intrinsic (rank-deficient) GMRF it is the
generalised-inverse diagonal on the non-null subspace,
`diag(Q⁺) = diag(inv(Q + V V') - V V')`.

`method`:
- `:auto` (default) — `:selinv` for proper GMRFs, `:dense` for intrinsic.
  The intrinsic path augments `Q` with the null-space basis, which
  defeats sparsity, so dense is the honest default there.
- `:selinv` — force the sparse Takahashi path. Errors on intrinsic GMRFs
  (Q is singular).
- `:dense` — densify. Correctness oracle only.
"""
function marginal_variances(g::AbstractGMRF; method::Symbol = :auto)
    r = rankdef(g)
    Q = precision_matrix(g)
    if method === :auto
        method = r == 0 ? :selinv : :dense
    end
    if method === :selinv
        r == 0 || throw(ArgumentError(
            "marginal_variances(g; method = :selinv): GMRF is rank-deficient " *
            "(r = $r); selinv requires a PD precision. Use :auto or :dense, or " *
            "augment with the null-space basis explicitly."))
        return marginal_variances(Q; method = :selinv)
    elseif method === :dense
        Qd = Matrix(Q)
        if r == 0
            return diag(inv(Symmetric(Qd)))
        else
            V = null_space_basis(g)
            Σ = inv(Symmetric(Qd + V * V')) - V * V'
            return diag(Σ)
        end
    else
        throw(ArgumentError("marginal_variances: unknown method :$method; " *
                            "use :auto, :selinv, or :dense"))
    end
end

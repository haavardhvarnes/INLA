# Model-level hard linear constraints for the Laplace inner loop.
#
# Each component `c` exposes a per-component constraint via
# `GMRFs.constraints(c)` (either `NoConstraint()` or a
# `LinearConstraint(A_c, e_c)` on `c`'s slice of `x`). The model-level
# constraint is the block-diagonal concatenation of those rows embedded
# at the right column offsets in the stacked `x`.
#
# Constraint enforcement in the Laplace step follows Rue & Held (2005)
# §2.3: regularise the rank-deficient prior precision Q with a rank-`k`
# null-space bump `V_C V_C^T` (where `V_C = C^T (C C^T)^{-1/2}` has
# orthonormal columns spanning `range(C^T)`), run Newton on the
# regularised posterior, and apply a kriging projection at the end to
# land on `{x : C x = e}`.
#
# The contract is `null(H) ⊆ range(C^T)` where `H = Q + A' D A` is the
# unbumped Hessian — equivalently, `H_reg = Q + V_C V_C^T + A' D A`
# must be PD. This holds whenever:
#   (a) `range(C^T) = null(Q)` (the strong contract — every BYM/BYM2/
#       Besag/RW component satisfies this), or
#   (b) `null(Q) ⊋ range(C^T)` and the residual `(rd - k)` null
#       directions are identified by `A' D A` (R-INLA's `seasonal`
#       convention: 1-row C, `s - 1`-dim null space, data identifies
#       the `s - 2` unconstrained directions through the likelihood).
# The Marriott-Van Loan log-determinant `_log_det_HC` is valid in both
# regimes — it is a pure linear-algebra identity for any PD `H_reg`
# and full-rank `C`. Violations surface as a `PosDefException` on
# `cholesky(H_reg)` rather than silently wrong mlik. We fail loud here
# and revisit defensive fallbacks (full null-space bump using
# `null_space_basis(c)`) in v0.2 if a real failure appears.

"""
    model_constraints(m::LatentGaussianModel) -> AbstractConstraint

Assemble the model-level hard constraint by stacking each component's
`GMRFs.constraints(c)` block into the full `(k_total × n_x)` constraint
matrix. Returns `NoConstraint()` if no component declares a constraint.
"""
function model_constraints(m::LatentGaussianModel)
    A_blocks = Matrix{Float64}[]
    e_blocks = Vector{Float64}[]
    for (i, c) in enumerate(m.components)
        kc = GMRFs.constraints(c)
        if kc isa GMRFs.NoConstraint
            continue
        end
        A_c = GMRFs.constraint_matrix(kc)
        e_c = GMRFs.constraint_rhs(kc)
        rng = m.latent_ranges[i]
        A_full = zeros(Float64, size(A_c, 1), m.n_x)
        @views A_full[:, rng] .= A_c
        push!(A_blocks, A_full)
        push!(e_blocks, Vector{Float64}(e_c))
    end
    isempty(A_blocks) && return GMRFs.NoConstraint()
    A = reduce(vcat, A_blocks)
    e = reduce(vcat, e_blocks)
    return GMRFs.LinearConstraint(A, e)
end

"""
    _null_bump(C::AbstractMatrix) -> SparseMatrixCSC

Return `C^T (C C^T)^{-1} C` as a sparse matrix. This is
`V_C V_C^T` for the orthonormalised `V_C = C^T (C C^T)^{-1/2}` and adds
a rank-`k` bump along `range(C^T)`. PD-ness of
`Q + V_C V_C^T + A' D A` is the responsibility of the caller — if
`null(Q) ⊋ range(C^T)`, the residual `rd - k` null directions must be
covered by `A' D A` (typically true for informative observations).
"""
function _null_bump(C::AbstractMatrix)
    CCt = Symmetric(Matrix(C * C'))
    CCt_inv = inv(CCt)
    B = C' * (CCt_inv * C)
    # Symmetrise against float-level asymmetry from the matrix product.
    B = (B + B') ./ 2
    return sparse(B)
end

"""
    _apply_kriging!(x, C, e, cache) -> x

Project `x` onto `{x : C x = e}` using the factored `H_reg` in `cache`:

    x ← x - U (C U)^{-1} (C x - e),   U = H_reg^{-1} C^T.

Returns `x`. Also returns the `(U, W_fact)` pair for reuse by the
posterior-variance correction.
"""
function _kriging_correction(cache::GMRFs.FactorCache, C::AbstractMatrix)
    # U = H_reg^{-1} C^T. Since k is small (1 per connected component of
    # each intrinsic component), this is k sparse solves.
    U = cache \ Matrix(C')
    W = C * U
    W_sym = Symmetric((W + W') ./ 2)
    W_fact = cholesky(W_sym)
    return U, W_fact
end

function _project_to_constraint!(x::AbstractVector, C::AbstractMatrix,
                                 e::AbstractVector, U::AbstractMatrix, W_fact)
    Δ = U * (W_fact \ (C * x .- e))
    x .-= Δ
    return x
end

"""
    _constrained_marginal_variances(H_reg, constraint_data) -> Vector{Float64}

Per-coordinate conditional variances under the hard constraint:

    Var(x_i | y, θ, C x = e) = (H_reg^{-1})_{ii} - (U W^{-1} U^T)_{ii}

For `constraint_data === nothing`, returns `diag(H_reg^{-1})`
unchanged. `H_reg` must be PD; callers supply the regularised
posterior precision produced by `laplace_mode`.
"""
function _constrained_marginal_variances(H_reg::AbstractSparseMatrix,
                                         constraint_data)
    base = GMRFs.marginal_variances(H_reg)
    constraint_data === nothing && return base
    U = constraint_data.U
    W_fact = constraint_data.W_fact
    # diag(U W^{-1} U^T)_i = U[i,:] * (W^{-1} U^T)[:, i]
    sol = W_fact \ U'                               # k × n_x
    corr = [dot(@view(U[i, :]), @view(sol[:, i])) for i in axes(U, 1)]
    return base .- corr
end

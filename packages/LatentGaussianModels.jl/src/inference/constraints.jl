# Model-level hard linear constraints for the Laplace inner loop.
#
# Each component `c` exposes a per-component constraint via
# `GMRFs.constraints(c)` (either `NoConstraint()` or a
# `LinearConstraint(A_c, e_c)` on `c`'s slice of `x`). The model-level
# constraint is the block-diagonal concatenation of those rows embedded
# at the right column offsets in the stacked `x`.
#
# Constraint enforcement in the Laplace step follows Rue & Held (2005)
# §2.3: regularise the rank-deficient prior precision Q with the null-
# space bump `V V^T` (where `V = C^T (C C^T)^{-1/2}` has orthonormal
# columns spanning null(Q)), run Newton on the regularised posterior,
# and apply a kriging projection at the end to land on `{x : C x = e}`.
#
# The current contract assumes `null(Q) = range(C^T)` and
# `A C^T = 0` — i.e. the constraint exactly spans the unidentified
# directions that the observation projector does not see. This is
# satisfied by our BYM2 + Intercept + Poisson fits and by every
# v0.1-scope intrinsic component. Violations would surface as a
# `PosDefException` on `cholesky(Q + V V^T)` — not silently wrong.

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
`V V^T` for the orthonormalised `V = C^T (C C^T)^{-1/2}` and adds
exactly the right rank-`k` bump to make `Q + V V^T` PD whenever
`null(Q) = range(C^T)`.
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

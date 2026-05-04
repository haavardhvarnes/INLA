"""
    Laplace(; maxiter = 50, tol = 1.0e-8)

Laplace approximation to `p(x | θ, y)` at a given `θ`. Finds the mode
`x̂` by Newton iteration and returns a Gaussian with precision
`H = Q(θ) - A' diag(∇²_η log p) A` evaluated at `x̂`. The signs follow
the convention that `∇²_η log p` is negative semi-definite so that `H`
is positive definite whenever `Q` is PD on the non-null subspace.

Hard linear constraints declared by components via
`GMRFs.constraints(c)` are stacked into a model-level constraint `C x = e`
and enforced via the Rue & Held (2005) §2.3 kriging correction — the
rank-deficient prior is regularised along null(Q) with `V V^T`
(`V = C^T (C C^T)^{-1/2}`), Newton is run on the PD regularised
posterior, and the final `x̂` is projected back to `{x : C x = e}`.
"""
Base.@kwdef struct Laplace <: AbstractInferenceStrategy
    maxiter::Int = 50
    tol::Float64 = 1.0e-8
end

"""
    LaplaceResult

The output of a Laplace fit at fixed `θ`.

- `mode::Vector{Float64}` — `x̂`, on the constrained surface `C x̂ = e`.
- `precision::SparseMatrixCSC{Float64,Int}` — posterior precision at
  `x̂` on the non-null subspace, regularised with the null-space bump
  `V V^T` so that it is PD and has the same quadratic form as the
  original `H = Q + A'DA` on `null(C)`.
- `factor::FactorCache` — cached sparse Cholesky of `precision`.
- `θ::Vector{Float64}` — internal-scale hyperparameters used.
- `log_joint::Float64` — `log p(x̂, y | θ)`.
- `log_marginal::Float64` — Laplace log marginal `log p(y | θ)`.
- `iterations::Int`, `converged::Bool`.
- `constraint::Union{Nothing, NamedTuple}` — if constrained, the triple
  `(C, U, W_fact)` where `U = H_reg^{-1} C^T` and `W_fact` is the
  Cholesky of `C U`. Used downstream to subtract the kriging correction
  from conditional marginal variances.
"""
struct LaplaceResult <: AbstractInferenceResult
    mode::Vector{Float64}
    precision::SparseMatrixCSC{Float64, Int}
    factor::GMRFs.FactorCache
    θ::Vector{Float64}
    log_joint::Float64
    log_marginal::Float64
    iterations::Int
    converged::Bool
    constraint::Union{Nothing, NamedTuple}
end

"""
    laplace_mode(model, y, θ;
                 strategy = Laplace(), x0 = nothing,
                 extra_constraint = nothing) -> LaplaceResult

Find the mode `x̂` of `log p(x | θ, y) ∝ log p(y | A x, θ_ℓ) - ½ x' Q(θ) x`
by Newton iteration. Returns a `LaplaceResult`.

`extra_constraint`, if supplied, is a `NamedTuple{(:rows, :rhs)}` whose
rows are stacked onto the model-level constraint produced by
[`model_constraints`](@ref). Used by [`FullLaplace`](@ref) to fix a
single coordinate `x_i = a` for the per-`x_i` Laplace refit; any
caller-supplied linear equality is acceptable as long as the augmented
constraint matrix is full-rank.
"""
function laplace_mode(m::LatentGaussianModel, y, θ::AbstractVector{<:Real};
        strategy::Laplace=Laplace(),
        x0::Union{Nothing, AbstractVector{<:Real}}=nothing,
        extra_constraint::Union{Nothing, NamedTuple}=nothing)
    Q = joint_precision(m, θ)
    μ = joint_prior_mean(m, θ)
    A = as_matrix(m.mapping)
    # Effective Jacobian `dη/dx` for the Newton step. Equals `A` for
    # models without `Copy`; otherwise includes per-block β-rows that
    # share a latent component into another linear-predictor block. β
    # is fixed within a single Laplace fit (θ is fixed), so we
    # materialise `J` once. `A` is kept separate for the forward pass
    # `η = A * x; joint_apply_copy_contributions!(η, …)`, where the
    # Copy share enters via the `η`-hook to keep that abstraction live.
    J = joint_effective_jacobian(m, θ)

    # --- constraint bookkeeping ---------------------------------------
    # Stack any caller-supplied `extra_constraint = (rows, rhs)` onto the
    # model-level constraint. Used by `FullLaplace` (PR-3) to inject a
    # per-`x_i` equality constraint `e_i^T x = a` into the inner Newton
    # without rebuilding the model. The augmented constraint flows through
    # `_null_bump`, the kriging projection, and `_log_det_HC` unchanged.
    constraint = model_constraints(m)
    has_model_constr = !(constraint isa GMRFs.NoConstraint)
    if has_model_constr
        C_model = GMRFs.constraint_matrix(constraint)
        e_model = GMRFs.constraint_rhs(constraint)
    else
        C_model = zeros(Float64, 0, m.n_x)
        e_model = Float64[]
    end
    if extra_constraint === nothing
        C = C_model
        e_c = e_model
    else
        C = vcat(C_model, extra_constraint.rows)
        e_c = vcat(e_model, extra_constraint.rhs)
    end
    has_constr = size(C, 1) > 0
    if has_constr
        bump = _null_bump(C)
        Q_reg = Q + bump
    else
        Q_reg = Q
    end

    x = x0 === nothing ? zeros(Float64, m.n_x) : Vector{Float64}(x0)

    # Build initial posterior precision and factor cache on Q_reg.
    # `J` is the effective Jacobian `dη/dx` (= `A` for non-Copy models;
    # `A + B(β)` when a `CopyTargetLikelihood` is present), used in the
    # Hessian `Q + Jᵀ D J` and gradient `Jᵀ ∇η - Q x`.
    η = A * x
    joint_apply_copy_contributions!(η, m, x, θ)
    ∇²η = joint_∇²_η_log_density(m, y, η, θ)
    D = Diagonal(-∇²η)
    H = Q_reg + (J' * D * J)
    H = _symmetrize!(H)
    cache = GMRFs.FactorCache(H)

    converged = false
    iter = 0
    for k in 1:(strategy.maxiter)
        iter = k
        η = A * x
        joint_apply_copy_contributions!(η, m, x, θ)
        ∇η = joint_∇_η_log_density(m, y, η, θ)
        ∇²η = joint_∇²_η_log_density(m, y, η, θ)
        D = Diagonal(-∇²η)
        H = Q_reg + (J' * D * J)
        H = _symmetrize!(H)
        GMRFs.update!(cache, H)

        # Gradient of log joint w.r.t. x uses the *original* Q (not Q_reg):
        #   g = Jᵀ ∇_η log p - Q (x - μ)
        # where μ = joint_prior_mean(m, θ). For all v0.1 components μ = 0
        # and this reduces to `Jᵀ ∇_η log p - Q x`. Measurement-error
        # components (MEB/MEC, ADR-023) supply non-zero μ.
        #
        # The bump `V V^T (x - μ)` contributes to H_reg on null(Q) only — it
        # keeps Newton out of the null direction without biasing the mode on
        # the constraint surface. At C x = e the bump's contribution is
        # `C^T (CC^T)^{-1} (e - C μ)`, a fixed offset; include it on the RHS
        # to keep the Newton target consistent with the regularised quadratic.
        if has_constr
            g = J' * ∇η - Q_reg * (x .- μ)
        else
            g = J' * ∇η - Q * (x .- μ)
        end

        Δx = cache \ g
        if has_constr
            # Project Δx onto null(C) so that C(x + Δx) stays at e.
            U_step, W_step = _kriging_correction(cache, C)
            Δx .-= U_step * (W_step \ (C * Δx))
        end
        step = norm(Δx, Inf)
        x .+= Δx

        if step ≤ strategy.tol * max(1.0, norm(x, Inf))
            converged = true
            break
        end
    end

    # Final evaluation at x̂.
    η = A * x
    joint_apply_copy_contributions!(η, m, x, θ)
    ∇²η = joint_∇²_η_log_density(m, y, η, θ)
    D = Diagonal(-∇²η)
    H = Q_reg + (J' * D * J)
    H = _symmetrize!(H)
    GMRFs.update!(cache, H)

    # Final kriging correction: ensure C x̂ = e exactly at the returned
    # mode (accumulated drift from per-step projection is bounded by tol,
    # but downstream code assumes exact satisfaction).
    if has_constr
        U_final, W_final = _kriging_correction(cache, C)
        _project_to_constraint!(x, C, e_c, U_final, W_final)
        η = A * x
        joint_apply_copy_contributions!(η, m, x, θ)
        constraint_data = (C=C, e=e_c, U=U_final, W_fact=W_final)
    else
        constraint_data = nothing
    end

    # log p(x̂, y | θ) — uses the *original* Q for the latent quadratic
    # `½ (x - μ)' Q (x - μ)`. With μ = 0 this reduces to `½ x' Q x`.
    Δμ = x .- μ
    log_joint = joint_log_density(m, y, η, θ) - 0.5 * dot(Δμ, Q * Δμ)

    # R-INLA-style Laplace marginal:
    #   log p(y | θ) ≈ log p(y | x̂) - ½ x̂' Q x̂
    #                  + ½ n_x log(2π) - ½ log|H_C|
    #                  + Σ_i log_normalizing_constant(c_i, θ_i)
    # where log|H_C| = log|H| + log|C H⁻¹ C^T| - log|C C^T| is the
    # constrained-Hessian log-determinant (Marriott-Van Loan), and the
    # per-component `log_normalizing_constant` follows R-INLA's GMRFLib
    # convention. The `½ n_x log(2π)` term uses the *full* latent
    # dimension (not `n_x - r`), matching R-INLA's reference-measure
    # convention for constrained intrinsic GMRFs; for unconstrained
    # proper priors the two coincide. Each intrinsic component drops
    # the structural `½ log|R̃|_+` from its `log_normalizing_constant`,
    # absorbed into the global `log|H_C|` correction.
    log_det_HC = _log_det_HC(cache, C, has_constr)
    log_normc_total = _sum_log_normalizing_constants(m, θ)
    log_marginal = joint_log_density(m, y, η, θ) - 0.5 * dot(Δμ, Q * Δμ) +
                   0.5 * m.n_x * log(2π) - 0.5 * log_det_HC +
                   log_normc_total

    return LaplaceResult(x, H, cache, collect(θ), log_joint, log_marginal,
        iter, converged, constraint_data)
end

# Constrained-Hessian log-determinant (Marriott-Van Loan):
#   log|H_C| = log|H_reg| + log|C H_reg⁻¹ C^T| - log|C C^T|
# `cache` factors `H_reg = Q + V_C V_C^T + A' D A`. The identity is a
# pure linear-algebra fact: it holds for any PD `H_reg` and full-rank
# `C`. Parameterise `{x : C x = e}` by `x = x_p + N z` with `N` an
# orthonormal basis of `null(C)`; then `N' V_C = 0` so
# `N' H_reg N = N' H N` (the bump vanishes on the constraint surface),
# and `|N' H_reg N| · |C C^T| = |H_reg| · |C H_reg⁻¹ C^T|` is the
# standard Schur-complement determinant identity. So the formula is
# correct regardless of whether `null(Q) = range(C^T)` (strong
# contract) or `null(Q) ⊋ range(C^T)` (e.g. Seasonal with single
# sum-to-zero, data identifies remaining null directions). The
# residual null-space PD-ness is the contract's only requirement —
# documented in `inference/constraints.jl`.
function _log_det_HC(cache::GMRFs.FactorCache, C::AbstractMatrix, has_constr::Bool)
    log_det_H = logdet(cache)
    has_constr || return log_det_H
    Ct = SparseMatrixCSC{Float64, Int}(C')
    H_inv_Ct = cache \ Matrix(Ct)
    log_det_CHinvCt = logdet(Symmetric(C * H_inv_Ct))
    log_det_CCt = logdet(Symmetric(Matrix(C * C')))
    return log_det_H + log_det_CHinvCt - log_det_CCt
end

# Sum of per-component R-INLA-style log normalizing constants at θ.
function _sum_log_normalizing_constants(m::LatentGaussianModel, θ)
    s = 0.0
    for (i, c) in enumerate(m.components)
        θ_i = θ[m.θ_ranges[i]]
        s += log_normalizing_constant(c, θ_i)
    end
    return s
end

# Symmetrise a sparse matrix in place (Q + A'DA can accumulate asymmetry
# at floating-point level through multiplication order).
function _symmetrize!(H::AbstractSparseMatrix)
    return (H + H') ./ 2
end

# log|Q|_+ = logdet(Q_reg) when has_constr and Q_reg = Q + V V^T is PD.
# For proper (unconstrained) Q, Q_reg ≡ Q and this is the ordinary log-det.
# A safety ridge is applied if the cholesky still fails — indicates a
# constraint/precision mismatch that should be surfaced in later work
# rather than silently breaking the θ-mode search.
function _logdet_Q_reg(Q_reg::AbstractSparseMatrix, has_constr::Bool)
    try
        F = cholesky(Symmetric(Q_reg))
        return logdet(F)
    catch err
        if err isa LinearAlgebra.PosDefException
            n = size(Q_reg, 1)
            F = cholesky(Symmetric(Q_reg + 1.0e-8 * I(n)))
            return logdet(F)
        else
            rethrow(err)
        end
    end
end

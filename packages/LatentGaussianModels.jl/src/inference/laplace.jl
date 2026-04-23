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
    laplace_mode(model::LatentGaussianModel, y, θ; strategy = Laplace(), x0 = nothing)

Find the mode `x̂` of `log p(x | θ, y) ∝ log p(y | A x, θ_ℓ) - ½ x' Q(θ) x`
by Newton iteration. Returns a `LaplaceResult`.
"""
function laplace_mode(m::LatentGaussianModel, y, θ::AbstractVector{<:Real};
                      strategy::Laplace = Laplace(),
                      x0::Union{Nothing, AbstractVector{<:Real}} = nothing)
    Q = joint_precision(m, θ)
    A = m.A
    ℓ = m.likelihood
    n_ℓ = nhyperparameters(ℓ)
    θ_ℓ = θ[1:n_ℓ]

    # --- constraint bookkeeping ---------------------------------------
    constraint = model_constraints(m)
    has_constr = !(constraint isa GMRFs.NoConstraint)
    if has_constr
        C = GMRFs.constraint_matrix(constraint)
        e_c = GMRFs.constraint_rhs(constraint)
        bump = _null_bump(C)
        Q_reg = Q + bump
    else
        C = zeros(Float64, 0, m.n_x)
        e_c = Float64[]
        Q_reg = Q
    end

    x = x0 === nothing ? zeros(Float64, m.n_x) : Vector{Float64}(x0)

    # Build initial posterior precision and factor cache on Q_reg.
    η = A * x
    ∇²η = ∇²_η_log_density(ℓ, y, η, θ_ℓ)
    D = Diagonal(-∇²η)
    H = Q_reg + (A' * D * A)
    H = _symmetrize!(H)
    cache = GMRFs.FactorCache(H)

    converged = false
    iter = 0
    for k in 1:strategy.maxiter
        iter = k
        η = A * x
        ∇η = ∇_η_log_density(ℓ, y, η, θ_ℓ)
        ∇²η = ∇²_η_log_density(ℓ, y, η, θ_ℓ)
        D = Diagonal(-∇²η)
        H = Q_reg + (A' * D * A)
        H = _symmetrize!(H)
        GMRFs.update!(cache, H)

        # Gradient of log joint w.r.t. x uses the *original* Q (not Q_reg):
        #   g = A' ∇_η log p - Q x
        # The bump `V V^T x` contributes to H_reg on null(Q) only — it keeps
        # Newton out of the null direction without biasing the mode on the
        # constraint surface (where C x = 0 ⟹ V^T x = 0 so V V^T x = 0).
        # But we start (and project after each step) at C x = e so the bump
        # does not bias: V V^T x = C^T (CC^T)^{-1} (C x) = C^T (CC^T)^{-1} e,
        # a fixed offset. Include it in the RHS to keep the Newton target
        # consistent with the regularised quadratic.
        if has_constr
            g = A' * ∇η - Q_reg * x
        else
            g = A' * ∇η - Q * x
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
    ∇²η = ∇²_η_log_density(ℓ, y, η, θ_ℓ)
    D = Diagonal(-∇²η)
    H = Q_reg + (A' * D * A)
    H = _symmetrize!(H)
    GMRFs.update!(cache, H)

    # Final kriging correction: ensure C x̂ = e exactly at the returned
    # mode (accumulated drift from per-step projection is bounded by tol,
    # but downstream code assumes exact satisfaction).
    if has_constr
        U_final, W_final = _kriging_correction(cache, C)
        _project_to_constraint!(x, C, e_c, U_final, W_final)
        η = A * x
        constraint_data = (C = C, e = e_c, U = U_final, W_fact = W_final)
    else
        constraint_data = nothing
    end

    # log p(x̂, y | θ) — uses the *original* Q for the latent quadratic.
    log_joint = log_density(ℓ, y, η, θ_ℓ) - 0.5 * dot(x, Q * x)

    # Laplace log marginal (Rue & Held 2005 §2.3 for constrained case):
    #   log p(y | θ) ≈ log p(y | x̂) - ½ x̂' Q x̂
    #                + ½ log|Q|_+  - ½ log|H|_+  (- ½ log|C C'|  for constrained)
    # where |·|_+ is the pseudo-determinant on null(C)⊥. With V orthonormal
    # and range(V) = null(Q), log|Q|_+ = logdet(Q + V V^T) = logdet(Q_reg).
    # Likewise for H under the assumption null(H) ⊆ range(C^T).
    log_det_H = logdet(cache)                 # = logdet(H_reg); equals log|H|_+
    log_det_Q = _logdet_Q_reg(Q_reg, has_constr)
    if has_constr
        log_det_CCt = logdet(Symmetric(C * C'))
    else
        log_det_CCt = 0.0
    end
    log_marginal = log_density(ℓ, y, η, θ_ℓ) - 0.5 * dot(x, Q * x) +
                   0.5 * log_det_Q - 0.5 * log_det_H - 0.5 * log_det_CCt

    return LaplaceResult(x, H, cache, collect(θ), log_joint, log_marginal,
                         iter, converged, constraint_data)
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

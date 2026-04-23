"""
    Laplace(; maxiter = 50, tol = 1.0e-8)

Laplace approximation to `p(x | θ, y)` at a given `θ`. Finds the mode
`x̂` by Newton iteration and returns a Gaussian with precision
`H = Q(θ) - A' diag(∇²_η log p) A` evaluated at `x̂`. The signs follow
the convention that `∇²_η log p` is negative semi-definite so that `H`
is positive definite whenever `Q` is PD on the non-null subspace.

Linear constraints on the latent field are currently ignored — a hard
correction will be wired through in the integration milestone.
"""
Base.@kwdef struct Laplace <: AbstractInferenceStrategy
    maxiter::Int = 50
    tol::Float64 = 1.0e-8
end

"""
    LaplaceResult

The output of a Laplace fit at fixed `θ`.

- `mode::Vector{Float64}` — `x̂`.
- `precision::SparseMatrixCSC{Float64,Int}` — posterior precision at `x̂`.
- `factor::FactorCache` — cached sparse Cholesky of `precision`.
- `θ::Vector{Float64}` — internal-scale hyperparameters used.
- `log_joint::Float64` — `log p(x̂, y | θ)`.
- `log_marginal::Float64` — Laplace log marginal `log p(y | θ)`.
- `iterations::Int`, `converged::Bool`.
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

    x = x0 === nothing ? zeros(Float64, m.n_x) : Vector{Float64}(x0)

    # Build initial posterior precision and factor cache.
    η = A * x
    ∇²η = ∇²_η_log_density(ℓ, y, η, θ_ℓ)
    D = Diagonal(-∇²η)           # -∇²_η = D ≥ 0 (non-neg since Hessian ≤ 0)
    H = Q + (A' * D * A)
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
        H = Q + (A' * D * A)
        H = _symmetrize!(H)
        GMRFs.update!(cache, H)

        # Gradient of log joint w.r.t. x:
        #   g = A' ∇_η log p - Q x
        g = A' * ∇η - Q * x

        # Newton step: H Δx = g
        Δx = cache \ g
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
    H = Q + (A' * D * A)
    H = _symmetrize!(H)
    GMRFs.update!(cache, H)

    log_joint = log_density(ℓ, y, η, θ_ℓ) - 0.5 * dot(x, Q * x)

    # Laplace log marginal (for proper Q):
    #   log p(y | θ) ≈ log p(x̂, y | θ) + ½ log|2π H⁻¹|
    #              = log p(x̂, y | θ) + (n_x/2) log(2π) - ½ log|H|
    # Add ½ log|Q| so that x̂ term prints normalized.
    # Here we return the common form: Laplace log marginal given θ.
    log_det_H = logdet(cache)
    log_det_Q = _logdet_Q(Q)
    n_x = m.n_x
    log_marginal = log_density(ℓ, y, η, θ_ℓ) - 0.5 * dot(x, Q * x) +
                   0.5 * log_det_Q - 0.5 * log_det_H

    return LaplaceResult(x, H, cache, collect(θ), log_joint, log_marginal, iter, converged)
end

# Symmetrize a sparse matrix in place (Q + A'DA can accumulate asymmetry
# at floating-point level through multiplication order).
function _symmetrize!(H::AbstractSparseMatrix)
    return (H + H') ./ 2
end

# log|Q| for a (possibly rank-deficient) sparse PSD Q.
# For proper GMRFs this is the usual log-det; for intrinsic ones a
# pseudo-log-det on the non-null subspace is what Laplace-on-θ needs,
# but we currently punt and just compute the log-det of the provided
# block. The intrinsic case is routed through constrained Laplace in a
# later milestone; this helper is adequate for proper components.
function _logdet_Q(Q::AbstractSparseMatrix)
    try
        F = cholesky(Symmetric(Q))
        return logdet(F)
    catch err
        if err isa LinearAlgebra.PosDefException
            # Small ridge to recover a pseudo-log-det until intrinsic
            # constraint handling is wired through.
            n = size(Q, 1)
            F = cholesky(Symmetric(Q + 1.0e-8 * I(n)))
            return logdet(F)
        else
            rethrow(err)
        end
    end
end

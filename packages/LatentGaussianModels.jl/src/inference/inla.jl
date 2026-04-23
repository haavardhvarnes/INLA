"""
    INLA(; int_strategy = :auto, laplace = Laplace(), θ0 = nothing,
          optim_options = NamedTuple())

Integrated Nested Laplace Approximation for a `LatentGaussianModel`.

The algorithm (Rue, Martino & Chopin 2009) proceeds in three stages:

1. Find the mode `θ̂ = argmax_θ log π(θ | y)` using the Laplace-approximated
   log marginal and Optimization.jl.
2. Compute the Hessian `H` of the negative log-posterior at `θ̂` by finite
   differences. Let `Σ = H⁻¹`.
3. Integrate `x | y`, `θ | y`, and `x_i | y` against `π(θ | y)` using a
   design `{θ_k, w_k}` that is accurate for Gaussian integrands; reweight
   each point by `π̂(θ_k | y) / q(θ_k)` where `q ~ N(θ̂, Σ)`.

### `int_strategy`

- `:auto` (default) — `Grid()` for `dim(θ) ≤ 2`, `CCD()` otherwise.
- `:grid` / `Grid()` — tensor-product midpoint grid.
- `:ccd` / `CCD()` — central composite design per Rue-Martino-Chopin §6.5.
- `:gauss_hermite` / `GaussHermite()` — tensor-product Gauss-Hermite.
- Any subtype of `AbstractIntegrationScheme` can be passed directly.
"""
struct INLA{I, S <: Laplace} <: AbstractInferenceStrategy
    int_strategy::I
    laplace::S
    θ0::Union{Nothing, Vector{Float64}}
    optim_options::NamedTuple
end

function INLA(; int_strategy = :auto,
              laplace::Laplace = Laplace(),
              θ0::Union{Nothing, AbstractVector{<:Real}} = nothing,
              optim_options::NamedTuple = NamedTuple())
    return INLA{typeof(int_strategy), typeof(laplace)}(
        int_strategy, laplace,
        θ0 === nothing ? nothing : collect(Float64, θ0),
        optim_options
    )
end

_resolve_scheme(s::AbstractIntegrationScheme, _m::Int) = s
function _resolve_scheme(s::Symbol, m::Int)
    s === :auto && return m ≤ 2 ? Grid() : CCD()
    s === :grid && return Grid()
    s === :ccd && return CCD()
    s === :gauss_hermite && return GaussHermite()
    throw(ArgumentError("unknown int_strategy :$s; use :auto, :grid, :ccd, :gauss_hermite"))
end

"""
    INLAResult <: AbstractInferenceResult

- `θ̂::Vector{Float64}` — posterior mode of `θ | y` on the internal scale.
- `Σθ::Matrix{Float64}` — Gaussian covariance approximation at `θ̂`.
- `θ_points::Vector{Vector{Float64}}` — integration design points.
- `θ_weights::Vector{Float64}` — normalised posterior weights on the points.
- `laplaces::Vector{LaplaceResult}` — Laplace fit at each design point.
- `x_mean::Vector{Float64}` — posterior mean of `x | y`, integrated over `θ`.
- `x_var::Vector{Float64}` — posterior variance of `x | y`, integrated over `θ`.
- `θ_mean::Vector{Float64}` — posterior mean of `θ | y` (internal scale).
- `log_marginal::Float64` — `log p(y)` estimate (marginal likelihood of the model).
- `optim_result` — raw Optimization.jl solution for `θ̂`.
"""
struct INLAResult <: AbstractInferenceResult
    θ̂::Vector{Float64}
    Σθ::Matrix{Float64}
    θ_points::Vector{Vector{Float64}}
    θ_weights::Vector{Float64}
    laplaces::Vector{LaplaceResult}
    x_mean::Vector{Float64}
    x_var::Vector{Float64}
    θ_mean::Vector{Float64}
    log_marginal::Float64
    optim_result::Any
end

# ---------------------------------------------------------------------
# θ-mode + Hessian
# ---------------------------------------------------------------------

"""
    _neg_log_posterior_θ(m, y, strategy) -> function

Return a closure `(θ, _p) -> -log π(θ | y)` evaluated by Laplace. The
returned function signature matches Optimization.jl's `OptimizationFunction`.
"""
function _neg_log_posterior_θ(m::LatentGaussianModel, y, laplace::Laplace)
    return function (θ, _p)
        local res
        try
            res = laplace_mode(m, y, θ; strategy = laplace)
        catch
            return Inf
        end
        !isfinite(res.log_marginal) && return Inf
        return -(res.log_marginal + log_hyperprior(m, θ))
    end
end

"""
    _θ_mode_and_hessian(m, y, strategy) -> (θ̂, H, optim_result)

Find the θ-mode via LBFGS and compute the Hessian of the negative log
posterior at the mode by finite differences.
"""
function _θ_mode_and_hessian(m::LatentGaussianModel, y, strategy::INLA)
    θ0 = strategy.θ0 === nothing ? initial_hyperparameters(m) : copy(strategy.θ0)
    f = _neg_log_posterior_θ(m, y, strategy.laplace)
    optf = Optimization.OptimizationFunction(f, Optimization.AutoFiniteDiff())
    prob = Optimization.OptimizationProblem(optf, θ0, nothing)
    opt_res = Optimization.solve(prob, OptimizationOptimJL.LBFGS();
                                 strategy.optim_options...)
    θ̂ = collect(opt_res.u)

    # Hessian at the mode. Use FiniteDiff with a two-arg wrapper.
    f1 = θ -> f(θ, nothing)
    H = FiniteDiff.finite_difference_hessian(f1, θ̂)
    H = Symmetric((H + H') / 2)
    return θ̂, Matrix(H), opt_res
end

# ---------------------------------------------------------------------
# Main fit
# ---------------------------------------------------------------------

function fit(m::LatentGaussianModel, y, strategy::INLA)
    θ̂, H, opt_res = _θ_mode_and_hessian(m, y, strategy)

    # Σ from H; if H is not PD (near boundary) regularise with a small ridge.
    Σθ = _safe_inverse_hessian(H)

    scheme = _resolve_scheme(strategy.int_strategy, length(θ̂))
    points, log_base_weights = integration_nodes(scheme, θ̂, Σθ)

    # Laplace at each point + Gaussian-q log density at each point.
    laplaces = Vector{LaplaceResult}(undef, length(points))
    log_π = Vector{Float64}(undef, length(points))
    log_q = Vector{Float64}(undef, length(points))

    mvec = length(θ̂)
    log2π = log(2π)
    logdetΣ = logdet(Symmetric(Σθ))
    Σinv = inv(Symmetric(Σθ))

    @inbounds for k in eachindex(points)
        θ_k = points[k]
        res = laplace_mode(m, y, θ_k; strategy = strategy.laplace)
        laplaces[k] = res
        log_π[k] = res.log_marginal + log_hyperprior(m, θ_k)
        δ = θ_k .- θ̂
        log_q[k] = -0.5 * mvec * log2π - 0.5 * logdetΣ -
                   0.5 * dot(δ, Σinv, δ)
    end

    # Importance-sampling reweighting: unnormalised log-weight_k =
    #   log(base_weight_k) + log_π_k - log_q_k.
    log_unnorm = log_base_weights .+ log_π .- log_q
    log_norm = _logsumexp(log_unnorm)
    w = exp.(log_unnorm .- log_norm)

    # Marginal likelihood estimate: E_q[π̂ / q] gives Z_π where π̂ = Z_π π.
    # In logs, log Z_π ≈ logsumexp(log_base_weights + log_π - log_q).
    # (This is the standard IS estimator for the normalising constant.)
    log_marginal = log_norm

    # Aggregated posterior summaries for x.
    n_x = m.n_x
    x_mean = zeros(Float64, n_x)
    x_m2 = zeros(Float64, n_x)                 # E[x²] - cond_var(x|θ)
    for k in eachindex(points)
        lp = laplaces[k]
        x_mean .+= w[k] .* lp.mode
        # E[x²] at θ_k: mode² + conditional variance (diag of posterior
        # precision inverse, with constraint correction if present).
        # Takahashi / selected inversion via GMRFs.marginal_variances (ADR-012);
        # kriging correction applied downstream in
        # `_constrained_marginal_variances` per Rue & Held (2005) §2.3.
        cond_var = _constrained_marginal_variances(lp.precision, lp.constraint)
        x_m2 .+= w[k] .* (lp.mode .^ 2 .+ cond_var)
    end
    x_var = x_m2 .- x_mean .^ 2
    # Numerical guard against tiny negatives from cancellation.
    x_var .= max.(x_var, 0.0)

    θ_mean = zeros(Float64, mvec)
    for k in eachindex(points)
        θ_mean .+= w[k] .* points[k]
    end

    return INLAResult(θ̂, Σθ, points, w, laplaces,
                      x_mean, x_var, θ_mean, log_marginal, opt_res)
end

"""
    inla(m, y; kwargs...)

Convenience alias for `fit(m, y, INLA(; kwargs...))`.
"""
inla(m::LatentGaussianModel, y; kwargs...) = fit(m, y, INLA(; kwargs...))

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

function _safe_inverse_hessian(H::AbstractMatrix)
    Hs = Symmetric(Matrix(H))
    if isposdef(Hs)
        return Matrix(inv(Hs))
    end
    # H has non-positive eigenvalues (saddle point or flat direction at
    # the optimum — common for small-n BYM2 fits). Floor the eigenvalues
    # at a small positive number before inverting so Σθ remains PD.
    ev = eigen(Hs)
    λ = max.(ev.values, 1.0e-6)
    return Matrix(Symmetric(ev.vectors * Diagonal(1 ./ λ) * ev.vectors'))
end

function _logsumexp(x::AbstractVector{<:Real})
    m = maximum(x)
    isfinite(m) || return m
    return m + log(sum(xi -> exp(xi - m), x))
end


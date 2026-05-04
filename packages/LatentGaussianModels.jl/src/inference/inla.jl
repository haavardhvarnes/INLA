"""
    INLA(; int_strategy = :auto, laplace = Laplace(),
          latent_strategy = Gaussian(),
          skewness_correction = false,
          θ0 = nothing, optim_options = NamedTuple())

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

### `latent_strategy`

Controls the per-θ approximation of the latent posterior summary
(`x_mean`, `x_var`) — this is R-INLA's `control.inla\$strategy`,
distinct from the θ-integration scheme above.

Accepts either an `AbstractMarginalStrategy` instance or the legacy
symbols (`:gaussian`, `:simplified_laplace`). See [ADR-026].

- `Gaussian()` / `:gaussian` (default) — use the Newton mode `x̂(θ)` and
  the constraint-corrected Laplace marginal variance. Bit-for-bit
  unchanged from prior releases.
- `SimplifiedLaplace()` / `:simplified_laplace` — apply the Rue-Martino
  mean-shift correction `Δx(θ) = ½ H⁻¹ Aᵀ (h³ ⊙ σ²_η)` per integration
  point before accumulating `x_mean` / `x_var`. Reduces to `Gaussian`
  exactly when the likelihood third derivative `∇³_η log p` is zero.

The mean-shift correction is independent of the density-shape skew
correction in `posterior_marginal_x(strategy = :simplified_laplace)`:
that one expands the per-coordinate density `p(x_i | y)` around the
unshifted Newton mode (Rue-Martino-Chopin 2009 §4.2). They can be
toggled independently — see ADR-016.

R-INLA's full `simplified.laplace` also includes a variance correction;
that piece is deferred to v0.3 (see ADR-006 amendment).

### `skewness_correction`

Opt-in asymmetric per-axis stretches for the `Grid` quadrature design
(Rue-Martino-Chopin 2009 §6.5). When `true`, after `θ̂` and `Σ` are
in hand the pipeline probes `log π̂(θ|y)` at `θ̂ ± Σ^{1/2} e_k` along
each eigen-axis `k` and rebuilds the `Grid` scheme with per-axis
`stdev_corr_pos` / `stdev_corr_neg` factors that align the design
points with the actual log-posterior shape — useful when the
hyperparameter posterior is heavy-tailed on one side (e.g. log-precision
posteriors with sharp upper-curvature walls and slow decay below).

Quadrature weights stay as standard-normal weights — the asymmetric
placement only relocates points; the IS reweight in step (3) absorbs
the proposal mismatch.

The flag has no effect for non-`Grid` schemes (CCD / GaussHermite); for
those a future PR can wire equivalent corrections. Default `false`
matches R-INLA's `control.inla\$skew.corr.positive = 1.0,
skew.corr.negative = 1.0` (off by default).
"""
struct INLA{I, S <: Laplace, M <: AbstractMarginalStrategy} <: AbstractInferenceStrategy
    int_strategy::I
    laplace::S
    latent_strategy::M
    skewness_correction::Bool
    θ0::Union{Nothing, Vector{Float64}}
    optim_options::NamedTuple
end

function INLA(; int_strategy=:auto,
        laplace::Laplace=Laplace(),
        latent_strategy::Union{Symbol, AbstractMarginalStrategy}=Gaussian(),
        skewness_correction::Bool=false,
        θ0::Union{Nothing, AbstractVector{<:Real}}=nothing,
        optim_options::NamedTuple=NamedTuple())
    ls = _resolve_marginal_strategy(latent_strategy)
    return INLA{typeof(int_strategy), typeof(laplace), typeof(ls)}(
        int_strategy, laplace, ls, skewness_correction,
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

When the inner Laplace step throws (typically a `PosDefException` from
Cholesky on an ill-conditioned `Q + GᵀWG` at extreme θ) or returns a
non-finite `log_marginal`, the closure returns a smooth large penalty
`1e10 + 1e3 · ‖θ‖²` rather than `Inf`. This is load-bearing for the
outer LBFGS line search (`LineSearches.HagerZhang` asserts
`isfinite(ϕ_c)`), which can otherwise crash mid-bracket when its trial
step lands in an infeasible region — see ADR-022 / IIDND_Sep{2} where
ρ → ±1 saturation triggers it. The penalty's `1e10` floor is much
larger than any feasible `-log π(θ | y)` we have ever observed (≲ 1e6
on Phase F–I oracles), so the optimum is unaffected, and the smooth
quadratic in θ gives a usable FD gradient pointing back to the origin.
"""
function _neg_log_posterior_θ(m::LatentGaussianModel, y, laplace::Laplace)
    @inline _penalty(θ) = 1.0e10 + 1.0e3 * sum(abs2, θ)
    return function (θ, _p)
        local res
        try
            res = laplace_mode(m, y, θ; strategy=laplace)
        catch
            return _penalty(θ)
        end
        !isfinite(res.log_marginal) && return _penalty(θ)
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
    # FD-gradient noise floor at AutoFiniteDiff defaults sits around
    # √eps ≈ 1e-8, i.e. Optim.jl's default g_tol. LBFGS then exhausts the
    # 1000-iteration limit hunting noise. g_tol = 1e-4 cuts PA BYM2 from
    # 18 s → 0.36 s with Δθ̂ ≈ 1.7e-4 — well under oracle test tolerances.
    default_opts = (; g_tol = 1.0e-4)
    merged_opts = merge(default_opts, strategy.optim_options)
    opt_res = Optimization.solve(prob, OptimizationOptimJL.LBFGS();
        merged_opts...)
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
    # Fast path: nothing to integrate over. A single Laplace fit at
    # θ = Float64[] carries the full posterior. Used by ADR-024's
    # multinomial-via-Poisson reformulation, where the per-row α is a
    # fixed-precision IID nuisance (no free hyperparameters) and the
    # rest of the latent block is FixedEffects (also no
    # hyperparameters). LBFGS over a 0-dim space and the integration
    # node generators are not defined for `m = 0`, so we bypass them
    # rather than special-casing every downstream path.
    if n_hyperparameters(m) == 0
        return _fit_inla_no_hyperparameters(m, y, strategy)
    end

    θ̂, H, opt_res = _θ_mode_and_hessian(m, y, strategy)

    # Σ from H; if H is not PD (near boundary) regularise with a small ridge.
    Σθ = _safe_inverse_hessian(H)

    scheme = _resolve_scheme(strategy.int_strategy, length(θ̂))
    if strategy.skewness_correction && scheme isa Grid && length(θ̂) ≥ 1
        # Rebuild the Grid scheme with asymmetric stretches estimated
        # from a few extra `log π̂` probes around θ̂. Keeps the same
        # `n_per_dim` / `span` so the design count is unchanged; the
        # IS reweight in the loop below absorbs the proposal mismatch.
        # `_neg_log_posterior_θ` returns `-(log_marginal + log_hyperprior)`,
        # so its negation is `log π̂(θ|y)` up to a θ-independent
        # constant — fine for the skewness ratios.
        f = _neg_log_posterior_θ(m, y, strategy.laplace)
        log_post = θ -> -f(θ, nothing)
        pos, neg = compute_skewness_corrections(log_post, θ̂, Σθ)
        scheme = Grid(scheme.n_per_dim, scheme.span, pos, neg)
    end

    return _inla_integrate(m, y, θ̂, Σθ, scheme, strategy.laplace,
        strategy.latent_strategy, opt_res)
end

# ---------------------------------------------------------------------
# Integration stage: separated out so `refine_hyperposterior` can
# re-run it against an existing (θ̂, Σθ) without redoing LBFGS.
# ---------------------------------------------------------------------

function _inla_integrate(m::LatentGaussianModel, y,
        θ̂::Vector{Float64}, Σθ::Matrix{Float64},
        scheme::AbstractIntegrationScheme,
        laplace::Laplace, latent_strategy::AbstractMarginalStrategy,
        opt_result)
    points, log_base_weights = integration_nodes(scheme, θ̂, Σθ)

    # Laplace at each point + Gaussian-q log density at each point.
    laplaces = Vector{LaplaceResult}(undef, length(points))
    log_π = Vector{Float64}(undef, length(points))
    log_q = Vector{Float64}(undef, length(points))

    mvec = length(θ̂)
    log2π = log(2π)
    logdetΣ = logdet(Symmetric(Σθ))
    Σinv = inv(Symmetric(Σθ))

    # Per-point Laplace can fail in tail regions of the integration design
    # (H = Q + A'DA numerically singular for extreme θ). Mirror the mode
    # search wrapper at `_neg_log_posterior_θ`: catch the failure, drop
    # the point, and let the remaining points carry the IS sum.
    keep_mask = trues(length(points))
    @inbounds for k in eachindex(points)
        θ_k = points[k]
        local res
        try
            res = laplace_mode(m, y, θ_k; strategy=laplace)
        catch
            keep_mask[k] = false
            continue
        end
        if !isfinite(res.log_marginal)
            keep_mask[k] = false
            continue
        end
        laplaces[k] = res
        log_π[k] = res.log_marginal + log_hyperprior(m, θ_k)
        δ = θ_k .- θ̂
        log_q[k] = -0.5 * mvec * log2π - 0.5 * logdetΣ -
                   0.5 * dot(δ, Σinv, δ)
    end

    if !all(keep_mask)
        keep = findall(keep_mask)
        isempty(keep) && error("INLA: Laplace failed at every integration " *
              "point; reduce span/n_per_dim or check the model")
        n_dropped = length(points) - length(keep)
        @warn "INLA: $(n_dropped) of $(length(points)) integration " *
              "points discarded (Laplace failure or non-finite log-marginal)"
        points = points[keep]
        log_base_weights = log_base_weights[keep]
        laplaces = laplaces[keep]
        log_π = log_π[keep]
        log_q = log_q[keep]
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
        # Per-θ mean: Newton mode by default; with `SimplifiedLaplace`,
        # apply the Rue-Martino mean shift before accumulating. The
        # underlying `LaplaceResult.mode` stays untouched so downstream
        # code (Hermite skew expansion, sampling, log-marginal) keeps
        # operating around the Newton fixed point.
        mode_k = lp.mode .+ _integration_mean_shift(latent_strategy, lp, m, y)
        x_mean .+= w[k] .* mode_k
        # E[x²] at θ_k: mode² + conditional variance (diag of posterior
        # precision inverse, with constraint correction if present).
        # Takahashi / selected inversion via GMRFs.marginal_variances (ADR-012);
        # kriging correction applied downstream in
        # `_constrained_marginal_variances` per Rue & Held (2005) §2.3.
        cond_var = _constrained_marginal_variances(lp.precision, lp.constraint)
        x_m2 .+= w[k] .* (mode_k .^ 2 .+ cond_var)
    end
    x_var = x_m2 .- x_mean .^ 2
    # Numerical guard against tiny negatives from cancellation.
    x_var .= max.(x_var, 0.0)

    θ_mean = zeros(Float64, mvec)
    for k in eachindex(points)
        θ_mean .+= w[k] .* points[k]
    end

    return INLAResult(θ̂, Σθ, points, w, laplaces,
        x_mean, x_var, θ_mean, log_marginal, opt_result)
end

"""
    inla(m, y; kwargs...)

Convenience alias for `fit(m, y, INLA(; kwargs...))`.
"""
inla(m::LatentGaussianModel, y; kwargs...) = fit(m, y, INLA(; kwargs...))

# 0-hyperparameter fast path: single Laplace fit at θ = Float64[] with
# trivial INLAResult layout (Σθ is 0×0, one design point of weight 1,
# log_marginal = log p(y | θ̂) read off the Laplace fit directly).
function _fit_inla_no_hyperparameters(m::LatentGaussianModel, y, strategy::INLA)
    θ̂ = Float64[]
    Σθ = zeros(Float64, 0, 0)
    lp = laplace_mode(m, y, θ̂; strategy=strategy.laplace)

    mode = lp.mode .+ _integration_mean_shift(strategy.latent_strategy, lp, m, y)
    cond_var = _constrained_marginal_variances(lp.precision, lp.constraint)
    x_mean = mode
    x_var = cond_var

    return INLAResult(θ̂, Σθ, [θ̂], [1.0], [lp],
        x_mean, x_var, θ̂, lp.log_marginal, nothing)
end

# Integration-stage hook. Defined per-strategy here so future strategies
# (FullLaplace in PR-3) can plug in without further routing changes.
# `Gaussian` returns a zero shift; `SimplifiedLaplace` returns the
# Rue-Martino correction. The caller is responsible for the broadcast
# add against `lp.mode`.
_integration_mean_shift(::Gaussian, lp::LaplaceResult,
    m::LatentGaussianModel, y) = zero(lp.mode)

_integration_mean_shift(::SimplifiedLaplace, lp::LaplaceResult,
    m::LatentGaussianModel, y) = _sla_mean_shift(lp, m, y)

# `FullLaplace` (PR-3) only intercepts `posterior_marginal_x`; the
# integration-stage summary stays Gaussian until PR-4 wires the
# per-coordinate Laplace refit into `_inla_integrate`.
_integration_mean_shift(::FullLaplace, lp::LaplaceResult,
    m::LatentGaussianModel, y) = zero(lp.mode)

# ---------------------------------------------------------------------
# refine_hyperposterior — R-INLA `inla.hyperpar` equivalent
# ---------------------------------------------------------------------

"""
    refine_hyperposterior(res, model, y;
                          n_grid = 15, span = 4.0,
                          skewness_correction = true,
                          laplace = Laplace(),
                          latent_strategy = Gaussian())
      -> INLAResult

Re-evaluate the INLA posterior on a denser hyperparameter grid given an
existing fit. R-INLA equivalent: `inla.hyperpar`.

Reuses `res.θ̂` and `res.Σθ` from the input fit (skipping the outer
LBFGS / FD-Hessian stage) and re-runs only the integration stage of
the INLA pipeline against a fresh `Grid(n_grid, span)`. The denser
quadrature improves IS estimates of `log p(y)`, `x_mean`, `x_var`, and
`θ_mean`, and gives smoother plotted hyperparameter densities. On a
well-conditioned posterior the refined `log_marginal` should agree
with the original within IS error.

Skewness correction is on by default here (versus `inla()`'s `false`
default): the typical reason a user calls `refine_hyperposterior` is
that the hyperparameter posterior is heavy-tailed enough that the
default 5×5 design under-resolves the tails, and asymmetric per-axis
stretches are exactly the right knob for that case.

Throws `ArgumentError` if `n_hyperparameters(model) == 0` — the
fixed-θ fast path has no hyperparameter design to refine.
"""
function refine_hyperposterior(res::INLAResult,
        model::LatentGaussianModel,
        y;
        n_grid::Integer=15,
        span::Real=4.0,
        skewness_correction::Bool=true,
        laplace::Laplace=Laplace(),
        latent_strategy::Union{Symbol, AbstractMarginalStrategy}=Gaussian())
    n_hyperparameters(model) > 0 ||
        throw(ArgumentError("refine_hyperposterior: model has 0 " *
                            "hyperparameters; nothing to refine"))
    n_grid ≥ 1 || throw(ArgumentError("n_grid must be ≥ 1"))
    span > 0 || throw(ArgumentError("span must be > 0"))
    ls = _resolve_marginal_strategy(latent_strategy)

    θ̂ = res.θ̂
    Σθ = res.Σθ

    scheme = Grid(; n_per_dim=n_grid, span=Float64(span))
    if skewness_correction
        f = _neg_log_posterior_θ(model, y, laplace)
        log_post = θ -> -f(θ, nothing)
        pos, neg = compute_skewness_corrections(log_post, θ̂, Σθ)
        scheme = Grid(scheme.n_per_dim, scheme.span, pos, neg)
    end

    return _inla_integrate(model, y, θ̂, Σθ, scheme, laplace,
        ls, res.optim_result)
end

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

function _safe_inverse_hessian(H::AbstractMatrix)
    Hs = Symmetric(Matrix(H))
    # Always project to positive eigenvalues. A finite-difference Hessian
    # at a flat ridge can have small negative eigenvalues from rounding;
    # we bound the resulting variance to keep the integration grid from
    # extending into wildly extrapolated regions. The floor 1/100 caps
    # the per-axis variance at 100 (std 10 on log-precision scale), well
    # beyond any plausible posterior width while preserving informative
    # axes when the Hessian is well-conditioned.
    ev = eigen(Hs)
    λ = max.(ev.values, 1.0e-2)
    return Matrix(Symmetric(ev.vectors * Diagonal(1 ./ λ) * ev.vectors'))
end

function _logsumexp(x::AbstractVector{<:Real})
    m = maximum(x)
    isfinite(m) || return m
    return m + log(sum(xi -> exp(xi - m), x))
end

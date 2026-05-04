# Posterior marginal densities for latent coordinates and hyperparameters.
#
# For `x_i | y` the strategies live under `AbstractMarginalStrategy` (see
# ADR-026):
#
# - `Gaussian()` — mixture of per-θ Gaussians with
#   `(mean = mode_i(θ_k), var = diag(H(θ_k)⁻¹)_i)`, weighted by `w_k`.
# - `SimplifiedLaplace()` — same mixture, but each Gaussian component is
#   multiplied by an Edgeworth first-order skewness correction
#   `(1 + γ/6 · H₃(s))` where `γ = κ_3/σ³` is the posterior skewness of
#   `x_i` under the Laplace at `θ_k` and `H₃(s) = s³ - 3s` is the third
#   Hermite polynomial. This corresponds to Rue-Martino-Chopin (2009) §4.2
#   with the Gaussian marginal augmented by the leading skew term.
# - `FullLaplace()` — per-`x_i` refitted Laplace via constraint injection.
#   At each output grid point `a`, runs `laplace_mode_fixed_xi` with the
#   augmented constraint `e_i^T x = a` and reads the constrained
#   log-marginal; mixes across θ. R-INLA's `strategy = "laplace"`. See
#   `inference/full_laplace.jl`.
#
# For `θ_j | y` the first-level output is a Gaussian at (θ̂_j, Σθ[j,j]).
# A numerically integrated density on the design points is a future
# refinement once we have CCD/Grid weights in a form that factorizes per
# coordinate.

"""
    posterior_marginal_x(res::INLAResult, i::Int;
                         strategy = Gaussian(),
                         model = nothing, y = nothing,
                         grid_size = 75, span = 5.0,
                         grid = nothing) -> @NamedTuple{x::Vector, pdf::Vector}

Posterior density of the `i`-th latent coordinate, evaluated on a grid.

Returns a named tuple `(x, pdf)`. The density is the θ-mixture

    p(x_i | y) ≈ Σ_k w_k · π_k(x_i)

where `π_k` depends on `strategy` (an `AbstractMarginalStrategy`
instance, or one of the legacy symbols `:gaussian` /
`:simplified_laplace` — see ADR-026):

- `Gaussian()` / `:gaussian` (default) — `π_k = φ(x_i; mode_k[i], var_k[i])`.
- `SimplifiedLaplace()` / `:simplified_laplace` — `π_k = φ · (1 + γ_k / 6 · H₃(s))`
  with `s = (x_i - mode_k[i]) / σ_k`, `H₃(s) = s³ - 3s`, and posterior
  skewness `γ_k = κ_3(x_i|θ_k) / σ_k³`. The third cumulant is
  assembled from `∇³_η_log_density` and the Laplace precision at
  `θ_k`; for a Gaussian likelihood this collapses to the Gaussian
  strategy. Requires `model` and `y` to be supplied.
- `FullLaplace()` / `:full_laplace` — at each grid point `a` in `xs`,
  refit the joint Newton mode under the augmented constraint
  `e_i^T x = a` (via [`laplace_mode_fixed_xi`](@ref)) and read the
  constrained Laplace log-marginal. Per-θ density is the renormalised
  ratio `exp(log p̂(y | θ_k, x_i = a) - log p̂(y | θ_k))`; the mixture
  is again renormalised on the grid. Requires `model` and `y`.

If `grid` is supplied it is used verbatim; otherwise a grid of `grid_size`
equally spaced points spanning `±span · √posterior_var` about the posterior
mean is generated.
"""
function posterior_marginal_x(res::INLAResult, i::Integer;
        strategy::Union{Symbol, AbstractMarginalStrategy}=Gaussian(),
        model::Union{Nothing, LatentGaussianModel}=nothing,
        y=nothing,
        grid_size::Integer=75,
        span::Real=5.0,
        grid::Union{Nothing, AbstractVector{<:Real}}=nothing)
    1 ≤ i ≤ length(res.x_mean) ||
        throw(ArgumentError("posterior_marginal_x: index $i out of bounds (1:$(length(res.x_mean)))"))
    s = _resolve_marginal_strategy(strategy)
    if (s isa SimplifiedLaplace || s isa FullLaplace) &&
       (model === nothing || y === nothing)
        throw(ArgumentError("strategy = $(typeof(s).name.name)() requires " *
                            "keyword arguments `model` and `y`"))
    end

    μ = res.x_mean[i]
    σ = sqrt(max(res.x_var[i], 0.0))
    xs = grid === nothing ? _default_grid(μ, σ, grid_size, span) : collect(Float64, grid)

    if s isa FullLaplace
        return (x=xs, pdf=_full_laplace_pdf(res, i, model::LatentGaussianModel, y, xs))
    end

    # Per-θ conditional mean and variance (constraint-corrected).
    m_k = [lp.mode[i] for lp in res.laplaces]
    v_k = [_constrained_marginal_variances(lp.precision, lp.constraint)[i]
           for lp in res.laplaces]

    # Precompute per-θ skewness if requested. `_density_skewness` returns
    # zero for any strategy that does not need skew correction; the inner
    # loop branches on `γ == 0` so the Gaussian path is bit-for-bit
    # equivalent regardless of which strategy is requested.
    γ_k = [_density_skewness(s, res.laplaces[k], model, y, i, v_k[k])
           for k in eachindex(res.laplaces)]

    pdf = zeros(Float64, length(xs))
    @inbounds for k in eachindex(res.laplaces)
        w = res.θ_weights[k]
        w == 0 && continue
        σk = sqrt(max(v_k[k], 0.0))
        σk == 0 && continue
        γ = γ_k[k]
        if γ == 0
            for (j, x) in pairs(xs)
                pdf[j] += w * _normal_pdf(x, m_k[k], σk)
            end
        else
            for (j, x) in pairs(xs)
                t = (x - m_k[k]) / σk
                # Edgeworth first-order skewness correction.
                # H_3(t) = t^3 - 3t.  ∫ φ(t) H_3(t) dt = 0 ⇒ density
                # integrates to 1 without renormalisation.
                c = 1 + γ / 6 * (t^3 - 3 * t)
                # Floor to zero — Edgeworth densities can go slightly
                # negative in the deep tails when |γ| is large. Clamping
                # preserves non-negativity without destroying mass.
                c = max(c, 0.0)
                pdf[j] += w * c * _normal_pdf(x, m_k[k], σk)
            end
        end
    end
    return (x=xs, pdf=pdf)
end

# Per-coordinate density-skew hook. `Gaussian` returns 0 (collapses to
# the unweighted Gaussian mixture). `SimplifiedLaplace` returns the
# Edgeworth coefficient assembled from `∇³_η_log_density`.
_density_skewness(::Gaussian, lp::LaplaceResult, model, y, i, var_i) = 0.0

function _density_skewness(::SimplifiedLaplace,
        lp::LaplaceResult,
        model::LatentGaussianModel,
        y,
        i::Integer,
        var_i::Real)
    return _latent_skewness(lp, model, y, i, var_i)
end

"""
    posterior_marginal_θ(res::INLAResult, j::Int;
                         grid_size = 75, span = 5.0,
                         grid = nothing) -> @NamedTuple{θ::Vector, pdf::Vector}

Gaussian marginal of the `j`-th hyperparameter on the internal scale,
centred at `res.θ̂[j]` with standard deviation `√res.Σθ[j,j]`. This is the
"gaussian" strategy in R-INLA. A numerically integrated density over the
INLA design is a future extension.
"""
function posterior_marginal_θ(res::INLAResult, j::Integer;
        grid_size::Integer=75,
        span::Real=5.0,
        grid::Union{Nothing, AbstractVector{<:Real}}=nothing)
    1 ≤ j ≤ length(res.θ̂) ||
        throw(ArgumentError("posterior_marginal_θ: index $j out of bounds (1:$(length(res.θ̂)))"))

    μ = res.θ̂[j]
    σ = sqrt(max(res.Σθ[j, j], 0.0))
    θs = grid === nothing ? _default_grid(μ, σ, grid_size, span) : collect(Float64, grid)
    pdf = [_normal_pdf(θ, μ, σ) for θ in θs]
    return (θ=θs, pdf=pdf)
end

function _default_grid(μ::Real, σ::Real, n::Integer, span::Real)
    σe = σ > 0 ? σ : 1.0
    lo = μ - span * σe
    hi = μ + span * σe
    return collect(range(lo, hi; length=n))
end

_normal_pdf(x::Real, μ::Real, σ::Real) = exp(-0.5 * ((x - μ) / σ)^2) / (σ * sqrt(2π))

# Posterior skewness of x_i under the Laplace at θ (including constraint
# correction). Returns `γ = κ_3(x_i) / σ_i³`.
#
# Derivation: log posterior = log π(x|θ) + log p(y|A x, θ_ℓ). The prior is
# Gaussian so its third derivative is zero. The likelihood factorises over
# observations with η_j = (A x)_j, so
#
#   ∂³ log p / ∂x_a ∂x_b ∂x_c = Σ_j c³_j · A_{ja} A_{jb} A_{jc}
#
# where c³_j = ∇³_η log p evaluated at η̂_j. Contracting with H^{-1} e_i in
# each slot (H^{-1} is the posterior precision inverse, constraint-corrected):
#
#   κ_3(x_i) = Σ_j c³_j · (A u_i)_j³   with   u_i = H^{-1} e_i (constrained)
#
# and σ_i² = (u_i)_i = constraint-corrected marginal variance.
function _latent_skewness(lp::LaplaceResult,
        model::LatentGaussianModel,
        y,
        i::Integer,
        var_i::Real)
    σ_i = sqrt(max(var_i, 0.0))
    σ_i == 0 && return 0.0

    A = as_matrix(model.mapping)
    # `J` is the effective Jacobian `dη/dx` — equals `A` for non-Copy
    # models; otherwise includes per-block β-rows. The skewness contracts
    # `c³` with `(J u)_j³`, so the Copy contribution must enter J here.
    J = joint_effective_jacobian(model, lp.θ)
    η̂ = A * lp.mode
    joint_apply_copy_contributions!(η̂, model, lp.mode, lp.θ)
    c³ = joint_∇³_η_log_density(model, y, η̂, lp.θ)
    all(iszero, c³) && return 0.0

    # u = H⁻¹ e_i (unconstrained). Sparse triangular solve against a unit
    # RHS — one forward + one back substitution.
    e_i = zeros(Float64, length(lp.mode))
    e_i[i] = 1.0
    u = lp.factor \ e_i

    if lp.constraint !== nothing
        U = lp.constraint.U
        W_fact = lp.constraint.W_fact
        C = lp.constraint.C
        u .-= U * (W_fact \ (C * u))
    end

    Ju = J * u
    κ3 = zero(Float64)
    @inbounds for j in eachindex(c³)
        κ3 += c³[j] * Ju[j]^3
    end
    return κ3 / σ_i^3
end

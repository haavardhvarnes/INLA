# Posterior marginal densities for latent coordinates and hyperparameters.
#
# For `x_i | y` we use the standard INLA approximation: the posterior is a
# mixture of the per-θ Gaussian Laplaces, weighted by the importance-sampling
# weights `w_k`. The v0.1 implementation uses the conditional marginal
# (mean = mode_i(θ_k), var = diag(H(θ_k)⁻¹)_i) — i.e. the "Gaussian" strategy
# in R-INLA terminology. Simplified Laplace / Laplace skew corrections are
# Phase-3 work; see ADR-010 and `plans/defaults-parity.md`.
#
# For `θ_j | y` the first-level output is a Gaussian at (θ̂_j, Σθ[j,j]).
# A numerically integrated density on the design points is a future
# refinement once we have CCD/Grid weights in a form that factorizes per
# coordinate.

"""
    posterior_marginal_x(res::INLAResult, i::Int;
                         grid_size = 75, span = 5.0,
                         grid = nothing) -> @NamedTuple{x::Vector, pdf::Vector}

Posterior density of the `i`-th latent coordinate, evaluated on a grid.

Returns a named tuple `(x, pdf)`. The density is the θ-mixture

    p(x_i | y) ≈ Σ_k w_k · φ(x_i; mode_k[i], var_k[i])

where `mode_k` and `var_k[i]` are the Laplace mode and conditional variance
at integration point `θ_k`, and `w_k` are the INLA weights.

If `grid` is supplied it is used verbatim; otherwise a grid of `grid_size`
equally spaced points spanning `±span · √posterior_var` about the posterior
mean is generated.
"""
function posterior_marginal_x(res::INLAResult, i::Integer;
                              grid_size::Integer = 75,
                              span::Real = 5.0,
                              grid::Union{Nothing, AbstractVector{<:Real}} = nothing)
    1 ≤ i ≤ length(res.x_mean) ||
        throw(ArgumentError("posterior_marginal_x: index $i out of bounds (1:$(length(res.x_mean)))"))

    μ = res.x_mean[i]
    σ = sqrt(max(res.x_var[i], 0.0))
    xs = grid === nothing ? _default_grid(μ, σ, grid_size, span) : collect(Float64, grid)

    # Per-θ conditional mean and variance (constraint-corrected).
    m_k = [lp.mode[i] for lp in res.laplaces]
    v_k = [_constrained_marginal_variances(lp.precision, lp.constraint)[i]
           for lp in res.laplaces]

    pdf = zeros(Float64, length(xs))
    @inbounds for k in eachindex(res.laplaces)
        w = res.θ_weights[k]
        w == 0 && continue
        σk = sqrt(max(v_k[k], 0.0))
        σk == 0 && continue
        for (j, x) in pairs(xs)
            pdf[j] += w * _normal_pdf(x, m_k[k], σk)
        end
    end
    return (x = xs, pdf = pdf)
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
                              grid_size::Integer = 75,
                              span::Real = 5.0,
                              grid::Union{Nothing, AbstractVector{<:Real}} = nothing)
    1 ≤ j ≤ length(res.θ̂) ||
        throw(ArgumentError("posterior_marginal_θ: index $j out of bounds (1:$(length(res.θ̂)))"))

    μ = res.θ̂[j]
    σ = sqrt(max(res.Σθ[j, j], 0.0))
    θs = grid === nothing ? _default_grid(μ, σ, grid_size, span) : collect(Float64, grid)
    pdf = [_normal_pdf(θ, μ, σ) for θ in θs]
    return (θ = θs, pdf = pdf)
end

function _default_grid(μ::Real, σ::Real, n::Integer, span::Real)
    σe = σ > 0 ? σ : 1.0
    lo = μ - span * σe
    hi = μ + span * σe
    return collect(range(lo, hi; length = n))
end

_normal_pdf(x::Real, μ::Real, σ::Real) =
    exp(-0.5 * ((x - μ) / σ)^2) / (σ * sqrt(2π))

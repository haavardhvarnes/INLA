"""
    PCGevtail(λ = 7.0, interval = (0.0, 1.0); xi_scale = 0.1)

Penalised-complexity prior on the GEV tail (shape) parameter
`ξ ∈ interval` (Opitz et al. 2018; R-INLA's `pc.gevtail`). The
reference value is `ξ = 0` (Gumbel), and the prior takes the
linearised PC distance `d(ξ) = ξ` exact for `ξ ≥ 0` close to the
reference. The user-scale density is the truncated exponential

    π_ξ(ξ) = λ exp(-λ ξ) / Z,   ξ ∈ [low, high],
    Z      = exp(-λ low) - exp(-λ high),

matching `inla.pc.dgevtail(xi, lambda, interval)` in R-INLA.

The internal scale matches [`GEVLikelihood`](@ref):
`θ = ξ / xi_scale`, with `|dξ/dθ| = xi_scale`. On the internal
scale the log-density is

    log π_θ(θ) = log λ - λ · xi_scale · θ - log Z + log(xi_scale),

returning `-Inf` outside `[low / xi_scale, high / xi_scale]`.

Defaults `λ = 7, interval = (0, 1), xi_scale = 0.1` match R-INLA's
`pc.gevtail` defaults composed with `GEVLikelihood`'s default
`xi_scale = 0.1` (R-INLA's `gev.scale.xi`).

# Example

```julia
hyper = PCGevtail(7.0, (0.0, 1.0); xi_scale = 0.1)
log_prior_density(hyper, 0.5)   # at θ = 0.5 ⇒ ξ = 0.05
```
"""
struct PCGevtail{T <: Real} <: AbstractHyperPrior
    λ::T
    low::T
    high::T
    xi_scale::T
    log_Z::T   # log(exp(-λ low) - exp(-λ high))

    function PCGevtail{T}(λ::T, low::T, high::T, xi_scale::T) where {T <: Real}
        λ > 0 ||
            throw(ArgumentError("PCGevtail: λ must be > 0, got $λ"))
        low < high || throw(ArgumentError(
            "PCGevtail: interval must be (low, high) with low < high; " *
            "got ($low, $high)"))
        low >= 0 || throw(ArgumentError(
            "PCGevtail: interval lower bound must be >= 0 (R-INLA's " *
            "`pc.gevtail` is one-sided on ξ ≥ 0); got low = $low"))
        xi_scale > 0 ||
            throw(ArgumentError("PCGevtail: xi_scale must be > 0, got $xi_scale"))
        log_Z = log(exp(-λ * low) - exp(-λ * high))
        return new{T}(λ, low, high, xi_scale, log_Z)
    end
end
function PCGevtail(λ::Real=7.0,
        interval::Tuple{<:Real, <:Real}=(0.0, 1.0);
        xi_scale::Real=0.1)
    T = typeof(float(λ * interval[1] * interval[2] * xi_scale))
    return PCGevtail{T}(T(λ), T(interval[1]), T(interval[2]), T(xi_scale))
end

prior_name(::PCGevtail) = :pc_gevtail
user_scale(p::PCGevtail, θ) = p.xi_scale * θ

function log_prior_density(p::PCGevtail{T}, θ) where {T}
    ξ = p.xi_scale * θ
    if ξ < p.low || ξ > p.high
        return typemin(promote_type(T, typeof(float(θ))))
    end
    return log(p.λ) - p.λ * ξ - p.log_Z + log(p.xi_scale)
end

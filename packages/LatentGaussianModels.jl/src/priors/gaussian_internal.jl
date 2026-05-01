"""
    GaussianPrior(μ, σ)

A `Normal(μ, σ)` prior on the **internal scale** `θ` (no transform).
Matches R-INLA's `gaussian(μ, prec)` with `σ = 1/√prec`.

Used for hyperparameters that already live on an unconstrained scale —
e.g. `θ = logit(p)` for the inflation probability of zero-inflated
likelihoods (R-INLA default `gaussian(0, prec=1)`), or `θ = log(α)` for
the type-2 inflation-intensity hyperparameter.

Defaults `μ = 0, σ = 1` correspond to a unit-precision Gaussian on the
internal scale — diffuse but proper.
"""
struct GaussianPrior{T <: Real} <: AbstractHyperPrior
    μ::T
    σ::T

    function GaussianPrior{T}(μ::T, σ::T) where {T <: Real}
        σ > 0 || throw(ArgumentError("GaussianPrior: σ must be > 0, got σ=$σ"))
        return new{T}(μ, σ)
    end
end
function GaussianPrior(μ::Real=0.0, σ::Real=1.0)
    T = typeof(float(μ * σ))
    return GaussianPrior{T}(T(μ), T(σ))
end

prior_name(::GaussianPrior) = :gaussian

user_scale(::GaussianPrior, θ) = θ

function log_prior_density(p::GaussianPrior, θ)
    return -0.5 * log(2π) - log(p.σ) - 0.5 * ((θ - p.μ) / p.σ)^2
end

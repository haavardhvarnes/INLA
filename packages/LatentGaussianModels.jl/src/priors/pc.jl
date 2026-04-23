"""
    PCPrecision(U, α)

The PC prior on precision (Simpson et al. 2017), specified via
`P(σ > U) = α` where `σ = 1/√τ`. User-facing parameter is `τ > 0`;
internal scale is `θ = log(τ)`.

The marginal prior on `σ` is `Exponential(λ)` with
`λ = -log(α) / U`. Under `τ = σ⁻²`, the density on the internal
scale `θ = log(τ)` is

    π(θ) = π_σ(σ(θ)) · |dσ/dθ|
         = λ exp(-λ σ) · (σ / 2)       (using σ = exp(-θ/2))

where the `σ/2` is `|dσ/dθ| = ½ · exp(-θ/2) = σ/2`.

Defaults `U = 1, α = 0.01` match R-INLA's `pc.prec`.
"""
struct PCPrecision{T <: Real} <: AbstractHyperPrior
    U::T
    α::T
    λ::T   # rate of the Exponential prior on σ

    function PCPrecision{T}(U::T, α::T) where {T <: Real}
        U > 0 || throw(ArgumentError("PCPrecision: U must be > 0, got U=$U"))
        0 < α < 1 || throw(ArgumentError("PCPrecision: α must be in (0,1), got α=$α"))
        λ = -log(α) / U
        return new{T}(U, α, λ)
    end
end
PCPrecision(U::Real = 1.0, α::Real = 0.01) = PCPrecision{typeof(float(U * α))}(float(U), float(α))

prior_name(::PCPrecision) = :pc_prec

user_scale(::PCPrecision, θ) = exp(θ)   # τ = exp(θ)

function log_prior_density(p::PCPrecision, θ)
    σ = exp(-θ / 2)
    # log π_σ(σ) = log λ - λ σ
    # log |dσ/dθ| = log(σ / 2) = -θ/2 - log(2)
    return log(p.λ) - p.λ * σ + (-θ / 2 - log(2))
end

"""
    GammaPrecision(shape, rate)

Classic Gamma prior on τ, `τ ~ Gamma(shape, rate)`. Internal scale
`θ = log(τ)`. Included for backwards compatibility with older R-INLA
workflows; PC priors are preferred.
"""
struct GammaPrecision{T <: Real} <: AbstractHyperPrior
    shape::T
    rate::T

    function GammaPrecision{T}(shape::T, rate::T) where {T <: Real}
        shape > 0 || throw(ArgumentError("GammaPrecision: shape must be > 0, got $shape"))
        rate > 0 || throw(ArgumentError("GammaPrecision: rate must be > 0, got $rate"))
        return new{T}(shape, rate)
    end
end
function GammaPrecision(shape::Real = 1.0, rate::Real = 5.0e-5)
    T = typeof(float(shape * rate))
    return GammaPrecision{T}(T(shape), T(rate))
end

prior_name(::GammaPrecision) = :gamma_prec
user_scale(::GammaPrecision, θ) = exp(θ)

function log_prior_density(p::GammaPrecision, θ)
    τ = exp(θ)
    # Gamma(shape a, rate b) on τ: log π(τ) = a log b - log Γ(a) + (a-1) log τ - b τ
    # Jacobian |dτ/dθ| = τ, so on internal scale:
    # log π(θ) = a log b - log Γ(a) + a log τ - b τ
    a = p.shape
    b = p.rate
    return a * log(b) - Distributions.loggamma(a) + a * θ - b * τ
end

"""
    LogNormalPrecision(μ, σ)

LogNormal prior on τ with parameters `μ, σ` on the log scale. Internal
scale is `θ = log(τ)`, so the prior is simply `N(μ, σ²)` on θ.
"""
struct LogNormalPrecision{T <: Real} <: AbstractHyperPrior
    μ::T
    σ::T

    function LogNormalPrecision{T}(μ::T, σ::T) where {T <: Real}
        σ > 0 || throw(ArgumentError("LogNormalPrecision: σ must be > 0, got $σ"))
        return new{T}(μ, σ)
    end
end
function LogNormalPrecision(μ::Real = 0.0, σ::Real = 1.0)
    T = typeof(float(μ * σ))
    return LogNormalPrecision{T}(T(μ), T(σ))
end

prior_name(::LogNormalPrecision) = :lognormal_prec
user_scale(::LogNormalPrecision, θ) = exp(θ)

function log_prior_density(p::LogNormalPrecision, θ)
    return -0.5 * log(2π) - log(p.σ) - 0.5 * ((θ - p.μ) / p.σ)^2
end

"""
    WeakPrior()

A deliberately-improper "nearly-flat" prior on the internal scale.
Returns `0.0` for any `θ`. Exists so that quick-and-dirty fits (and
EB-with-point-mass-prior tests) have something to dispatch on.
"""
struct WeakPrior <: AbstractHyperPrior end
prior_name(::WeakPrior) = :weak
user_scale(::WeakPrior, θ) = θ
log_prior_density(::WeakPrior, θ) = zero(θ)

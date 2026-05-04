"""
    SkewNormalLikelihood(; link = IdentityLink(),
                        precision_prior = GammaPrecision(1.0, 5.0e-5),
                        skew_prior = GaussianPrior(0.0, 1.0))

Skew-normal observation model in R-INLA's `family = "sn"`
parameterisation. With `z_i = (y_i − η_i) √τ` we have `z_i ∼ f` where

    f(z) = (2 / ω_α) · φ((z − ξ_α) / ω_α) · Φ(α (z − ξ_α) / ω_α)

is the *standardised* skew-normal density (mean 0, variance 1). The
shape `α`, location `ξ_α`, and scale `ω_α` depend only on the
standardised skewness `γ ∈ (−0.988, 0.988)` via

    γ = (4 − π) / 2 · (δ √(2/π))³ / (1 − 2δ²/π)^{3/2},   δ = α / √(1 + α²)
    ω_α = 1 / √(1 − 2δ²/π)
    ξ_α = − ω_α · δ · √(2/π)

so the mean of `y_i` is `η_i` and the variance is `1/τ`. Two
likelihood hyperparameters on the internal scale:

    θ[1] = log τ
    θ[2] = logit-skew, mapped to γ via  γ = 0.988 · tanh(θ[2] / 2)

The user-facing skewness is `γ`; the upper magnitude `0.988` matches
R-INLA's hard cap (slightly tighter than the theoretical SN limit).

Defaults match R-INLA's `family = "sn"`:
- `precision_prior = GammaPrecision(1, 5e-5)` ↔ `loggamma(1, 5e-5)` on `log τ`
- `skew_prior = GaussianPrior(0, 1)` for `θ[2]` (override of R-INLA's
  PC prior `pc.sn`, which has no closed-form Julia equivalent yet);
  pin both sides explicitly when validating an oracle.
"""
struct SkewNormalLikelihood{L <: AbstractLinkFunction,
    Pτ <: AbstractHyperPrior,
    Pγ <: AbstractHyperPrior} <: AbstractLikelihood
    link::L
    precision_prior::Pτ
    skew_prior::Pγ
end

const _SN_SKEW_MAX = 0.988
# c = (4 − π) / 2 · (2/π)^{3/2}, the leading constant in the γ(δ) map.
const _SN_C = (4 - π) / 2 * (2 / π)^(3 / 2)

function SkewNormalLikelihood(; link::AbstractLinkFunction=IdentityLink(),
        precision_prior::AbstractHyperPrior=GammaPrecision(1.0, 5.0e-5),
        skew_prior::AbstractHyperPrior=GaussianPrior(0.0, 1.0))
    link isa IdentityLink ||
        throw(ArgumentError("SkewNormalLikelihood: only IdentityLink is supported, got $(typeof(link))"))
    return SkewNormalLikelihood(link, precision_prior, skew_prior)
end

link(ℓ::SkewNormalLikelihood) = ℓ.link
nhyperparameters(::SkewNormalLikelihood) = 2
initial_hyperparameters(::SkewNormalLikelihood) = [0.0, 0.0]   # τ = 1, γ = 0
function log_hyperprior(ℓ::SkewNormalLikelihood, θ)
    return log_prior_density(ℓ.precision_prior, θ[1]) +
           log_prior_density(ℓ.skew_prior, θ[2])
end

# γ ∈ (−0.988, 0.988) from logit-skew internal coordinate.
@inline _sn_gamma_from_theta(t) = _SN_SKEW_MAX * tanh(t / 2)

# Closed-form δ from γ via the identity
#   r = (γ / c)^{1/3} = δ √(2/π) / √(1 − 2δ²/π)
#   ⇒ δ² = r² π / (π + 2 r²)
@inline function _sn_delta_from_gamma(γ::T) where {T <: Real}
    γ == zero(T) && return zero(T)
    r = (abs(γ) / _SN_C)^(1 / 3)
    δ² = r^2 * π / (π + 2 * r^2)
    return sign(γ) * sqrt(δ²)
end

# Bundle the η-independent SN parameters needed in the inner loop.
@inline function _sn_params(θ2)
    γ = _sn_gamma_from_theta(θ2)
    δ = _sn_delta_from_gamma(γ)
    ωα = 1 / sqrt(1 - 2 * δ^2 / π)
    ξα = -ωα * δ * sqrt(2 / π)
    α = δ / sqrt(max(1 - δ^2, eps(typeof(δ))))
    return ξα, ωα, α
end

# Numerically stable log-CDF and inverse Mills ratio for the standard normal.
@inline _stdnormal_logcdf(t) =
    Distributions.logcdf(Distributions.Normal(zero(t), one(t)), t)
@inline _stdnormal_logpdf(t) =
    Distributions.logpdf(Distributions.Normal(zero(t), one(t)), t)
@inline _stdnormal_λ(t) = exp(_stdnormal_logpdf(t) - _stdnormal_logcdf(t))

# --- identity-link closed forms ---------------------------------------
# Let z = (y − η) √τ, u = (z − ξα)/ωα. Then with `λ(t) = φ(t)/Φ(t)`:
#   log p(y) = log 2 − log ωα − ½ log(2π) − ½ u² + log Φ(α u) + ½ log τ
#   ∂  log p / ∂η  =  (u − α λ(α u)) · √τ / ωα
#   ∂² log p / ∂η² = − τ / ωα² · (1 + α² λ(α u) (α u + λ(α u)))
# (∂³ inherits the abstract finite-difference fallback.)

function log_density(ℓ::SkewNormalLikelihood{IdentityLink}, y, η, θ)
    τ = exp(θ[1])
    sqτ = sqrt(τ)
    ξα, ωα, α = _sn_params(θ[2])
    s = zero(promote_type(eltype(η), eltype(y), Float64))
    log_ωα = log(ωα)
    half_log_τ = θ[1] / 2
    half_log_2π = 0.5 * log(2π)
    log2 = log(2.0)
    @inbounds for i in eachindex(y)
        z = (y[i] - η[i]) * sqτ
        u = (z - ξα) / ωα
        s += log2 - log_ωα - half_log_2π - 0.5 * u^2 +
             _stdnormal_logcdf(α * u) + half_log_τ
    end
    return s
end

function ∇_η_log_density(ℓ::SkewNormalLikelihood{IdentityLink}, y, η, θ)
    τ = exp(θ[1])
    sqτ = sqrt(τ)
    ξα, ωα, α = _sn_params(θ[2])
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    inv_ωα_sqτ = sqτ / ωα
    @inbounds for i in eachindex(y)
        z = (y[i] - η[i]) * sqτ
        u = (z - ξα) / ωα
        λ = _stdnormal_λ(α * u)
        out[i] = (u - α * λ) * inv_ωα_sqτ
    end
    return out
end

function ∇²_η_log_density(ℓ::SkewNormalLikelihood{IdentityLink}, y, η, θ)
    τ = exp(θ[1])
    sqτ = sqrt(τ)
    ξα, ωα, α = _sn_params(θ[2])
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    neg_τ_inv_ωα² = -τ / ωα^2
    @inbounds for i in eachindex(y)
        z = (y[i] - η[i]) * sqτ
        u = (z - ξα) / ωα
        αu = α * u
        λ = _stdnormal_λ(αu)
        out[i] = neg_τ_inv_ωα² * (1 + α^2 * λ * (αu + λ))
    end
    return out
end

# --- pointwise log-density (CPO/WAIC) ---------------------------------

function pointwise_log_density(ℓ::SkewNormalLikelihood{IdentityLink}, y, η, θ)
    τ = exp(θ[1])
    sqτ = sqrt(τ)
    ξα, ωα, α = _sn_params(θ[2])
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    log_ωα = log(ωα)
    half_log_τ = θ[1] / 2
    half_log_2π = 0.5 * log(2π)
    log2 = log(2.0)
    @inbounds for i in eachindex(y)
        z = (y[i] - η[i]) * sqτ
        u = (z - ξα) / ωα
        out[i] = log2 - log_ωα - half_log_2π - 0.5 * u^2 +
                 _stdnormal_logcdf(α * u) + half_log_τ
    end
    return out
end

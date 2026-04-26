"""
    GammaLikelihood(; link = LogLink(), hyperprior = GammaPrecision(1.0, 5.0e-5))

`y_i | η_i, φ ∼ Gamma(μ_i, φ)` in R-INLA's mean-precision parameterisation:
mean `μ_i = g⁻¹(η_i)`, variance `μ_i² / φ`. The precision `φ > 0` is
carried as a single hyperparameter on the internal scale `θ = log(φ)`.

Density (with shape `a = φ`, rate `b = φ/μ`):

    p(y | μ, φ) = (φ/μ)^φ / Γ(φ) · y^(φ-1) · exp(-φ y / μ),  y > 0

The default hyperprior matches R-INLA's `family = "gamma"` default
(`loggamma(1, 0.00005)` on `log(φ)`). Currently only the canonical
`LogLink` is supported. The likelihood requires strictly positive
observations.
"""
struct GammaLikelihood{L <: AbstractLinkFunction,
                       P <: AbstractHyperPrior} <: AbstractLikelihood
    link::L
    hyperprior::P
end

function GammaLikelihood(; link::AbstractLinkFunction = LogLink(),
                          hyperprior::AbstractHyperPrior = GammaPrecision(1.0, 5.0e-5))
    link isa LogLink ||
        throw(ArgumentError("GammaLikelihood: only LogLink is supported, got $(typeof(link))"))
    return GammaLikelihood(link, hyperprior)
end

link(ℓ::GammaLikelihood) = ℓ.link
nhyperparameters(::GammaLikelihood) = 1
initial_hyperparameters(::GammaLikelihood) = [0.0]   # log(φ) = 0, φ = 1
log_hyperprior(ℓ::GammaLikelihood, θ) = log_prior_density(ℓ.hyperprior, θ[1])

# --- log-link closed form ---------------------------------------------
# log p(y | μ, φ) = φ log φ - φ log μ - log Γ(φ) + (φ-1) log y - φ y / μ
# Under log link, log μ = η, exp(-η) = 1/μ.

function log_density(ℓ::GammaLikelihood{LogLink}, y, η, θ)
    φ = exp(θ[1])
    s = zero(promote_type(eltype(η), eltype(y), Float64))
    log_φ = log(φ)
    lgamma_φ = Distributions.loggamma(φ)
    @inbounds for i in eachindex(y)
        y[i] > 0 || return -Inf
        s += φ * log_φ - φ * η[i] - lgamma_φ +
             (φ - 1) * log(y[i]) - φ * y[i] * exp(-η[i])
    end
    return s
end

function ∇_η_log_density(ℓ::GammaLikelihood{LogLink}, y, η, θ)
    φ = exp(θ[1])
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        # ∂/∂η [-φ η - φ y exp(-η)] = -φ + φ y exp(-η) = φ (y/μ - 1)
        out[i] = φ * (y[i] * exp(-η[i]) - 1)
    end
    return out
end

function ∇²_η_log_density(ℓ::GammaLikelihood{LogLink}, y, η, θ)
    φ = exp(θ[1])
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        # ∂²/∂η² = -φ y exp(-η) = -φ y / μ
        out[i] = -φ * y[i] * exp(-η[i])
    end
    return out
end

function ∇³_η_log_density(ℓ::GammaLikelihood{LogLink}, y, η, θ)
    φ = exp(θ[1])
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        # ∂³/∂η³ = +φ y exp(-η) = φ y / μ
        out[i] = φ * y[i] * exp(-η[i])
    end
    return out
end

# --- pointwise log-density + CDF for diagnostics ----------------------
# Distributions.Gamma(α, θ) has mean α θ and variance α θ². With our
# (μ, φ): match α = φ and θ = μ/φ, so mean = μ, var = μ²/φ.

function pointwise_log_density(ℓ::GammaLikelihood{LogLink}, y, η, θ)
    φ = exp(θ[1])
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    log_φ = log(φ)
    lgamma_φ = Distributions.loggamma(φ)
    @inbounds for i in eachindex(y)
        if y[i] > 0
            out[i] = φ * log_φ - φ * η[i] - lgamma_φ +
                     (φ - 1) * log(y[i]) - φ * y[i] * exp(-η[i])
        else
            out[i] = T(-Inf)
        end
    end
    return out
end

function pointwise_cdf(ℓ::GammaLikelihood{LogLink}, y, η, θ)
    φ = exp(θ[1])
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        μ = exp(η[i])
        out[i] = Distributions.cdf(Distributions.Gamma(φ, μ / φ), y[i])
    end
    return out
end

"""
    GammaLikelihood(; link = LogLink(),
                    hyperprior = GammaPrecision(1.0, 0.1))

`y_i | η_i, φ ∼ Gamma(shape = φ, rate = φ / μ_i)` with
`μ_i = g⁻¹(η_i)` and precision parameter `φ > 0` (R-INLA's
`prec`). Mean `μ`, variance `μ²/φ`. The single likelihood
hyperparameter is `log(φ)` on the internal scale.

The default `LogLink` gives `μ = exp(η)`, the canonical form for
positive response modelling. The default hyperprior
`GammaPrecision(1.0, 0.1)` matches R-INLA's `loggamma(1, 0.1)`
default on `log(prec)`.

`y` must be strictly positive; the Gamma log-density is `-Inf`
otherwise.
"""
struct GammaLikelihood{L <: AbstractLinkFunction,
                       P <: AbstractHyperPrior} <: AbstractLikelihood
    link::L
    hyperprior::P
end

function GammaLikelihood(; link::AbstractLinkFunction = LogLink(),
                         hyperprior::AbstractHyperPrior = GammaPrecision(1.0, 0.1))
    return GammaLikelihood(link, hyperprior)
end

link(ℓ::GammaLikelihood) = ℓ.link
nhyperparameters(::GammaLikelihood) = 1
initial_hyperparameters(::GammaLikelihood) = [0.0]         # log(φ) = 0, φ = 1
log_hyperprior(ℓ::GammaLikelihood, θ) = log_prior_density(ℓ.hyperprior, θ[1])

# --- log-link closed form ---------------------------------------------
# With μ = exp(η):
#   log p(y|μ,φ) = φ log φ - loggamma(φ) + (φ-1) log y - φ η - φ y exp(-η)
#   ∂/∂η   =  φ (y/μ - 1)
#   ∂²/∂η² = -φ y / μ
#   ∂³/∂η³ =  φ y / μ

function log_density(ℓ::GammaLikelihood{LogLink}, y, η, θ)
    φ = exp(θ[1])
    s = zero(promote_type(eltype(η), eltype(y), Float64))
    log_φ = θ[1]                           # log φ is exactly θ[1]
    loggammaφ = Distributions.loggamma(φ)
    @inbounds for i in eachindex(y)
        y[i] > 0 || return typemin(typeof(s))
        μ = exp(η[i])
        s += φ * log_φ - loggammaφ + (φ - 1) * log(y[i]) - φ * η[i] - φ * y[i] / μ
    end
    return s
end

function ∇_η_log_density(ℓ::GammaLikelihood{LogLink}, y, η, θ)
    φ = exp(θ[1])
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        μ = exp(η[i])
        out[i] = φ * (y[i] / μ - 1)
    end
    return out
end

function ∇²_η_log_density(ℓ::GammaLikelihood{LogLink}, y, η, θ)
    φ = exp(θ[1])
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        μ = exp(η[i])
        out[i] = -φ * y[i] / μ
    end
    return out
end

function ∇³_η_log_density(ℓ::GammaLikelihood{LogLink}, y, η, θ)
    φ = exp(θ[1])
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        μ = exp(η[i])
        out[i] = φ * y[i] / μ
    end
    return out
end

# --- generic link via chain rule --------------------------------------
# log p = φ log φ - loggamma(φ) + (φ-1) log y - φ log μ - φ y / μ
# ∂/∂η     = -φ μ'/μ + φ y μ'/μ²
#          =  φ (μ'/μ) (y/μ - 1)
# ∂²/∂η²   = -φ [μ''/μ - (μ'/μ)²]
#            + φ y [μ''/μ² - 2(μ')²/μ³]

function log_density(ℓ::GammaLikelihood, y, η, θ)
    φ = exp(θ[1])
    g = ℓ.link
    log_φ = θ[1]
    loggammaφ = Distributions.loggamma(φ)
    s = zero(promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        (y[i] > 0) || return typemin(typeof(s))
        μi = inverse_link(g, η[i])
        μi > 0 || return typemin(typeof(s))
        s += φ * log_φ - loggammaφ + (φ - 1) * log(y[i]) - φ * log(μi) - φ * y[i] / μi
    end
    return s
end

function ∇_η_log_density(ℓ::GammaLikelihood, y, η, θ)
    φ = exp(θ[1])
    g = ℓ.link
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        μi = inverse_link(g, η[i])
        dμ = ∂inverse_link(g, η[i])
        out[i] = φ * (dμ / μi) * (y[i] / μi - 1)
    end
    return out
end

function ∇²_η_log_density(ℓ::GammaLikelihood, y, η, θ)
    φ = exp(θ[1])
    g = ℓ.link
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        μi = inverse_link(g, η[i])
        dμ = ∂inverse_link(g, η[i])
        d²μ = ∂²inverse_link(g, η[i])
        t1 = -φ * (d²μ / μi - (dμ / μi)^2)
        t2 = φ * y[i] * (d²μ / μi^2 - 2 * dμ^2 / μi^3)
        out[i] = t1 + t2
    end
    return out
end

# --- pointwise log-density + CDF for diagnostics ----------------------

function pointwise_log_density(ℓ::GammaLikelihood{LogLink}, y, η, θ)
    φ = exp(θ[1])
    log_φ = θ[1]
    loggammaφ = Distributions.loggamma(φ)
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        if y[i] <= 0
            out[i] = T(-Inf)
        else
            μ = exp(η[i])
            out[i] = φ * log_φ - loggammaφ + (φ - 1) * log(y[i]) - φ * η[i] - φ * y[i] / μ
        end
    end
    return out
end

function pointwise_log_density(ℓ::GammaLikelihood, y, η, θ)
    φ = exp(θ[1])
    log_φ = θ[1]
    loggammaφ = Distributions.loggamma(φ)
    g = ℓ.link
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        μi = inverse_link(g, η[i])
        if y[i] <= 0 || μi <= 0
            out[i] = T(-Inf)
        else
            out[i] = φ * log_φ - loggammaφ + (φ - 1) * log(y[i]) - φ * log(μi) - φ * y[i] / μi
        end
    end
    return out
end

function pointwise_cdf(ℓ::GammaLikelihood, y, η, θ)
    φ = exp(θ[1])
    g = ℓ.link
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        μi = inverse_link(g, η[i])
        # Distributions.Gamma(α=shape, θ=scale): mean = α·θ, variance = α·θ².
        # Setting α = φ, θ = μ/φ gives mean μ and variance μ²/φ.
        out[i] = Distributions.cdf(Distributions.Gamma(φ, μi / φ), y[i])
    end
    return out
end

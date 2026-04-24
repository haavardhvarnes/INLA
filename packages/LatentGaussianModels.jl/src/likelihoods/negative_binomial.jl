"""
    NegativeBinomialLikelihood(; link = LogLink(), E = nothing,
                               hyperprior = GammaPrecision(1.0, 0.1))

`y_i | η_i, r ∼ NegBin(mean = E_i · g⁻¹(η_i), size = r)` with
dispersion parameter `r > 0` (R-INLA's `size`). Variance
`μ + μ²/r`; the limit `r → ∞` recovers Poisson. `E_i` is an
optional exposure/offset vector (disease-mapping convention);
defaults to `1`. With the canonical `LogLink`, the mean is
`E · exp(η)`.

The single likelihood hyperparameter is `log(r)` on the internal
scale. `initial_hyperparameters(ℓ) = [0.0]` corresponds to `r = 1`
(Geometric). The default `GammaPrecision(1.0, 0.1)` hyperprior
matches R-INLA's `loggamma(1, 0.1)` default on `log(size)`.
"""
struct NegativeBinomialLikelihood{L <: AbstractLinkFunction,
                                  V <: Union{Nothing, AbstractVector},
                                  P <: AbstractHyperPrior} <: AbstractLikelihood
    link::L
    E::V
    hyperprior::P
end

function NegativeBinomialLikelihood(; link::AbstractLinkFunction = LogLink(),
                                    E::Union{Nothing, AbstractVector} = nothing,
                                    hyperprior::AbstractHyperPrior = GammaPrecision(1.0, 0.1))
    return NegativeBinomialLikelihood(link, E, hyperprior)
end

link(ℓ::NegativeBinomialLikelihood) = ℓ.link
nhyperparameters(::NegativeBinomialLikelihood) = 1
initial_hyperparameters(::NegativeBinomialLikelihood) = [0.0]     # log(r) = 0, r = 1
log_hyperprior(ℓ::NegativeBinomialLikelihood, θ) = log_prior_density(ℓ.hyperprior, θ[1])

# --- log-link closed form ---------------------------------------------
# With μ = E·exp(η), λ = r + μ:
#   log p(y|μ,r) = loggamma(y+r) - loggamma(r) - loggamma(y+1)
#                + r log r - (r+y) log λ + y log μ
#   ∂/∂η   =  r(y - μ) / λ
#   ∂²/∂η² = -r μ (r + y) / λ²
#   ∂³/∂η³ = -r (r + y) μ (r - μ) / λ³

function log_density(ℓ::NegativeBinomialLikelihood{LogLink}, y, η, θ)
    r = exp(θ[1])
    s = zero(promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        λ = r + μ
        logμ = log(E) + η[i]
        s += Distributions.loggamma(y[i] + r) - Distributions.loggamma(r) -
             _loggamma_int(y[i] + 1) +
             r * log(r) - (r + y[i]) * log(λ) + y[i] * logμ
    end
    return s
end

function ∇_η_log_density(ℓ::NegativeBinomialLikelihood{LogLink}, y, η, θ)
    r = exp(θ[1])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        out[i] = r * (y[i] - μ) / (r + μ)
    end
    return out
end

function ∇²_η_log_density(ℓ::NegativeBinomialLikelihood{LogLink}, y, η, θ)
    r = exp(θ[1])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        λ = r + μ
        out[i] = -r * μ * (r + y[i]) / λ^2
    end
    return out
end

function ∇³_η_log_density(ℓ::NegativeBinomialLikelihood{LogLink}, y, η, θ)
    r = exp(θ[1])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        λ = r + μ
        out[i] = -r * (r + y[i]) * μ * (r - μ) / λ^3
    end
    return out
end

# --- generic link via chain rule --------------------------------------
# log p = C(y,r) + r log r - (r+y) log(r+μ) + y log μ,  μ = E · g⁻¹(η)
# ∂/∂η   = μ' · [ r(y-μ) / (μ(r+μ)) ]
# ∂²/∂η² = -(r+y) · (μ''/(r+μ) - (μ')²/(r+μ)²)
#          + y · (μ''/μ - (μ')²/μ²)

function log_density(ℓ::NegativeBinomialLikelihood, y, η, θ)
    r = exp(θ[1])
    g = ℓ.link
    s = zero(promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μi = E * inverse_link(g, η[i])
        μi > 0 || return -Inf
        λ = r + μi
        s += Distributions.loggamma(y[i] + r) - Distributions.loggamma(r) -
             _loggamma_int(y[i] + 1) +
             r * log(r) - (r + y[i]) * log(λ) + y[i] * log(μi)
    end
    return s
end

function ∇_η_log_density(ℓ::NegativeBinomialLikelihood, y, η, θ)
    r = exp(θ[1])
    g = ℓ.link
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μi = E * inverse_link(g, η[i])
        dμ = E * ∂inverse_link(g, η[i])
        out[i] = dμ * r * (y[i] - μi) / (μi * (r + μi))
    end
    return out
end

function ∇²_η_log_density(ℓ::NegativeBinomialLikelihood, y, η, θ)
    r = exp(θ[1])
    g = ℓ.link
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μi = E * inverse_link(g, η[i])
        dμ = E * ∂inverse_link(g, η[i])
        d²μ = E * ∂²inverse_link(g, η[i])
        λ = r + μi
        t1 = -(r + y[i]) * (d²μ / λ - dμ^2 / λ^2)
        t2 = y[i] * (d²μ / μi - dμ^2 / μi^2)
        out[i] = t1 + t2
    end
    return out
end

# --- pointwise log-density + CDF for diagnostics ----------------------

function pointwise_log_density(ℓ::NegativeBinomialLikelihood{LogLink}, y, η, θ)
    r = exp(θ[1])
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        λ = r + μ
        logμ = log(E) + η[i]
        out[i] = Distributions.loggamma(y[i] + r) - Distributions.loggamma(r) -
                 _loggamma_int(y[i] + 1) +
                 r * log(r) - (r + y[i]) * log(λ) + y[i] * logμ
    end
    return out
end

function pointwise_log_density(ℓ::NegativeBinomialLikelihood, y, η, θ)
    r = exp(θ[1])
    g = ℓ.link
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μi = E * inverse_link(g, η[i])
        if μi <= 0
            out[i] = T(-Inf)
        else
            λ = r + μi
            out[i] = Distributions.loggamma(y[i] + r) - Distributions.loggamma(r) -
                     _loggamma_int(y[i] + 1) +
                     r * log(r) - (r + y[i]) * log(λ) + y[i] * log(μi)
        end
    end
    return out
end

function pointwise_cdf(ℓ::NegativeBinomialLikelihood, y, η, θ)
    r = exp(θ[1])
    g = ℓ.link
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μi = E * inverse_link(g, η[i])
        # Distributions' parameterisation: NegBin(r, p) has mean r(1-p)/p.
        # Setting p = r/(r+μ) gives mean μ.
        p = r / (r + μi)
        out[i] = Distributions.cdf(Distributions.NegativeBinomial(r, p), y[i])
    end
    return out
end

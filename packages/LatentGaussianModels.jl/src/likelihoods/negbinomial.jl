"""
    NegativeBinomialLikelihood(; link = LogLink(), E = nothing,
                                 hyperprior = GammaPrecision(1.0, 0.1))

`y_i | η_i, n ∼ NegBinomial(μ_i, n)` with `μ_i = E_i · g⁻¹(η_i)` and
size (overdispersion) parameter `n > 0`. The mean is `μ` and the
variance is `μ + μ² / n`; as `n → ∞` this recovers Poisson.

`E` is an optional offset / exposure vector (e.g. expected counts in
disease mapping); defaults to `1`. The size parameter is carried as
a single hyperparameter on the internal scale `θ = log(n)`. The
default hyperprior matches R-INLA's `family = "nbinomial"` default
(`loggamma(1, 0.1)` on `log(n)`).

Currently only the canonical `LogLink` is supported.
"""
struct NegativeBinomialLikelihood{L <: AbstractLinkFunction,
    V <: Union{Nothing, AbstractVector},
    P <: AbstractHyperPrior} <: AbstractLikelihood
    link::L
    E::V
    hyperprior::P
end

function NegativeBinomialLikelihood(; link::AbstractLinkFunction=LogLink(),
        E::Union{Nothing, AbstractVector}=nothing,
        hyperprior::AbstractHyperPrior=GammaPrecision(1.0, 0.1))
    link isa LogLink ||
        throw(ArgumentError("NegativeBinomialLikelihood: only LogLink is supported, got $(typeof(link))"))
    return NegativeBinomialLikelihood(link, E, hyperprior)
end

link(ℓ::NegativeBinomialLikelihood) = ℓ.link
nhyperparameters(::NegativeBinomialLikelihood) = 1
initial_hyperparameters(::NegativeBinomialLikelihood) = [0.0]   # log(n) = 0, n = 1
log_hyperprior(ℓ::NegativeBinomialLikelihood, θ) = log_prior_density(ℓ.hyperprior, θ[1])

# --- log-link closed form ---------------------------------------------
# log p(y | μ, n) = lgamma(y+n) - lgamma(n) - lgamma(y+1)
#                 + n log n + y log μ - (n+y) log(n+μ),    μ = E exp(η).
# Under log link, log μ = log E + η.

function log_density(ℓ::NegativeBinomialLikelihood{LogLink}, y, η, θ)
    n = exp(θ[1])
    s = zero(promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        s += Distributions.loggamma(y[i] + n) - Distributions.loggamma(n) -
             Distributions.loggamma(y[i] + 1) +
             n * log(n) + y[i] * (log(E) + η[i]) - (n + y[i]) * log(n + μ)
    end
    return s
end

function ∇_η_log_density(ℓ::NegativeBinomialLikelihood{LogLink}, y, η, θ)
    n = exp(θ[1])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        # ∂/∂η [y log μ - (n+y) log(n+μ)] = y - (n+y) μ/(n+μ) = n(y-μ)/(n+μ)
        out[i] = n * (y[i] - μ) / (n + μ)
    end
    return out
end

function ∇²_η_log_density(ℓ::NegativeBinomialLikelihood{LogLink}, y, η, θ)
    n = exp(θ[1])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        # ∂²/∂η² = -n μ (n+y) / (n+μ)²
        out[i] = -n * μ * (n + y[i]) / (n + μ)^2
    end
    return out
end

function ∇³_η_log_density(ℓ::NegativeBinomialLikelihood{LogLink}, y, η, θ)
    # d/dη [-n μ (n+y) / (n+μ)²]
    #  with u = -n(n+y) μ, v = (n+μ)². u' = -n(n+y) μ. v' = 2(n+μ) μ.
    # (u/v)' = (u' v - u v')/v² = -n(n+y) μ (n - μ) / (n+μ)³
    n = exp(θ[1])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        out[i] = -n * (n + y[i]) * μ * (n - μ) / (n + μ)^3
    end
    return out
end

# --- pointwise log-density + CDF for diagnostics ----------------------
# Distributions.NegativeBinomial(r, p): mean = r(1-p)/p, var = r(1-p)/p².
# Our (n, μ): mean = μ, var = μ + μ²/n = μ(1 + μ/n).
# Match r = n and (1-p)/p = μ/n  ⇒  p = n / (n + μ).

function pointwise_log_density(ℓ::NegativeBinomialLikelihood{LogLink}, y, η, θ)
    n = exp(θ[1])
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        out[i] = Distributions.loggamma(y[i] + n) - Distributions.loggamma(n) -
                 Distributions.loggamma(y[i] + 1) +
                 n * log(n) + y[i] * (log(E) + η[i]) - (n + y[i]) * log(n + μ)
    end
    return out
end

function pointwise_cdf(ℓ::NegativeBinomialLikelihood{LogLink}, y, η, θ)
    n = exp(θ[1])
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        p = n / (n + μ)
        out[i] = Distributions.cdf(Distributions.NegativeBinomial(n, p), y[i])
    end
    return out
end

# --- posterior-predictive sampling ------------------------------------

function sample_y(rng::Random.AbstractRNG, ℓ::NegativeBinomialLikelihood{LogLink},
        η, θ)
    n = exp(θ[1])
    nobs = length(η)
    out = Vector{Float64}(undef, nobs)
    @inbounds for i in 1:nobs
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        p = n / (n + μ)
        out[i] = rand(rng, Distributions.NegativeBinomial(n, p))
    end
    return out
end

"""
    BetaLikelihood(; link = LogitLink(), hyperprior = GammaPrecision(1.0, 0.01))

`y_i | η_i, φ ∼ Beta(μ_i φ, (1 - μ_i) φ)` in R-INLA's mean-dispersion
parameterisation: mean `μ_i = g⁻¹(η_i) ∈ (0, 1)`, variance
`μ_i (1 - μ_i) / (φ + 1)`. The dispersion `φ > 0` is a single
likelihood hyperparameter carried on the internal scale `θ = log(φ)`.

Density (for `y ∈ (0, 1)`):

    p(y | μ, φ) = Γ(φ) / [Γ(μ φ) Γ((1 - μ) φ)] ·
                  y^(μ φ - 1) · (1 - y)^((1 - μ) φ - 1)

Currently only the canonical `LogitLink` is supported. Boundary values
(`y = 0` or `y = 1`) yield `-Inf` — these need a separate zero-/one-
inflated family.

The default hyperprior matches R-INLA's `family = "beta"` default
(`loggamma(1, 0.01)` on `log(φ)`). For PR-level oracle fits, override
on both sides to keep the prior bit-for-bit identical.
"""
struct BetaLikelihood{L <: AbstractLinkFunction,
    P <: AbstractHyperPrior} <: AbstractLikelihood
    link::L
    hyperprior::P
end

function BetaLikelihood(; link::AbstractLinkFunction=LogitLink(),
        hyperprior::AbstractHyperPrior=GammaPrecision(1.0, 0.01))
    link isa LogitLink ||
        throw(ArgumentError("BetaLikelihood: only LogitLink is supported, got $(typeof(link))"))
    return BetaLikelihood(link, hyperprior)
end

link(ℓ::BetaLikelihood) = ℓ.link
nhyperparameters(::BetaLikelihood) = 1
initial_hyperparameters(::BetaLikelihood) = [0.0]   # log(φ) = 0, φ = 1
log_hyperprior(ℓ::BetaLikelihood, θ) = log_prior_density(ℓ.hyperprior, θ[1])

# --- logit-link closed forms ------------------------------------------
# Let μ = expit(η), so dμ/dη = μ(1 - μ) =: u, and define
#   h(η) = ψ((1 - μ) φ) - ψ(μ φ) + log(y/(1 - y)),
# where ψ is the digamma function. Then:
#   ∂   log p / ∂η  = φ u h
#   ∂²  log p / ∂η² = φ u (1 - 2μ) h  - φ² u² [ψ'(μ φ) + ψ'((1 - μ) φ)]
# (The trigamma sum is the Fisher-information contribution; the first
# term vanishes at the score's zero.)

function log_density(ℓ::BetaLikelihood{LogitLink}, y, η, θ)
    φ = exp(θ[1])
    s = zero(promote_type(eltype(η), eltype(y), Float64))
    lgamma_φ = SpecialFunctions.loggamma(φ)
    @inbounds for i in eachindex(y)
        (0 < y[i] < 1) || return -Inf
        μ = inverse_link(LogitLink(), η[i])
        a = μ * φ
        b = (1 - μ) * φ
        s += lgamma_φ -
             SpecialFunctions.loggamma(a) -
             SpecialFunctions.loggamma(b) +
             (a - 1) * log(y[i]) +
             (b - 1) * log1p(-y[i])
    end
    return s
end

function ∇_η_log_density(ℓ::BetaLikelihood{LogitLink}, y, η, θ)
    φ = exp(θ[1])
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        μ = inverse_link(LogitLink(), η[i])
        u = μ * (1 - μ)
        h = SpecialFunctions.digamma((1 - μ) * φ) -
            SpecialFunctions.digamma(μ * φ) +
            log(y[i]) - log1p(-y[i])
        out[i] = φ * u * h
    end
    return out
end

function ∇²_η_log_density(ℓ::BetaLikelihood{LogitLink}, y, η, θ)
    φ = exp(θ[1])
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        μ = inverse_link(LogitLink(), η[i])
        u = μ * (1 - μ)
        h = SpecialFunctions.digamma((1 - μ) * φ) -
            SpecialFunctions.digamma(μ * φ) +
            log(y[i]) - log1p(-y[i])
        ψ′ = SpecialFunctions.trigamma(μ * φ) +
             SpecialFunctions.trigamma((1 - μ) * φ)
        out[i] = φ * u * (1 - 2μ) * h - φ^2 * u^2 * ψ′
    end
    return out
end

# --- pointwise log-density + CDF for diagnostics ----------------------

function pointwise_log_density(ℓ::BetaLikelihood{LogitLink}, y, η, θ)
    φ = exp(θ[1])
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    lgamma_φ = SpecialFunctions.loggamma(φ)
    @inbounds for i in eachindex(y)
        if 0 < y[i] < 1
            μ = inverse_link(LogitLink(), η[i])
            a = μ * φ
            b = (1 - μ) * φ
            out[i] = lgamma_φ -
                     SpecialFunctions.loggamma(a) -
                     SpecialFunctions.loggamma(b) +
                     (a - 1) * log(y[i]) +
                     (b - 1) * log1p(-y[i])
        else
            out[i] = T(-Inf)
        end
    end
    return out
end

function pointwise_cdf(ℓ::BetaLikelihood{LogitLink}, y, η, θ)
    φ = exp(θ[1])
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        μ = inverse_link(LogitLink(), η[i])
        a = μ * φ
        b = (1 - μ) * φ
        out[i] = Distributions.cdf(Distributions.Beta(a, b), y[i])
    end
    return out
end

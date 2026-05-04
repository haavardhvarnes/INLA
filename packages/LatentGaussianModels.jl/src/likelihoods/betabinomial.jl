"""
    BetaBinomialLikelihood(n_trials; link = LogitLink(),
                           hyperprior = GaussianPrior(0.0, sqrt(2.0)))

`y_i | n_i, η_i, ρ ∼ BetaBinomial(n_i, μ_i (1−ρ)/ρ, (1−μ_i)(1−ρ)/ρ)` in
R-INLA's mean-overdispersion parameterisation: mean
`E[y_i] = n_i μ_i` with `μ_i = g⁻¹(η_i) ∈ (0, 1)`, variance
`n_i μ_i (1−μ_i) (1 + (n_i−1) ρ)`. The overdispersion `ρ ∈ (0, 1)` is a
single likelihood hyperparameter carried on the internal scale
`θ = logit(ρ)`. Equivalently, `s := (1−ρ)/ρ = exp(−θ)` is the
Beta-prior "size" / effective sample, and the canonical α/β form is
`α = μ s`, `β = (1−μ) s`.

Density (with `α = μ s`, `β = (1−μ) s`, `s = exp(−θ)`):

    p(y | n, μ, ρ) = C(n, y) · B(y + α, n − y + β) / B(α, β)

Currently only the canonical `LogitLink` is supported. `n_trials` is a
fixed vector of positive integers; pass per-observation trial counts at
construction.

The default hyperprior matches R-INLA's `family = "betabinomial"`
default (`gaussian(0, prec = 0.5)` on `logit(ρ)`, i.e. mean 0 and
σ = √2 on the internal scale). For PR-level oracle fits, override on
both sides to keep the prior bit-for-bit identical.
"""
struct BetaBinomialLikelihood{L <: AbstractLinkFunction,
    V <: AbstractVector{<:Integer},
    P <: AbstractHyperPrior} <: AbstractLikelihood
    link::L
    n_trials::V
    hyperprior::P
end

function BetaBinomialLikelihood(n_trials::AbstractVector{<:Integer};
        link::AbstractLinkFunction=LogitLink(),
        hyperprior::AbstractHyperPrior=GaussianPrior(0.0, sqrt(2.0)))
    link isa LogitLink ||
        throw(ArgumentError("BetaBinomialLikelihood: only LogitLink is supported, got $(typeof(link))"))
    all(>(0), n_trials) || throw(ArgumentError("n_trials must be strictly positive"))
    return BetaBinomialLikelihood(link, n_trials, hyperprior)
end

link(ℓ::BetaBinomialLikelihood) = ℓ.link
nhyperparameters(::BetaBinomialLikelihood) = 1
initial_hyperparameters(::BetaBinomialLikelihood) = [0.0]   # logit(ρ) = 0, ρ = 0.5
log_hyperprior(ℓ::BetaBinomialLikelihood, θ) = log_prior_density(ℓ.hyperprior, θ[1])

# --- logit-link closed forms ------------------------------------------
# Let μ = expit(η), so dμ/dη = μ(1 − μ) =: u, s = exp(−θ),
# α = μ s, β = (1 − μ) s, and define
#   h(η) = ψ(y + α) − ψ(n − y + β) − ψ(α) + ψ(β),
# where ψ is the digamma. Then:
#   ∂   log p / ∂η  = s · u · h
#   ∂²  log p / ∂η² = s u (1 − 2μ) h
#                     + s² u² [ψ'(y + α) − ψ'(α) + ψ'(n − y + β) − ψ'(β)]
# (The trigamma sum carries opposite signs to Beta because the y and
# n − y shifts add positive contributions; verified vs FD across θ.)

function log_density(ℓ::BetaBinomialLikelihood{LogitLink}, y, η, θ)
    s = exp(-θ[1])
    out = zero(promote_type(eltype(η), Float64))
    lgamma_s = SpecialFunctions.loggamma(s)
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        μ = inverse_link(LogitLink(), η[i])
        a = μ * s
        b = (1 - μ) * s
        out += SpecialFunctions.loggamma(n + 1) -
               SpecialFunctions.loggamma(y[i] + 1) -
               SpecialFunctions.loggamma(n - y[i] + 1) +
               SpecialFunctions.loggamma(y[i] + a) +
               SpecialFunctions.loggamma(n - y[i] + b) -
               SpecialFunctions.loggamma(n + s) -
               SpecialFunctions.loggamma(a) -
               SpecialFunctions.loggamma(b) +
               lgamma_s
    end
    return out
end

function ∇_η_log_density(ℓ::BetaBinomialLikelihood{LogitLink}, y, η, θ)
    s = exp(-θ[1])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        μ = inverse_link(LogitLink(), η[i])
        u = μ * (1 - μ)
        a = μ * s
        b = (1 - μ) * s
        h = SpecialFunctions.digamma(y[i] + a) -
            SpecialFunctions.digamma(n - y[i] + b) -
            SpecialFunctions.digamma(a) +
            SpecialFunctions.digamma(b)
        out[i] = s * u * h
    end
    return out
end

function ∇²_η_log_density(ℓ::BetaBinomialLikelihood{LogitLink}, y, η, θ)
    s = exp(-θ[1])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        μ = inverse_link(LogitLink(), η[i])
        u = μ * (1 - μ)
        a = μ * s
        b = (1 - μ) * s
        h = SpecialFunctions.digamma(y[i] + a) -
            SpecialFunctions.digamma(n - y[i] + b) -
            SpecialFunctions.digamma(a) +
            SpecialFunctions.digamma(b)
        ψ′ = SpecialFunctions.trigamma(y[i] + a) -
             SpecialFunctions.trigamma(a) +
             SpecialFunctions.trigamma(n - y[i] + b) -
             SpecialFunctions.trigamma(b)
        out[i] = s * u * (1 - 2μ) * h + s^2 * u^2 * ψ′
    end
    return out
end

# --- pointwise log-density + CDF for diagnostics ----------------------

function pointwise_log_density(ℓ::BetaBinomialLikelihood{LogitLink}, y, η, θ)
    s = exp(-θ[1])
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    lgamma_s = SpecialFunctions.loggamma(s)
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        μ = inverse_link(LogitLink(), η[i])
        a = μ * s
        b = (1 - μ) * s
        out[i] = SpecialFunctions.loggamma(n + 1) -
                 SpecialFunctions.loggamma(y[i] + 1) -
                 SpecialFunctions.loggamma(n - y[i] + 1) +
                 SpecialFunctions.loggamma(y[i] + a) +
                 SpecialFunctions.loggamma(n - y[i] + b) -
                 SpecialFunctions.loggamma(n + s) -
                 SpecialFunctions.loggamma(a) -
                 SpecialFunctions.loggamma(b) +
                 lgamma_s
    end
    return out
end

function pointwise_cdf(ℓ::BetaBinomialLikelihood{LogitLink}, y, η, θ)
    s = exp(-θ[1])
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        μ = inverse_link(LogitLink(), η[i])
        a = μ * s
        b = (1 - μ) * s
        out[i] = Distributions.cdf(Distributions.BetaBinomial(n, a, b), y[i])
    end
    return out
end

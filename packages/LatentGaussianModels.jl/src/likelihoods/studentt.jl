"""
    StudentTLikelihood(; link = IdentityLink(),
                       precision_prior = GammaPrecision(1.0, 1.0e-4),
                       dof_prior = GaussianPrior(2.5, 1.0))

Scaled Student-t observation model:

    y_i = η_i + σ ε_i,    ε_i ~ Student-t(ν),    σ = 1 / √τ

with mean `η_i` (so the canonical link is identity), precision `τ > 0`,
and degrees of freedom `ν > 2`. Two likelihood hyperparameters,
`θ = (log τ, log(ν − 2))`. The `ν > 2` floor (rather than `ν > 0`)
ensures finite variance — this matches R-INLA's `family = "T"`.

Density:

    p(y | η, τ, ν) = Γ((ν+1)/2) / [√(πν) Γ(ν/2)] · √τ
                     · [1 + τ (y − η)² / ν]^{−(ν+1)/2}

Defaults match R-INLA's `family = "T"` in spirit (informative-precision
gamma + mildly-informative Gaussian on `log(ν − 2)`); for PR-level
oracle fits, override on both sides to keep the prior bit-for-bit
identical.
"""
struct StudentTLikelihood{L <: AbstractLinkFunction,
    Pτ <: AbstractHyperPrior,
    Pν <: AbstractHyperPrior} <: AbstractLikelihood
    link::L
    precision_prior::Pτ
    dof_prior::Pν
end

function StudentTLikelihood(; link::AbstractLinkFunction=IdentityLink(),
        precision_prior::AbstractHyperPrior=GammaPrecision(1.0, 1.0e-4),
        dof_prior::AbstractHyperPrior=GaussianPrior(2.5, 1.0))
    link isa IdentityLink ||
        throw(ArgumentError("StudentTLikelihood: only IdentityLink is supported, got $(typeof(link))"))
    return StudentTLikelihood(link, precision_prior, dof_prior)
end

link(ℓ::StudentTLikelihood) = ℓ.link
nhyperparameters(::StudentTLikelihood) = 2
initial_hyperparameters(::StudentTLikelihood) = [0.0, 2.5]   # τ = 1, ν ≈ 14
function log_hyperprior(ℓ::StudentTLikelihood, θ)
    return log_prior_density(ℓ.precision_prior, θ[1]) +
           log_prior_density(ℓ.dof_prior, θ[2])
end

# --- identity-link closed forms ---------------------------------------
# r = y − η, q = τ r² / ν, A = 1 + q. Then with `c(ν)` the t-density
# normaliser:
#   log p = log c(ν) + ½ log τ − (ν+1)/2 · log A
#   ∂   log p / ∂η  =  (ν+1) τ r / (ν + τ r²)
#   ∂²  log p / ∂η² = (ν+1) τ · (τ r² − ν) / (ν + τ r²)²
# (∂³ inherits the abstract finite-difference fallback.)

function log_density(ℓ::StudentTLikelihood{IdentityLink}, y, η, θ)
    τ = exp(θ[1])
    ν = exp(θ[2]) + 2
    s = zero(promote_type(eltype(η), eltype(y), Float64))
    log_c = SpecialFunctions.loggamma((ν + 1) / 2) -
            SpecialFunctions.loggamma(ν / 2) -
            0.5 * log(π * ν)
    @inbounds for i in eachindex(y)
        r = y[i] - η[i]
        A = 1 + τ * r^2 / ν
        s += log_c + 0.5 * log(τ) - 0.5 * (ν + 1) * log(A)
    end
    return s
end

function ∇_η_log_density(ℓ::StudentTLikelihood{IdentityLink}, y, η, θ)
    τ = exp(θ[1])
    ν = exp(θ[2]) + 2
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        r = y[i] - η[i]
        out[i] = (ν + 1) * τ * r / (ν + τ * r^2)
    end
    return out
end

function ∇²_η_log_density(ℓ::StudentTLikelihood{IdentityLink}, y, η, θ)
    τ = exp(θ[1])
    ν = exp(θ[2]) + 2
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        r = y[i] - η[i]
        denom = ν + τ * r^2
        out[i] = (ν + 1) * τ * (τ * r^2 - ν) / denom^2
    end
    return out
end

# --- pointwise log-density + CDF for diagnostics ----------------------

function pointwise_log_density(ℓ::StudentTLikelihood{IdentityLink}, y, η, θ)
    τ = exp(θ[1])
    ν = exp(θ[2]) + 2
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    log_c = SpecialFunctions.loggamma((ν + 1) / 2) -
            SpecialFunctions.loggamma(ν / 2) -
            0.5 * log(π * ν)
    @inbounds for i in eachindex(y)
        r = y[i] - η[i]
        A = 1 + τ * r^2 / ν
        out[i] = log_c + 0.5 * log(τ) - 0.5 * (ν + 1) * log(A)
    end
    return out
end

function pointwise_cdf(ℓ::StudentTLikelihood{IdentityLink}, y, η, θ)
    τ = exp(θ[1])
    ν = exp(θ[2]) + 2
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    σ = 1 / sqrt(τ)
    @inbounds for i in eachindex(y)
        z = (y[i] - η[i]) / σ
        out[i] = Distributions.cdf(Distributions.TDist(ν), z)
    end
    return out
end

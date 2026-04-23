"""
    GaussianLikelihood(; link = IdentityLink(), hyperprior = PCPrecision(1.0, 0.01))

`y_i | η_i, τ ∼ N(μ_i, τ⁻¹)` with `μ_i = g⁻¹(η_i)` and
observation precision `τ` carried as a single hyperparameter on the
internal scale `log(τ)`. The default hyperprior is the PC prior on
standard deviation `σ = τ^(-1/2)` with default
`P(σ > 1) = 0.01` (matching R-INLA's `family = 'gaussian'` default).

With `IdentityLink`, `η = μ`, which is the common case and admits a
closed-form posterior when paired with a Gaussian latent field.
"""
struct GaussianLikelihood{L <: AbstractLinkFunction, P <: AbstractHyperPrior} <: AbstractLikelihood
    link::L
    hyperprior::P
end

function GaussianLikelihood(; link::AbstractLinkFunction = IdentityLink(),
                            hyperprior::AbstractHyperPrior = PCPrecision(1.0, 0.01))
    return GaussianLikelihood(link, hyperprior)
end

link(ℓ::GaussianLikelihood) = ℓ.link
nhyperparameters(::GaussianLikelihood) = 1
initial_hyperparameters(::GaussianLikelihood) = [0.0]     # log(τ) = 0, i.e. τ = 1
log_hyperprior(ℓ::GaussianLikelihood, θ) = log_prior_density(ℓ.hyperprior, θ[1])

# --- identity link: closed form ---------------------------------------

function log_density(ℓ::GaussianLikelihood{IdentityLink}, y, η, θ)
    τ = exp(θ[1])
    n = length(y)
    s = zero(eltype(η))
    @inbounds for i in eachindex(y)
        s += (y[i] - η[i])^2
    end
    return -0.5 * n * log(2π) + 0.5 * n * log(τ) - 0.5 * τ * s
end

function ∇_η_log_density(ℓ::GaussianLikelihood{IdentityLink}, y, η, θ)
    τ = exp(θ[1])
    # τ (y - η), elementwise
    return τ .* (y .- η)
end

function ∇²_η_log_density(ℓ::GaussianLikelihood{IdentityLink}, y, η, θ)
    τ = exp(θ[1])
    return fill(-τ, length(y))
end

# --- generic link: chain through ∂μ/∂η --------------------------------

function log_density(ℓ::GaussianLikelihood, y, η, θ)
    τ = exp(θ[1])
    n = length(y)
    g = ℓ.link
    s = zero(promote_type(eltype(η), eltype(y)))
    @inbounds for i in eachindex(y)
        μi = inverse_link(g, η[i])
        s += (y[i] - μi)^2
    end
    return -0.5 * n * log(2π) + 0.5 * n * log(τ) - 0.5 * τ * s
end

function ∇_η_log_density(ℓ::GaussianLikelihood, y, η, θ)
    τ = exp(θ[1])
    g = ℓ.link
    out = similar(η, promote_type(eltype(η), eltype(y)))
    @inbounds for i in eachindex(y)
        μi = inverse_link(g, η[i])
        dμ = ∂inverse_link(g, η[i])
        out[i] = τ * (y[i] - μi) * dμ
    end
    return out
end

function ∇²_η_log_density(ℓ::GaussianLikelihood, y, η, θ)
    τ = exp(θ[1])
    g = ℓ.link
    out = similar(η, promote_type(eltype(η), eltype(y)))
    @inbounds for i in eachindex(y)
        μi = inverse_link(g, η[i])
        dμ = ∂inverse_link(g, η[i])
        d²μ = ∂²inverse_link(g, η[i])
        out[i] = τ * ((y[i] - μi) * d²μ - dμ^2)
    end
    return out
end

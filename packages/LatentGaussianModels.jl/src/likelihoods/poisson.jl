"""
    PoissonLikelihood(; link = LogLink(), E = nothing)

`y_i | η_i ∼ Poisson(E_i · g⁻¹(η_i))`. `E_i` is an optional offset /
exposure vector (e.g. expected counts in disease mapping); defaults
to `1`. With the canonical `LogLink`, the mean is `E · exp(η)`.

No likelihood hyperparameters. `θ` is ignored.
"""
struct PoissonLikelihood{L <: AbstractLinkFunction, V <: Union{Nothing, AbstractVector}} <: AbstractLikelihood
    link::L
    E::V
end

function PoissonLikelihood(; link::AbstractLinkFunction = LogLink(),
                           E::Union{Nothing, AbstractVector} = nothing)
    return PoissonLikelihood(link, E)
end

link(ℓ::PoissonLikelihood) = ℓ.link
nhyperparameters(::PoissonLikelihood) = 0
initial_hyperparameters(::PoissonLikelihood) = Float64[]

# Exposure lookup that broadcasts correctly regardless of whether E is
# `nothing` or a vector.
@inline _exposure(::Nothing, i) = 1
@inline _exposure(E::AbstractVector, i) = E[i]

# --- log-link closed form ---------------------------------------------

function log_density(ℓ::PoissonLikelihood{LogLink}, y, η, θ)
    s = zero(promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        # log p(y | λ) = y log λ - λ - log(y!), with λ = E exp(η)
        s += y[i] * (log(E) + η[i]) - E * exp(η[i]) - _loggamma_int(y[i] + 1)
    end
    return s
end

function ∇_η_log_density(ℓ::PoissonLikelihood{LogLink}, y, η, θ)
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        out[i] = y[i] - E * exp(η[i])
    end
    return out
end

function ∇²_η_log_density(ℓ::PoissonLikelihood{LogLink}, y, η, θ)
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        out[i] = -E * exp(η[i])
    end
    return out
end

function ∇³_η_log_density(ℓ::PoissonLikelihood{LogLink}, y, η, θ)
    # log p = y η - E exp(η); ∂³/∂η³ = -E exp(η).
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        out[i] = -E * exp(η[i])
    end
    return out
end

# --- generic link via chain rule --------------------------------------
# log p(y|λ) = y log λ - λ - log Γ(y+1),   λ = E · g⁻¹(η)
# d/dη:   y (λ'/λ) - λ'
# d²/dη²: y (λ'' λ - (λ')²)/λ² - λ''

function log_density(ℓ::PoissonLikelihood, y, η, θ)
    g = ℓ.link
    s = zero(promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μi = inverse_link(g, η[i])
        λ = E * μi
        λ > 0 || return -Inf
        s += y[i] * log(λ) - λ - _loggamma_int(y[i] + 1)
    end
    return s
end

function ∇_η_log_density(ℓ::PoissonLikelihood, y, η, θ)
    g = ℓ.link
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μi = inverse_link(g, η[i])
        dμ = ∂inverse_link(g, η[i])
        λ = E * μi
        out[i] = y[i] * (dμ / μi) - E * dμ
    end
    return out
end

function ∇²_η_log_density(ℓ::PoissonLikelihood, y, η, θ)
    g = ℓ.link
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μi = inverse_link(g, η[i])
        dμ = ∂inverse_link(g, η[i])
        d²μ = ∂²inverse_link(g, η[i])
        out[i] = y[i] * ((d²μ * μi - dμ^2) / μi^2) - E * d²μ
    end
    return out
end

# log Γ(n+1) for non-negative integer n via Distributions' logfactorial /
# SpecialFunctions.loggamma. Keep it local to avoid pulling in
# SpecialFunctions.jl as a dep; Distributions re-exports what we need.
_loggamma_int(n::Integer) = Distributions.loggamma(n)

# --- pointwise log-density + CDF for diagnostics ----------------------

function pointwise_log_density(ℓ::PoissonLikelihood{LogLink}, y, η, θ)
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        out[i] = y[i] * (log(E) + η[i]) - E * exp(η[i]) - _loggamma_int(y[i] + 1)
    end
    return out
end

function pointwise_log_density(ℓ::PoissonLikelihood, y, η, θ)
    g = ℓ.link
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μi = inverse_link(g, η[i])
        λ = E * μi
        out[i] = λ > 0 ? y[i] * log(λ) - λ - _loggamma_int(y[i] + 1) : T(-Inf)
    end
    return out
end

function pointwise_cdf(ℓ::PoissonLikelihood, y, η, θ)
    g = ℓ.link
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μi = inverse_link(g, η[i])
        λ = E * μi
        out[i] = Distributions.cdf(Distributions.Poisson(λ), y[i])
    end
    return out
end

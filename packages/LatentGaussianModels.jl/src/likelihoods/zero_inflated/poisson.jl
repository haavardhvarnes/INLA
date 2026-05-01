"""
    ZeroInflatedPoissonLikelihood0(; link = LogLink(), E = nothing,
                                     hyperprior = GaussianPrior(0.0, 1.0))

R-INLA `family = "zeroinflatedpoisson0"` — *hurdle* parameterisation.

```
P(y = 0)        = π
P(y = k | k>0)  = (1 - π) · μ^k e^{-μ} / k! / (1 - e^{-μ})
```

The point mass at zero `π` replaces the count component's zero entirely;
the count component is the Poisson distribution truncated to `y ≥ 1`.

Hyperparameter `θ = logit(π)` (one scalar, identical to R-INLA's
internal scale). The default hyperprior is the R-INLA default
`gaussian(mean = 0, prec = 1)` on the internal scale (`σ = 1`).

`E` is an optional offset / exposure vector. With `LogLink`,
`μ = E · exp(η)`. Currently only `LogLink` is supported.
"""
struct ZeroInflatedPoissonLikelihood0{L <: AbstractLinkFunction,
    V <: Union{Nothing, AbstractVector},
    P <: AbstractHyperPrior} <: AbstractLikelihood
    link::L
    E::V
    hyperprior::P
end

"""
    ZeroInflatedPoissonLikelihood1(; link = LogLink(), E = nothing,
                                     hyperprior = GaussianPrior(0.0, 1.0))

R-INLA `family = "zeroinflatedpoisson1"` — *standard mixture*
parameterisation.

```
P(y = 0)  = π + (1 - π) · e^{-μ}
P(y = k)  = (1 - π) · μ^k e^{-μ} / k!     (k ≥ 1)
```

The Poisson zero is *not* removed; `π` is an additional mixing
probability of an extra mass at zero. This is the most common
zero-inflated parameterisation in disease-mapping and ecology
applications.

Hyperparameter `θ = logit(π)`; default prior `gaussian(0, 1)` on the
internal scale matches R-INLA.
"""
struct ZeroInflatedPoissonLikelihood1{L <: AbstractLinkFunction,
    V <: Union{Nothing, AbstractVector},
    P <: AbstractHyperPrior} <: AbstractLikelihood
    link::L
    E::V
    hyperprior::P
end

"""
    ZeroInflatedPoissonLikelihood2(; link = LogLink(), E = nothing,
                                     hyperprior = GaussianPrior(0.0, 1.0))

R-INLA `family = "zeroinflatedpoisson2"` — *intensity-modulated*
mixture, where the zero-inflation probability is a function of the
mean:

```
π_i = 1 - (μ_i / (1 + μ_i))^α,    μ_i = E_i · exp(η_i)
P(y = 0)  = π_i + (1 - π_i) · e^{-μ_i}
P(y = k)  = (1 - π_i) · μ_i^k e^{-μ_i} / k!     (k ≥ 1)
```

Smaller intensities push more mass into the zero point — useful when
the excess-zero process scales with the latent risk. The single
hyperparameter is `α > 0`, carried internally as `θ = log α`. Default
prior `gaussian(0, 1)` on `θ` matches R-INLA.
"""
struct ZeroInflatedPoissonLikelihood2{L <: AbstractLinkFunction,
    V <: Union{Nothing, AbstractVector},
    P <: AbstractHyperPrior} <: AbstractLikelihood
    link::L
    E::V
    hyperprior::P
end

# --- constructors -----------------------------------------------------

for T in (:ZeroInflatedPoissonLikelihood0,
    :ZeroInflatedPoissonLikelihood1,
    :ZeroInflatedPoissonLikelihood2)
    @eval function $T(; link::AbstractLinkFunction=LogLink(),
            E::Union{Nothing, AbstractVector}=nothing,
            hyperprior::AbstractHyperPrior=GaussianPrior(0.0, 1.0))
        link isa LogLink ||
            throw(ArgumentError(string(
                $(QuoteNode(T)), ": only LogLink is supported, got $(typeof(link))")))
        return $T(link, E, hyperprior)
    end

    @eval link(ℓ::$T) = ℓ.link
    @eval nhyperparameters(::$T) = 1
    @eval initial_hyperparameters(::$T) = [0.0]
    @eval log_hyperprior(ℓ::$T, θ) = log_prior_density(ℓ.hyperprior, θ[1])
end

# --- log densities ----------------------------------------------------
# All three types share the same y > 0 / y == 0 branching pattern; only
# the closed-form expressions differ. Vectors of `y` may arrive as
# `Float64` (the multi-likelihood seam in ADR-017 PR2 forces a single
# element type across all blocks); `_loggamma_int` accepts any `Real`.

function log_density(ℓ::ZeroInflatedPoissonLikelihood0{LogLink}, y, η, θ)
    π = inv(one(θ[1]) + exp(-θ[1]))   # sigmoid(θ)
    log_π = -_softplus(-θ[1])
    log_1mπ = -_softplus(θ[1])
    s = zero(promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        if y[i] == 0
            s += log_π
        else
            E = _exposure(ℓ.E, i)
            μ = E * exp(η[i])
            # log(1 - e^{-μ}), stable for small μ:
            log1me = log(-expm1(-μ))
            s += log_1mπ + y[i] * (log(E) + η[i]) - μ - _loggamma_int(y[i] + 1) - log1me
        end
    end
    return s
end

function log_density(ℓ::ZeroInflatedPoissonLikelihood1{LogLink}, y, η, θ)
    log_π = -_softplus(-θ[1])
    log_1mπ = -_softplus(θ[1])
    s = zero(promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        if y[i] == 0
            # log(π + (1-π) e^{-μ}) via logsumexp.
            a = log_π
            b = log_1mπ - μ
            s += logsumexp2(a, b)
        else
            s += log_1mπ + y[i] * (log(E) + η[i]) - μ - _loggamma_int(y[i] + 1)
        end
    end
    return s
end

function log_density(ℓ::ZeroInflatedPoissonLikelihood2{LogLink}, y, η, θ)
    α = exp(θ[1])
    s = zero(promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        log_q = log(μ) - log1p(μ)            # log(μ/(1+μ))
        log_1mπ = α * log_q                  # log(q^α)
        if y[i] == 0
            # π = 1 - q^α; log(π + (1-π)·e^{-μ}) = log(1 - q^α(1 - e^{-μ})).
            # Use logsumexp to combine log π and log(1-π) - μ stably.
            log_π = log1p(-exp(log_1mπ))
            s += logsumexp2(log_π, log_1mπ - μ)
        else
            s += log_1mπ + y[i] * (log(E) + η[i]) - μ - _loggamma_int(y[i] + 1)
        end
    end
    return s
end

# --- gradients --------------------------------------------------------
# Type 0 (hurdle):
#   y = 0:  log p depends only on π — derivatives wrt η vanish.
#   y > 0:  ∂η = y - r,   ∂²η = -r + r²·t,   t = e^{-μ}, r = μ/(1-t).
#
# Type 1 (mixture):
#   y > 0:  ∂η = y - μ,   ∂²η = -μ,   ∂³η = -μ.
#   y = 0:  with w = (1-π)·e^{-μ} / (π + (1-π)·e^{-μ}),
#       ∂η  = -μ·w
#       ∂²η = -μ·w + μ²·w·(1-w)
#       ∂³η = -μ·w + 3μ²·w·(1-w) - μ³·w·(1-w)·(1-2w)
#
# Type 2 (intensity-modulated):
#   y > 0: ∂η = α/(1+μ) + y - μ,  ∂²η = -α·μ/(1+μ)² - μ.
#   y = 0: log p = log h, h = 1 - q^α·(1-t).
#       ∂η h = -q^α·[α(1-t)/(1+μ) + μ·t],  ∂η log h = ∂η h / h.
#       ∂²η log h derived via quotient rule.
# ∇³ for types 0 and 2 falls back to the AbstractLikelihood FD default.

function ∇_η_log_density(ℓ::ZeroInflatedPoissonLikelihood0{LogLink}, y, η, θ)
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        if y[i] == 0
            out[i] = 0
        else
            E = _exposure(ℓ.E, i)
            μ = E * exp(η[i])
            r = μ / (-expm1(-μ))
            out[i] = y[i] - r
        end
    end
    return out
end

function ∇²_η_log_density(ℓ::ZeroInflatedPoissonLikelihood0{LogLink}, y, η, θ)
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        if y[i] == 0
            out[i] = 0
        else
            E = _exposure(ℓ.E, i)
            μ = E * exp(η[i])
            t = exp(-μ)
            r = μ / (1 - t)
            out[i] = -r + r * r * t
        end
    end
    return out
end

function ∇_η_log_density(ℓ::ZeroInflatedPoissonLikelihood1{LogLink}, y, η, θ)
    log_π = -_softplus(-θ[1])
    log_1mπ = -_softplus(θ[1])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        if y[i] == 0
            # w = (1-π)·e^{-μ} / (π + (1-π)·e^{-μ}) via stable softmax.
            a = log_π
            b = log_1mπ - μ
            w = exp(b - logsumexp2(a, b))
            out[i] = -μ * w
        else
            out[i] = y[i] - μ
        end
    end
    return out
end

function ∇²_η_log_density(ℓ::ZeroInflatedPoissonLikelihood1{LogLink}, y, η, θ)
    log_π = -_softplus(-θ[1])
    log_1mπ = -_softplus(θ[1])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        if y[i] == 0
            a = log_π
            b = log_1mπ - μ
            w = exp(b - logsumexp2(a, b))
            out[i] = -μ * w + μ * μ * w * (1 - w)
        else
            out[i] = -μ
        end
    end
    return out
end

function ∇³_η_log_density(ℓ::ZeroInflatedPoissonLikelihood1{LogLink}, y, η, θ)
    log_π = -_softplus(-θ[1])
    log_1mπ = -_softplus(θ[1])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        if y[i] == 0
            a = log_π
            b = log_1mπ - μ
            w = exp(b - logsumexp2(a, b))
            out[i] = -μ * w + 3 * μ * μ * w * (1 - w) -
                     μ^3 * w * (1 - w) * (1 - 2w)
        else
            out[i] = -μ
        end
    end
    return out
end

function ∇_η_log_density(ℓ::ZeroInflatedPoissonLikelihood2{LogLink}, y, η, θ)
    α = exp(θ[1])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        if y[i] == 0
            # h = 1 - q^α·(1-t),  q = μ/(1+μ),  t = e^{-μ}.
            # ∂η h = -q^α · [α(1-t)/(1+μ) + μ·t].
            t = exp(-μ)
            log_q = log(μ) - log1p(μ)
            qα = exp(α * log_q)
            ∂h = -qα * (α * (1 - t) / (1 + μ) + μ * t)
            h = 1 - qα * (1 - t)
            out[i] = ∂h / h
        else
            out[i] = α / (1 + μ) + y[i] - μ
        end
    end
    return out
end

function ∇²_η_log_density(ℓ::ZeroInflatedPoissonLikelihood2{LogLink}, y, η, θ)
    α = exp(θ[1])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        if y[i] == 0
            # ∂²/∂η² log h = (h·∂²h - (∂h)²) / h².
            t = exp(-μ)
            log_q = log(μ) - log1p(μ)
            qα = exp(α * log_q)
            # g(η) := q^α · (1 - t) so that h = 1 - g.
            # g' = q^α · [α(1-t)/(1+μ) + μ·t].
            # g'' = ∂η g' obtained by direct differentiation:
            #   d/dη [q^α/(1+μ)] = q^α · (α/(1+μ) - μ/(1+μ)) / (1+μ)
            #                    = q^α · (α - μ)/(1+μ)²
            #   d/dη [α(1-t)/(1+μ)] = α · [(μ·t)(1+μ) - (1-t)·μ] / (1+μ)²
            #                       + α(1-t) · ∂η[1/(1+μ)] = ...
            # Collect via the product q^α · F(η) with F = α(1-t)/(1+μ) + μ·t:
            F = α * (1 - t) / (1 + μ) + μ * t
            ∂qα = qα * α / (1 + μ)
            # F' = α · ∂η[(1-t)/(1+μ)] + ∂η[μ·t]
            # ∂η[(1-t)/(1+μ)] = (μ·t·(1+μ) - (1-t)·μ)/(1+μ)²
            #                = μ(t(1+μ) - (1-t))/(1+μ)²
            #                = μ(t + μt - 1 + t)/(1+μ)² = μ(2t + μt - 1)/(1+μ)²
            # ∂η[μ·t] = μ·t + μ·(-μt) = μt(1 - μ)
            ∂F = α * μ * (2t + μ * t - 1) / (1 + μ)^2 + μ * t * (1 - μ)
            ∂g = qα * F
            ∂²g = ∂qα * F + qα * ∂F
            h = 1 - qα * (1 - t)
            ∂h = -∂g
            ∂²h = -∂²g
            out[i] = (h * ∂²h - ∂h * ∂h) / (h * h)
        else
            out[i] = -α * μ / (1 + μ)^2 - μ
        end
    end
    return out
end

# --- pointwise log density --------------------------------------------

function pointwise_log_density(ℓ::ZeroInflatedPoissonLikelihood0{LogLink}, y, η, θ)
    log_π = -_softplus(-θ[1])
    log_1mπ = -_softplus(θ[1])
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        if y[i] == 0
            out[i] = log_π
        else
            E = _exposure(ℓ.E, i)
            μ = E * exp(η[i])
            log1me = log(-expm1(-μ))
            out[i] = log_1mπ + y[i] * (log(E) + η[i]) - μ -
                     _loggamma_int(y[i] + 1) - log1me
        end
    end
    return out
end

function pointwise_log_density(ℓ::ZeroInflatedPoissonLikelihood1{LogLink}, y, η, θ)
    log_π = -_softplus(-θ[1])
    log_1mπ = -_softplus(θ[1])
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        if y[i] == 0
            out[i] = logsumexp2(log_π, log_1mπ - μ)
        else
            out[i] = log_1mπ + y[i] * (log(E) + η[i]) - μ - _loggamma_int(y[i] + 1)
        end
    end
    return out
end

function pointwise_log_density(ℓ::ZeroInflatedPoissonLikelihood2{LogLink}, y, η, θ)
    α = exp(θ[1])
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        log_q = log(μ) - log1p(μ)
        log_1mπ = α * log_q
        if y[i] == 0
            log_π = log1p(-exp(log_1mπ))
            out[i] = logsumexp2(log_π, log_1mπ - μ)
        else
            out[i] = log_1mπ + y[i] * (log(E) + η[i]) - μ - _loggamma_int(y[i] + 1)
        end
    end
    return out
end

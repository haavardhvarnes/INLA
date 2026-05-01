"""
    ZeroInflatedNegativeBinomialLikelihood0(; link = LogLink(), E = nothing,
                                            hyperprior_size = GammaPrecision(1.0, 0.1),
                                            hyperprior_zi = GaussianPrior(0.0, 1.0))

R-INLA `family = "zeroinflatednbinomial0"` — *hurdle* parameterisation
of the negative-binomial.

```
P(y = 0)        = π
P(y = k | k>0)  = (1 - π) · NB(k | s, μ) / (1 - NB(0 | s, μ))
```

with `μ = E · exp(η)` and size `s` (overdispersion). Two hyperparameters:
`θ[1] = log s` (size, identical to plain NB) and `θ[2] = logit(π)`.
Defaults match R-INLA: `loggamma(1, 0.1)` on `log s`, `gaussian(0, 1)`
on `logit π`.
"""
struct ZeroInflatedNegativeBinomialLikelihood0{L <: AbstractLinkFunction,
    V <: Union{Nothing, AbstractVector},
    P1 <: AbstractHyperPrior,
    P2 <: AbstractHyperPrior} <: AbstractLikelihood
    link::L
    E::V
    hyperprior_size::P1
    hyperprior_zi::P2
end

"""
    ZeroInflatedNegativeBinomialLikelihood1(; link = LogLink(), E = nothing,
                                            hyperprior_size = GammaPrecision(1.0, 0.1),
                                            hyperprior_zi = GaussianPrior(0.0, 1.0))

R-INLA `family = "zeroinflatednbinomial1"` — *standard mixture*
parameterisation:

```
P(y = 0)  = π + (1 - π) · NB(0 | s, μ)
P(y = k)  = (1 - π) · NB(k | s, μ)        (k ≥ 1)
```

with `NB(0 | s, μ) = (s/(s+μ))^s`. Hyperparameters and defaults match
type 0.
"""
struct ZeroInflatedNegativeBinomialLikelihood1{L <: AbstractLinkFunction,
    V <: Union{Nothing, AbstractVector},
    P1 <: AbstractHyperPrior,
    P2 <: AbstractHyperPrior} <: AbstractLikelihood
    link::L
    E::V
    hyperprior_size::P1
    hyperprior_zi::P2
end

"""
    ZeroInflatedNegativeBinomialLikelihood2(; link = LogLink(), E = nothing,
                                            hyperprior_size = GammaPrecision(1.0, 0.1),
                                            hyperprior_zi = GaussianPrior(0.0, 1.0))

R-INLA `family = "zeroinflatednbinomial2"` — *intensity-modulated*
parameterisation: `π_i = 1 - (μ_i / (1 + μ_i))^α`. Two hyperparameters:
`θ[1] = log s` (size) and `θ[2] = log α` (intensity-modulation
exponent). Defaults: `loggamma(1, 0.1)` on `log s`, `gaussian(0, 1)`
on `log α`.
"""
struct ZeroInflatedNegativeBinomialLikelihood2{L <: AbstractLinkFunction,
    V <: Union{Nothing, AbstractVector},
    P1 <: AbstractHyperPrior,
    P2 <: AbstractHyperPrior} <: AbstractLikelihood
    link::L
    E::V
    hyperprior_size::P1
    hyperprior_zi::P2
end

# --- constructors -----------------------------------------------------

for T in (:ZeroInflatedNegativeBinomialLikelihood0,
    :ZeroInflatedNegativeBinomialLikelihood1,
    :ZeroInflatedNegativeBinomialLikelihood2)
    @eval function $T(; link::AbstractLinkFunction=LogLink(),
            E::Union{Nothing, AbstractVector}=nothing,
            hyperprior_size::AbstractHyperPrior=GammaPrecision(1.0, 0.1),
            hyperprior_zi::AbstractHyperPrior=GaussianPrior(0.0, 1.0))
        link isa LogLink ||
            throw(ArgumentError(string(
                $(QuoteNode(T)), ": only LogLink is supported, got $(typeof(link))")))
        return $T(link, E, hyperprior_size, hyperprior_zi)
    end

    @eval link(ℓ::$T) = ℓ.link
    @eval nhyperparameters(::$T) = 2
    @eval initial_hyperparameters(::$T) = [0.0, 0.0]
    @eval log_hyperprior(ℓ::$T, θ) = log_prior_density(ℓ.hyperprior_size, θ[1]) +
                                     log_prior_density(ℓ.hyperprior_zi, θ[2])
end

# --- log densities ----------------------------------------------------
# NB pmf at k: log p_NB(k|s,μ) = lgamma(k+s) - lgamma(s) - lgamma(k+1)
#   + s log s + k log μ - (s+k) log(s+μ).

function log_density(ℓ::ZeroInflatedNegativeBinomialLikelihood0{LogLink}, y, η, θ)
    s = exp(θ[1])
    log_π = -_softplus(-θ[2])
    log_1mπ = -_softplus(θ[2])
    out = zero(promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        if y[i] == 0
            out += log_π
        else
            E = _exposure(ℓ.E, i)
            μ = E * exp(η[i])
            M = s + μ
            # NB log pmf at y_i, plus log(1 - NB pmf at 0).
            # NB pmf at 0 = (s/M)^s ⇒ log = s·log(s/M).
            log_pn0 = s * (log(s) - log(M))
            log_1mpn0 = log1p(-exp(log_pn0))
            log_nb = Distributions.loggamma(y[i] + s) - Distributions.loggamma(s) -
                     Distributions.loggamma(y[i] + 1) +
                     s * log(s) + y[i] * (log(E) + η[i]) - (s + y[i]) * log(M)
            out += log_1mπ + log_nb - log_1mpn0
        end
    end
    return out
end

function log_density(ℓ::ZeroInflatedNegativeBinomialLikelihood1{LogLink}, y, η, θ)
    s = exp(θ[1])
    log_π = -_softplus(-θ[2])
    log_1mπ = -_softplus(θ[2])
    out = zero(promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        M = s + μ
        log_nb = Distributions.loggamma(y[i] + s) - Distributions.loggamma(s) -
                 Distributions.loggamma(y[i] + 1) +
                 s * log(s) + y[i] * (log(E) + η[i]) - (s + y[i]) * log(M)
        if y[i] == 0
            out += logsumexp2(log_π, log_1mπ + log_nb)
        else
            out += log_1mπ + log_nb
        end
    end
    return out
end

function log_density(ℓ::ZeroInflatedNegativeBinomialLikelihood2{LogLink}, y, η, θ)
    s = exp(θ[1])
    α = exp(θ[2])
    out = zero(promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        M = s + μ
        log_q = log(μ) - log1p(μ)
        log_1mπ = α * log_q
        log_nb = Distributions.loggamma(y[i] + s) - Distributions.loggamma(s) -
                 Distributions.loggamma(y[i] + 1) +
                 s * log(s) + y[i] * (log(E) + η[i]) - (s + y[i]) * log(M)
        if y[i] == 0
            log_π = log1p(-exp(log_1mπ))
            out += logsumexp2(log_π, log_1mπ + log_nb)
        else
            out += log_1mπ + log_nb
        end
    end
    return out
end

# --- gradients --------------------------------------------------------
# Common shorthand: M = s + μ, p_nb = s/M, pn0 = p_nb^s = NB pmf at 0.
# Plain-NB derivatives (in η) are reused unchanged on the y > 0 branch.

# Type 0:
#   y = 0:  ∂η log p = 0, ∂²η log p = 0.
#   y > 0:  ∂η = s·(y - μ)/M − K,    K := s·μ·pn0/(M·D),   D := 1 - pn0.
#           ∂²η = -s·μ·(s+y)/M² − ∂η K, with ∂η K worked out below.

function ∇_η_log_density(ℓ::ZeroInflatedNegativeBinomialLikelihood0{LogLink}, y, η, θ)
    s = exp(θ[1])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        if y[i] == 0
            out[i] = 0
        else
            E = _exposure(ℓ.E, i)
            μ = E * exp(η[i])
            M = s + μ
            log_pn0 = s * (log(s) - log(M))
            pn0 = exp(log_pn0)
            D = 1 - pn0
            K = s * μ * pn0 / (M * D)
            out[i] = s * (y[i] - μ) / M - K
        end
    end
    return out
end

function ∇²_η_log_density(ℓ::ZeroInflatedNegativeBinomialLikelihood0{LogLink}, y, η, θ)
    s = exp(θ[1])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        if y[i] == 0
            out[i] = 0
        else
            E = _exposure(ℓ.E, i)
            μ = E * exp(η[i])
            M = s + μ
            log_pn0 = s * (log(s) - log(M))
            pn0 = exp(log_pn0)
            D = 1 - pn0
            K = s * μ * pn0 / (M * D)
            # ∂η log K = 1 - (s+1)·μ/M - s·μ·pn0/(M·D)
            ∂logK = 1 - (s + 1) * μ / M - s * μ * pn0 / (M * D)
            ∂K = K * ∂logK
            ∂²nb = -s * μ * (s + y[i]) / (M * M)
            out[i] = ∂²nb - ∂K
        end
    end
    return out
end

# Type 1:
#   y > 0:  plain NB derivatives (∂η = s(y-μ)/M, ∂²η = -sμ(s+y)/M²).
#   y = 0:  with w = (1-π)·pn0/(π+(1-π)·pn0),
#       ∂η  = -s·μ·w/M
#       ∂²η = -s²·μ·w·(1 - μ + μ·w)/M²

function ∇_η_log_density(ℓ::ZeroInflatedNegativeBinomialLikelihood1{LogLink}, y, η, θ)
    s = exp(θ[1])
    log_π = -_softplus(-θ[2])
    log_1mπ = -_softplus(θ[2])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        M = s + μ
        if y[i] == 0
            log_pn0 = s * (log(s) - log(M))
            a = log_π
            b = log_1mπ + log_pn0
            w = exp(b - logsumexp2(a, b))
            out[i] = -s * μ * w / M
        else
            out[i] = s * (y[i] - μ) / M
        end
    end
    return out
end

function ∇²_η_log_density(ℓ::ZeroInflatedNegativeBinomialLikelihood1{LogLink}, y, η, θ)
    s = exp(θ[1])
    log_π = -_softplus(-θ[2])
    log_1mπ = -_softplus(θ[2])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        M = s + μ
        if y[i] == 0
            log_pn0 = s * (log(s) - log(M))
            a = log_π
            b = log_1mπ + log_pn0
            w = exp(b - logsumexp2(a, b))
            out[i] = -s * s * μ * w * (1 - μ + μ * w) / (M * M)
        else
            out[i] = -s * μ * (s + y[i]) / (M * M)
        end
    end
    return out
end

# Type 2: π = 1 - q^α, q = μ/(1+μ).
#   y > 0:  ∂η = α/(1+μ) + s·(y-μ)/M,
#           ∂²η = -α·μ/(1+μ)² - s·μ·(s+y)/M².
#   y = 0:  log p = log(1 - q^α·(1-pn0)) = log f, f = 1 - q^α·D, D = 1 - pn0.
#           ∂η log f = -q^α · A / f,
#               A := α·D/(1+μ) + s·μ·pn0/M.
#           ∂²η computed as (∂²η f · f - (∂η f)²)/f²,  ∂²η f = -q^α[α A/(1+μ) + ∂η A].
#           ∂η A is expanded out in code.

function ∇_η_log_density(ℓ::ZeroInflatedNegativeBinomialLikelihood2{LogLink}, y, η, θ)
    s = exp(θ[1])
    α = exp(θ[2])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        M = s + μ
        if y[i] == 0
            log_pn0 = s * (log(s) - log(M))
            pn0 = exp(log_pn0)
            D = 1 - pn0
            log_q = log(μ) - log1p(μ)
            qα = exp(α * log_q)
            A = α * D / (1 + μ) + s * μ * pn0 / M
            ∂h = qα * A
            f = 1 - qα * D
            out[i] = -∂h / f
        else
            out[i] = α / (1 + μ) + s * (y[i] - μ) / M
        end
    end
    return out
end

function ∇²_η_log_density(ℓ::ZeroInflatedNegativeBinomialLikelihood2{LogLink}, y, η, θ)
    s = exp(θ[1])
    α = exp(θ[2])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        M = s + μ
        if y[i] == 0
            log_pn0 = s * (log(s) - log(M))
            pn0 = exp(log_pn0)
            D = 1 - pn0
            log_q = log(μ) - log1p(μ)
            qα = exp(α * log_q)
            A = α * D / (1 + μ) + s * μ * pn0 / M
            # ∂η A: see derivation in the comment block above.
            #   d/dη[D/(1+μ)] = μ·[s·pn0·(1+μ)/M - D]/(1+μ)²
            #   d/dη[s·μ·pn0/M] = s²·μ·(1-μ)·pn0 / M²
            ∂A = α * μ * (s * pn0 * (1 + μ) / M - D) / (1 + μ)^2 +
                 s * s * μ * (1 - μ) * pn0 / (M * M)
            ∂h = qα * A
            ∂²h = qα * (α * A / (1 + μ) + ∂A)
            f = 1 - qα * D
            ∂f = -∂h
            ∂²f = -∂²h
            out[i] = (∂²f * f - ∂f * ∂f) / (f * f)
        else
            out[i] = -α * μ / (1 + μ)^2 - s * μ * (s + y[i]) / (M * M)
        end
    end
    return out
end

# --- pointwise log density --------------------------------------------

function pointwise_log_density(ℓ::ZeroInflatedNegativeBinomialLikelihood0{LogLink}, y, η, θ)
    s = exp(θ[1])
    log_π = -_softplus(-θ[2])
    log_1mπ = -_softplus(θ[2])
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        if y[i] == 0
            out[i] = log_π
        else
            E = _exposure(ℓ.E, i)
            μ = E * exp(η[i])
            M = s + μ
            log_pn0 = s * (log(s) - log(M))
            log_1mpn0 = log1p(-exp(log_pn0))
            log_nb = Distributions.loggamma(y[i] + s) - Distributions.loggamma(s) -
                     Distributions.loggamma(y[i] + 1) +
                     s * log(s) + y[i] * (log(E) + η[i]) - (s + y[i]) * log(M)
            out[i] = log_1mπ + log_nb - log_1mpn0
        end
    end
    return out
end

function pointwise_log_density(ℓ::ZeroInflatedNegativeBinomialLikelihood1{LogLink}, y, η, θ)
    s = exp(θ[1])
    log_π = -_softplus(-θ[2])
    log_1mπ = -_softplus(θ[2])
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        M = s + μ
        log_nb = Distributions.loggamma(y[i] + s) - Distributions.loggamma(s) -
                 Distributions.loggamma(y[i] + 1) +
                 s * log(s) + y[i] * (log(E) + η[i]) - (s + y[i]) * log(M)
        if y[i] == 0
            out[i] = logsumexp2(log_π, log_1mπ + log_nb)
        else
            out[i] = log_1mπ + log_nb
        end
    end
    return out
end

function pointwise_log_density(ℓ::ZeroInflatedNegativeBinomialLikelihood2{LogLink}, y, η, θ)
    s = exp(θ[1])
    α = exp(θ[2])
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        E = _exposure(ℓ.E, i)
        μ = E * exp(η[i])
        M = s + μ
        log_q = log(μ) - log1p(μ)
        log_1mπ = α * log_q
        log_nb = Distributions.loggamma(y[i] + s) - Distributions.loggamma(s) -
                 Distributions.loggamma(y[i] + 1) +
                 s * log(s) + y[i] * (log(E) + η[i]) - (s + y[i]) * log(M)
        if y[i] == 0
            log_π = log1p(-exp(log_1mπ))
            out[i] = logsumexp2(log_π, log_1mπ + log_nb)
        else
            out[i] = log_1mπ + log_nb
        end
    end
    return out
end

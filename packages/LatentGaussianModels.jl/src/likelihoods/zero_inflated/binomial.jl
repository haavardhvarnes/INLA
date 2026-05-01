"""
    ZeroInflatedBinomialLikelihood0(n_trials; link = LogitLink(),
                                    hyperprior = GaussianPrior(0.0, 1.0))

R-INLA `family = "zeroinflatedbinomial0"` — *hurdle* parameterisation.

```
P(y = 0)        = π
P(y = k | k>0)  = (1 - π) · C(n,k) p^k (1-p)^{n-k} / (1 - (1-p)^n)
```

Hyperparameter `θ = logit(π)`; default prior `gaussian(0, 1)` on the
internal scale matches R-INLA. Currently only `LogitLink` is supported
on the count component.
"""
struct ZeroInflatedBinomialLikelihood0{L <: AbstractLinkFunction,
    V <: AbstractVector{<:Integer},
    P <: AbstractHyperPrior} <: AbstractLikelihood
    link::L
    n_trials::V
    hyperprior::P
end

"""
    ZeroInflatedBinomialLikelihood1(n_trials; link = LogitLink(),
                                    hyperprior = GaussianPrior(0.0, 1.0))

R-INLA `family = "zeroinflatedbinomial1"` — *standard mixture*
parameterisation.

```
P(y = 0)  = π + (1 - π) (1 - p)^n
P(y = k)  = (1 - π) C(n,k) p^k (1-p)^{n-k}     (k ≥ 1)
```

Hyperparameter `θ = logit(π)`; default prior `gaussian(0, 1)` matches
R-INLA. Currently only `LogitLink` is supported.
"""
struct ZeroInflatedBinomialLikelihood1{L <: AbstractLinkFunction,
    V <: AbstractVector{<:Integer},
    P <: AbstractHyperPrior} <: AbstractLikelihood
    link::L
    n_trials::V
    hyperprior::P
end

"""
    ZeroInflatedBinomialLikelihood2(n_trials; link = LogitLink(),
                                    hyperprior = GaussianPrior(0.0, 1.0))

R-INLA `family = "zeroinflatedbinomial2"` — *intensity-modulated*
mixture with `π_i = 1 - p_i^α`, `p_i = sigmoid(η_i)`. Larger Bernoulli
success probabilities push more mass into the count component; smaller
ones inflate the zero. Single hyperparameter `α > 0` carried internally
as `θ = log α`. Default prior `gaussian(0, 1)` matches R-INLA.
"""
struct ZeroInflatedBinomialLikelihood2{L <: AbstractLinkFunction,
    V <: AbstractVector{<:Integer},
    P <: AbstractHyperPrior} <: AbstractLikelihood
    link::L
    n_trials::V
    hyperprior::P
end

# --- constructors -----------------------------------------------------

for T in (:ZeroInflatedBinomialLikelihood0,
    :ZeroInflatedBinomialLikelihood1,
    :ZeroInflatedBinomialLikelihood2)
    @eval function $T(n_trials::AbstractVector{<:Integer};
            link::AbstractLinkFunction=LogitLink(),
            hyperprior::AbstractHyperPrior=GaussianPrior(0.0, 1.0))
        link isa LogitLink ||
            throw(ArgumentError(string($(QuoteNode(T)), ": only LogitLink is supported, got $(typeof(link))")))
        all(>(0), n_trials) || throw(ArgumentError("n_trials must be strictly positive"))
        return $T(link, n_trials, hyperprior)
    end

    @eval link(ℓ::$T) = ℓ.link
    @eval nhyperparameters(::$T) = 1
    @eval initial_hyperparameters(::$T) = [0.0]
    @eval log_hyperprior(ℓ::$T, θ) = log_prior_density(ℓ.hyperprior, θ[1])
end

# --- log densities ----------------------------------------------------

function log_density(ℓ::ZeroInflatedBinomialLikelihood0{LogitLink}, y, η, θ)
    log_π = -_softplus(-θ[1])
    log_1mπ = -_softplus(θ[1])
    s = zero(promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        if y[i] == 0
            s += log_π
        else
            log_p = -_softplus(-η[i])
            log_1mp = -_softplus(η[i])
            log_pn0 = n * log_1mp                # log((1-p)^n)
            # log(1 - pn0) via stable form: 1 - exp(log_pn0).
            log_1mpn0 = log1p(-exp(log_pn0))
            lnchoose = Distributions.loggamma(n + 1) - Distributions.loggamma(y[i] + 1) -
                       Distributions.loggamma(n - y[i] + 1)
            s += log_1mπ + lnchoose + y[i] * log_p + (n - y[i]) * log_1mp - log_1mpn0
        end
    end
    return s
end

function log_density(ℓ::ZeroInflatedBinomialLikelihood1{LogitLink}, y, η, θ)
    log_π = -_softplus(-θ[1])
    log_1mπ = -_softplus(θ[1])
    s = zero(promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        log_p = -_softplus(-η[i])
        log_1mp = -_softplus(η[i])
        if y[i] == 0
            # log(π + (1-π)·(1-p)^n) via logsumexp.
            s += logsumexp2(log_π, log_1mπ + n * log_1mp)
        else
            lnchoose = Distributions.loggamma(n + 1) - Distributions.loggamma(y[i] + 1) -
                       Distributions.loggamma(n - y[i] + 1)
            s += log_1mπ + lnchoose + y[i] * log_p + (n - y[i]) * log_1mp
        end
    end
    return s
end

function log_density(ℓ::ZeroInflatedBinomialLikelihood2{LogitLink}, y, η, θ)
    α = exp(θ[1])
    s = zero(promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        log_p = -_softplus(-η[i])
        log_1mp = -_softplus(η[i])
        log_1mπ = α * log_p          # log(p^α)
        if y[i] == 0
            log_π = log1p(-exp(log_1mπ))
            log_pn0 = n * log_1mp
            s += logsumexp2(log_π, log_1mπ + log_pn0)
        else
            lnchoose = Distributions.loggamma(n + 1) - Distributions.loggamma(y[i] + 1) -
                       Distributions.loggamma(n - y[i] + 1)
            s += log_1mπ + lnchoose + y[i] * log_p + (n - y[i]) * log_1mp
        end
    end
    return s
end

# --- gradients --------------------------------------------------------
# Notation: p = σ(η),  q = 1-p,  pn0 = q^n.  ∂η p = p·q.

# Type 0:
#   y = 0:  log p = log π — η-derivatives vanish.
#   y > 0:  ∂η = y - s,  s = n·p/(1 - pn0).
#           ∂²η = s·[n·p·pn0/(1 - pn0) - (1 - p)] = s·[n·p·v - (1-p)] where v = pn0/(1-pn0).
function ∇_η_log_density(ℓ::ZeroInflatedBinomialLikelihood0{LogitLink}, y, η, θ)
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        if y[i] == 0
            out[i] = 0
        else
            n = ℓ.n_trials[i]
            p = inverse_link(LogitLink(), η[i])
            pn0 = (1 - p)^n
            s = n * p / (1 - pn0)
            out[i] = y[i] - s
        end
    end
    return out
end

function ∇²_η_log_density(ℓ::ZeroInflatedBinomialLikelihood0{LogitLink}, y, η, θ)
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        if y[i] == 0
            out[i] = 0
        else
            n = ℓ.n_trials[i]
            p = inverse_link(LogitLink(), η[i])
            pn0 = (1 - p)^n
            s = n * p / (1 - pn0)
            v = pn0 / (1 - pn0)
            out[i] = s * (n * p * v - (1 - p))
        end
    end
    return out
end

# Type 1:
#   y > 0:  ∂η = y - n·p,  ∂²η = -n·p·(1-p),  ∂³η = -n·p·(1-p)·(1-2p).
#   y = 0:  with w = (1-π)·pn0 / (π + (1-π)·pn0),
#       ∂η  = -n·p·w
#       ∂²η = -n·p·(1-p)·w + n²·p²·w·(1-w)
#       ∂³η = -n·p·(1-p)·(1-2p)·w + 3·n²·p²·(1-p)·w·(1-w)
#             - n³·p³·w·(1-w)·(1-2w)

function ∇_η_log_density(ℓ::ZeroInflatedBinomialLikelihood1{LogitLink}, y, η, θ)
    log_π = -_softplus(-θ[1])
    log_1mπ = -_softplus(θ[1])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        p = inverse_link(LogitLink(), η[i])
        if y[i] == 0
            log_1mp = -_softplus(η[i])
            a = log_π
            b = log_1mπ + n * log_1mp
            w = exp(b - logsumexp2(a, b))
            out[i] = -n * p * w
        else
            out[i] = y[i] - n * p
        end
    end
    return out
end

function ∇²_η_log_density(ℓ::ZeroInflatedBinomialLikelihood1{LogitLink}, y, η, θ)
    log_π = -_softplus(-θ[1])
    log_1mπ = -_softplus(θ[1])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        p = inverse_link(LogitLink(), η[i])
        if y[i] == 0
            log_1mp = -_softplus(η[i])
            a = log_π
            b = log_1mπ + n * log_1mp
            w = exp(b - logsumexp2(a, b))
            out[i] = -n * p * (1 - p) * w + n^2 * p^2 * w * (1 - w)
        else
            out[i] = -n * p * (1 - p)
        end
    end
    return out
end

function ∇³_η_log_density(ℓ::ZeroInflatedBinomialLikelihood1{LogitLink}, y, η, θ)
    log_π = -_softplus(-θ[1])
    log_1mπ = -_softplus(θ[1])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        p = inverse_link(LogitLink(), η[i])
        if y[i] == 0
            log_1mp = -_softplus(η[i])
            a = log_π
            b = log_1mπ + n * log_1mp
            w = exp(b - logsumexp2(a, b))
            out[i] = -n * p * (1 - p) * (1 - 2p) * w +
                     3 * n^2 * p^2 * (1 - p) * w * (1 - w) -
                     n^3 * p^3 * w * (1 - w) * (1 - 2w)
        else
            out[i] = -n * p * (1 - p) * (1 - 2p)
        end
    end
    return out
end

# Type 2: π = 1 - p^α.
#   y > 0:  ∂η = α(1-p) + y - n·p,
#           ∂²η = -(n+α) p (1-p).
#   y = 0:  log p = log(1 - p^α (1 - pn0)).
#           Let A = α(1-p)(1-pn0) + n·p·pn0,
#               g_h := p^α · A   (so that ∂η h = g_h, where h = p^α(1-pn0)),
#               f := 1 - h.
#           ∂η log f = -g_h / f.
#           ∂²η log f = -∂η g_h / f - (g_h)² / f² ⋅ ((sign carried below)).
#       Working form below uses (∂²f·f - (∂f)²)/f² with ∂f = -g_h.

function ∇_η_log_density(ℓ::ZeroInflatedBinomialLikelihood2{LogitLink}, y, η, θ)
    α = exp(θ[1])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        p = inverse_link(LogitLink(), η[i])
        if y[i] == 0
            log_p = -_softplus(-η[i])
            log_1mp = -_softplus(η[i])
            pα = exp(α * log_p)
            pn0 = exp(n * log_1mp)
            A = α * (1 - p) * (1 - pn0) + n * p * pn0
            ∂h = pα * A
            f = 1 - pα * (1 - pn0)
            out[i] = -∂h / f
        else
            out[i] = α * (1 - p) + y[i] - n * p
        end
    end
    return out
end

function ∇²_η_log_density(ℓ::ZeroInflatedBinomialLikelihood2{LogitLink}, y, η, θ)
    α = exp(θ[1])
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        p = inverse_link(LogitLink(), η[i])
        if y[i] == 0
            log_p = -_softplus(-η[i])
            log_1mp = -_softplus(η[i])
            pα = exp(α * log_p)
            pn0 = exp(n * log_1mp)
            # A := α(1-p)(1-pn0) + n·p·pn0.
            A = α * (1 - p) * (1 - pn0) + n * p * pn0
            ∂h = pα * A
            # ∂η A — derivative of each piece in η-space.
            ∂A_term1 = α * p * (1 - p) * ((n + 1) * pn0 - 1)
            ∂A_term2 = n * p * pn0 * (1 - (n + 1) * p)
            ∂A = ∂A_term1 + ∂A_term2
            ∂²h = pα * (α * (1 - p) * A + ∂A)
            f = 1 - pα * (1 - pn0)
            ∂f = -∂h
            ∂²f = -∂²h
            out[i] = (∂²f * f - ∂f * ∂f) / (f * f)
        else
            out[i] = -(n + α) * p * (1 - p)
        end
    end
    return out
end

# --- pointwise log density --------------------------------------------

function pointwise_log_density(ℓ::ZeroInflatedBinomialLikelihood0{LogitLink}, y, η, θ)
    log_π = -_softplus(-θ[1])
    log_1mπ = -_softplus(θ[1])
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        if y[i] == 0
            out[i] = log_π
        else
            log_p = -_softplus(-η[i])
            log_1mp = -_softplus(η[i])
            log_pn0 = n * log_1mp
            log_1mpn0 = log1p(-exp(log_pn0))
            lnchoose = Distributions.loggamma(n + 1) - Distributions.loggamma(y[i] + 1) -
                       Distributions.loggamma(n - y[i] + 1)
            out[i] = log_1mπ + lnchoose + y[i] * log_p +
                     (n - y[i]) * log_1mp - log_1mpn0
        end
    end
    return out
end

function pointwise_log_density(ℓ::ZeroInflatedBinomialLikelihood1{LogitLink}, y, η, θ)
    log_π = -_softplus(-θ[1])
    log_1mπ = -_softplus(θ[1])
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        log_p = -_softplus(-η[i])
        log_1mp = -_softplus(η[i])
        if y[i] == 0
            out[i] = logsumexp2(log_π, log_1mπ + n * log_1mp)
        else
            lnchoose = Distributions.loggamma(n + 1) - Distributions.loggamma(y[i] + 1) -
                       Distributions.loggamma(n - y[i] + 1)
            out[i] = log_1mπ + lnchoose + y[i] * log_p + (n - y[i]) * log_1mp
        end
    end
    return out
end

function pointwise_log_density(ℓ::ZeroInflatedBinomialLikelihood2{LogitLink}, y, η, θ)
    α = exp(θ[1])
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        log_p = -_softplus(-η[i])
        log_1mp = -_softplus(η[i])
        log_1mπ = α * log_p
        if y[i] == 0
            log_π = log1p(-exp(log_1mπ))
            log_pn0 = n * log_1mp
            out[i] = logsumexp2(log_π, log_1mπ + log_pn0)
        else
            lnchoose = Distributions.loggamma(n + 1) - Distributions.loggamma(y[i] + 1) -
                       Distributions.loggamma(n - y[i] + 1)
            out[i] = log_1mπ + lnchoose + y[i] * log_p + (n - y[i]) * log_1mp
        end
    end
    return out
end

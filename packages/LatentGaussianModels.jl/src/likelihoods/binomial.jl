"""
    BinomialLikelihood(n_trials; link = LogitLink())

`y_i | η_i ∼ Binomial(n_trials[i], p_i)` with `p_i = g⁻¹(η_i)`.
Canonical link is logit.

`n_trials` must be a vector of positive integers of length matching
`y` at fit time.
"""
struct BinomialLikelihood{L <: AbstractLinkFunction, V <: AbstractVector{<:Integer}} <: AbstractLikelihood
    link::L
    n_trials::V
end

function BinomialLikelihood(n_trials::AbstractVector{<:Integer};
                            link::AbstractLinkFunction = LogitLink())
    all(>(0), n_trials) || throw(ArgumentError("n_trials must be strictly positive"))
    return BinomialLikelihood(link, n_trials)
end

link(ℓ::BinomialLikelihood) = ℓ.link

# --- logit (canonical) closed form ------------------------------------
# log p(y|p) = log C(n,y) + y log p + (n-y) log(1-p)
# With p = sigmoid(η): log p = η - log(1+e^η);  log(1-p) = -log(1+e^η)
# ⇒ log lik = log C(n,y) + y η - n log(1+e^η)
# ∇_η = y - n p
# ∇²_η = -n p (1-p)

function log_density(ℓ::BinomialLikelihood{LogitLink}, y, η, θ)
    s = zero(promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        # log(1 + e^η), stable form
        lse = η[i] > 0 ? η[i] + log1p(exp(-η[i])) : log1p(exp(η[i]))
        s += Distributions.loggamma(n + 1) - Distributions.loggamma(y[i] + 1) -
             Distributions.loggamma(n - y[i] + 1) +
             y[i] * η[i] - n * lse
    end
    return s
end

function ∇_η_log_density(ℓ::BinomialLikelihood{LogitLink}, y, η, θ)
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        p = inverse_link(LogitLink(), η[i])
        out[i] = y[i] - n * p
    end
    return out
end

function ∇²_η_log_density(ℓ::BinomialLikelihood{LogitLink}, y, η, θ)
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        p = inverse_link(LogitLink(), η[i])
        out[i] = -n * p * (1 - p)
    end
    return out
end

function ∇³_η_log_density(ℓ::BinomialLikelihood{LogitLink}, y, η, θ)
    # d/dη [-n p(1-p)] = -n · p(1-p)(1-2p). Independent of y.
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        p = inverse_link(LogitLink(), η[i])
        out[i] = -n * p * (1 - p) * (1 - 2p)
    end
    return out
end

# --- generic link via chain rule --------------------------------------
# log p(y|p) = log C(n,y) + y log p + (n-y) log(1-p),  p = g⁻¹(η)
# ∂/∂η log p(y|p) = y (p'/p) - (n-y) (p'/(1-p))
# ∂²/∂η² = y (p''·p - p'²)/p² - (n-y) [ -(p''·(1-p) + p'²) ] / (1-p)²
#        = y (p''·p - p'²)/p² + (n-y) (p''·(1-p) + p'²) / (1-p)²
# (Sign on the (n-y) part: d/dη log(1-p) = -p'/(1-p); second derivative
# gives (p''(1-p) + p'²)/(1-p)² after one more differentiation with
# the outer minus sign.)

function log_density(ℓ::BinomialLikelihood, y, η, θ)
    g = ℓ.link
    s = zero(promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        p = inverse_link(g, η[i])
        (0 < p < 1) || return -Inf
        s += Distributions.loggamma(n + 1) - Distributions.loggamma(y[i] + 1) -
             Distributions.loggamma(n - y[i] + 1) +
             y[i] * log(p) + (n - y[i]) * log(1 - p)
    end
    return s
end

function ∇_η_log_density(ℓ::BinomialLikelihood, y, η, θ)
    g = ℓ.link
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        p = inverse_link(g, η[i])
        dp = ∂inverse_link(g, η[i])
        out[i] = y[i] * (dp / p) - (n - y[i]) * (dp / (1 - p))
    end
    return out
end

function ∇²_η_log_density(ℓ::BinomialLikelihood, y, η, θ)
    g = ℓ.link
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        p = inverse_link(g, η[i])
        dp = ∂inverse_link(g, η[i])
        d²p = ∂²inverse_link(g, η[i])
        t1 = y[i] * (d²p * p - dp^2) / p^2
        t2 = -(n - y[i]) * (d²p * (1 - p) + dp^2) / (1 - p)^2
        out[i] = t1 + t2
    end
    return out
end

# --- pointwise log-density + CDF for diagnostics ----------------------

function pointwise_log_density(ℓ::BinomialLikelihood{LogitLink}, y, η, θ)
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        lse = η[i] > 0 ? η[i] + log1p(exp(-η[i])) : log1p(exp(η[i]))
        out[i] = Distributions.loggamma(n + 1) - Distributions.loggamma(y[i] + 1) -
                 Distributions.loggamma(n - y[i] + 1) +
                 y[i] * η[i] - n * lse
    end
    return out
end

function pointwise_log_density(ℓ::BinomialLikelihood, y, η, θ)
    g = ℓ.link
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        p = inverse_link(g, η[i])
        if !(0 < p < 1)
            out[i] = T(-Inf)
        else
            out[i] = Distributions.loggamma(n + 1) - Distributions.loggamma(y[i] + 1) -
                     Distributions.loggamma(n - y[i] + 1) +
                     y[i] * log(p) + (n - y[i]) * log(1 - p)
        end
    end
    return out
end

function pointwise_cdf(ℓ::BinomialLikelihood, y, η, θ)
    g = ℓ.link
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        n = ℓ.n_trials[i]
        p = inverse_link(g, η[i])
        out[i] = Distributions.cdf(Distributions.Binomial(n, p), y[i])
    end
    return out
end

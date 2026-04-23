"""
    AbstractLinkFunction

Link between the linear predictor `η = x[1] + x[2] + …` and the
likelihood's mean parameter `μ`. Concrete subtypes implement:

- `inverse_link(g, η)` — `μ` as a function of `η` (scalar → scalar).
- `∂inverse_link(g, η)` — derivative `dμ/dη`.
- `∂²inverse_link(g, η)` — second derivative (needed for closed-form
  Gaussian-likelihood fit with a non-identity link).

Link functions are stateless singletons; the concrete structs carry no
fields.
"""
abstract type AbstractLinkFunction end

"""
    inverse_link(g::AbstractLinkFunction, η) -> μ

Apply the inverse link. Works scalar-in / scalar-out and broadcasts.
"""
function inverse_link end

"""
    ∂inverse_link(g::AbstractLinkFunction, η)

Derivative `dμ/dη`.
"""
function ∂inverse_link end

"""
    ∂²inverse_link(g::AbstractLinkFunction, η)

Second derivative `d²μ/dη²`.
"""
function ∂²inverse_link end

# ---------- Identity ----------
struct IdentityLink <: AbstractLinkFunction end
inverse_link(::IdentityLink, η) = η
∂inverse_link(::IdentityLink, η) = one(η)
∂²inverse_link(::IdentityLink, η) = zero(η)

# ---------- Log ----------
struct LogLink <: AbstractLinkFunction end
inverse_link(::LogLink, η) = exp(η)
∂inverse_link(::LogLink, η) = exp(η)
∂²inverse_link(::LogLink, η) = exp(η)

# ---------- Logit ----------
struct LogitLink <: AbstractLinkFunction end
function inverse_link(::LogitLink, η)
    # numerically stable sigmoid
    if η ≥ 0
        e = exp(-η)
        return 1 / (1 + e)
    else
        e = exp(η)
        return e / (1 + e)
    end
end
function ∂inverse_link(::LogitLink, η)
    μ = inverse_link(LogitLink(), η)
    return μ * (1 - μ)
end
function ∂²inverse_link(::LogitLink, η)
    μ = inverse_link(LogitLink(), η)
    return μ * (1 - μ) * (1 - 2μ)
end

# ---------- Probit ----------
struct ProbitLink <: AbstractLinkFunction end
inverse_link(::ProbitLink, η) = _Φ(η)
∂inverse_link(::ProbitLink, η) = _ϕ(η)
∂²inverse_link(::ProbitLink, η) = -η * _ϕ(η)

# Standard-normal CDF / PDF via Distributions.jl.
const _STDNORM = Distributions.Normal()
_Φ(η) = Distributions.cdf(_STDNORM, η)
_ϕ(η) = Distributions.pdf(_STDNORM, η)

# ---------- Complementary log-log ----------
struct CloglogLink <: AbstractLinkFunction end
inverse_link(::CloglogLink, η) = 1 - exp(-exp(η))
function ∂inverse_link(::CloglogLink, η)
    e = exp(η)
    return exp(-e) * e
end
function ∂²inverse_link(::CloglogLink, η)
    e = exp(η)
    return exp(-e) * e * (1 - e)
end

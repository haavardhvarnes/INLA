"""
    LogitBeta(a, b)

`Beta(a, b)` prior on a parameter `ρ ∈ (0, 1)` expressed on the
internal logit scale `θ = logit(ρ)`. Includes the Jacobian
`|dρ/dθ| = ρ(1 - ρ)`, so on the internal scale

    log π(θ) = a · log(ρ) + b · log(1 - ρ) - log B(a, b)

where `B(a, b) = Γ(a) Γ(b) / Γ(a + b)`. With `a = b = 1` this is the
uniform-on-`ρ` prior (i.e. a standard logistic on `θ`), matching
R-INLA's default logit-Beta(1, 1) prior on Leroux's mixing parameter.

Numerically stable via the `-softplus` identities
`log σ(θ) = -log(1 + e^{-θ})`, `log(1 - σ(θ)) = -log(1 + e^{θ})`.
"""
struct LogitBeta{T <: Real} <: AbstractHyperPrior
    a::T
    b::T

    function LogitBeta{T}(a::T, b::T) where {T <: Real}
        a > 0 || throw(ArgumentError("LogitBeta: a must be > 0, got $a"))
        b > 0 || throw(ArgumentError("LogitBeta: b must be > 0, got $b"))
        return new{T}(a, b)
    end
end
function LogitBeta(a::Real = 1.0, b::Real = 1.0)
    T = typeof(float(a * b))
    return LogitBeta{T}(T(a), T(b))
end

prior_name(::LogitBeta) = :logit_beta
user_scale(::LogitBeta, θ) = inv(one(θ) + exp(-θ))

# Numerically stable softplus: log(1 + exp(x)).
_softplus(x) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))

function log_prior_density(p::LogitBeta, θ)
    log_ρ   = -_softplus(-θ)
    log_1mρ = -_softplus(θ)
    log_B = Distributions.loggamma(p.a) + Distributions.loggamma(p.b) -
            Distributions.loggamma(p.a + p.b)
    return p.a * log_ρ + p.b * log_1mρ - log_B
end

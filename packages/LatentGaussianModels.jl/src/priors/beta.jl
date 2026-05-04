"""
    BetaPrior(a, b)
    BetaPrior(d::Distributions.Beta)

Generic `Beta(a, b)` prior on a user-scale bounded-ratio parameter
`p ∈ (0, 1)`. Internally expressed via the logit transform
`θ = logit(p)` with Jacobian `|dp/dθ| = p(1-p)`, so on the internal
scale

    log π_θ(θ) = a · log(p) + b · log(1 - p) - log B(a, b),
    p          = σ(θ) = 1 / (1 + exp(-θ)).

This is a Distributions.jl-friendly entry point that complements
[`LogitBeta`](@ref): both implement the same prior with the same
numerically stable softplus form, but `BetaPrior` accepts a
`Distributions.Beta(a, b)` instance directly so authors can pass a
shared distribution object across components. Functionally
`BetaPrior(a, b) ≡ LogitBeta(a, b)`; pick whichever reads cleanest at
the call site.

Defaults `a = b = 1` give the uniform-on-`p` prior (standard logistic
on `θ`).

# Example

```julia
using Distributions
hyper = BetaPrior(Beta(2.0, 5.0))                    # via Distributions
hyper = BetaPrior(2.0, 5.0)                          # equivalent
log_prior_density(hyper, 0.0)                        # at p = 1/2
```
"""
struct BetaPrior{T <: Real} <: AbstractHyperPrior
    a::T
    b::T

    function BetaPrior{T}(a::T, b::T) where {T <: Real}
        a > 0 || throw(ArgumentError("BetaPrior: a must be > 0, got $a"))
        b > 0 || throw(ArgumentError("BetaPrior: b must be > 0, got $b"))
        return new{T}(a, b)
    end
end
function BetaPrior(a::Real=1.0, b::Real=1.0)
    T = typeof(float(a * b))
    return BetaPrior{T}(T(a), T(b))
end
function BetaPrior(d::Distributions.Beta)
    a, b = Distributions.params(d)
    return BetaPrior(a, b)
end

prior_name(::BetaPrior) = :beta
user_scale(::BetaPrior, θ) = inv(one(θ) + exp(-θ))

function log_prior_density(p::BetaPrior, θ)
    log_ρ = -_softplus(-θ)
    log_1mρ = -_softplus(θ)
    log_B = Distributions.loggamma(p.a) + Distributions.loggamma(p.b) -
            Distributions.loggamma(p.a + p.b)
    return p.a * log_ρ + p.b * log_1mρ - log_B
end

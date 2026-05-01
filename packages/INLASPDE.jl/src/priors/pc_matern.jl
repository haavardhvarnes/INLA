"""
    PCMatern(; range_U, range_α, sigma_U, sigma_α)

Joint penalised-complexity (PC) prior on the Matérn range `ρ` and
marginal standard deviation `σ`, following Fuglstad, Simpson, Lindgren
and Rue (2019). The prior is elicited by the two tail probabilities

    P(ρ < range_U) = range_α,     P(σ > sigma_U) = sigma_α,

with `range_U, sigma_U > 0` and both `range_α, sigma_α ∈ (0, 1)`.

# Scope

This prior is valid only for the *integer-α* 2D SPDE with smoothness
`ν = α - d/2 = α - 1 ≥ 1`, i.e. `α ∈ {2, 3, …}`. The `SPDE2{α=2}`
component uses it; fractional-α support (Bolin–Kirchner) is deferred
to v0.3.

# Density

Marginally, the PC-Matern prior is a product of two independent pieces:

- Range: `π(ρ) = (d/2) · λ_ρ · ρ^(-d/2-1) · exp(-λ_ρ · ρ^(-d/2))`,
  with `λ_ρ = -log(range_α) · range_U^(d/2)`.
- Sigma: `π(σ) = λ_σ · exp(-λ_σ · σ)`,
  with `λ_σ = -log(sigma_α) / sigma_U`.

For `d = 2` (the only supported spatial dimension in v0.1)
`π(ρ) = λ_ρ · ρ^(-2) · exp(-λ_ρ / ρ)` with `λ_ρ = -log(range_α) · range_U`.

See [`pc_matern_log_density`](@ref) for the evaluator used by `SPDE2`.

# Defaults

Defaults `(range_U = 1.0, range_α = 0.05, sigma_U = 1.0,
sigma_α = 0.01)` follow R-INLA's `inla.spde2.pcmatern`. Always
override these with problem-specific scales.
"""
struct PCMatern{T <: Real}
    range_U::T
    range_α::T
    sigma_U::T
    sigma_α::T
    λ_ρ::T
    λ_σ::T
end

function PCMatern(;
        range_U::Real=1.0, range_α::Real=0.05,
        sigma_U::Real=1.0, sigma_α::Real=0.01
)
    range_U > 0 ||
        throw(ArgumentError("PCMatern: range_U must be positive; got $range_U"))
    sigma_U > 0 ||
        throw(ArgumentError("PCMatern: sigma_U must be positive; got $sigma_U"))
    0 < range_α < 1 ||
        throw(ArgumentError("PCMatern: range_α must be in (0, 1); got $range_α"))
    0 < sigma_α < 1 ||
        throw(ArgumentError("PCMatern: sigma_α must be in (0, 1); got $sigma_α"))
    T = promote_type(
        typeof(float(range_U)), typeof(float(range_α)),
        typeof(float(sigma_U)), typeof(float(sigma_α))
    )
    d = 2
    λ_ρ = T(-log(range_α) * range_U^(d / 2))
    λ_σ = T(-log(sigma_α) / sigma_U)
    return PCMatern{T}(
        T(range_U), T(range_α), T(sigma_U), T(sigma_α), λ_ρ, λ_σ
    )
end

"""
    pc_matern_log_density(pc::PCMatern, log_ρ, log_σ) -> Real

Log density of a 2D PC-Matern prior evaluated on the
`(log ρ, log σ)` scale. Includes the `log ρ + log σ` Jacobian that
converts from density on `(ρ, σ)` to density on `(log ρ, log σ)`.

Used internally by `SPDE2`'s `log_hyperprior`; the change of variables
from `(log ρ, log σ)` to the internal `(log τ, log κ)` has unit
Jacobian and does not contribute an extra term (ADR-013).
"""
function pc_matern_log_density(pc::PCMatern, log_ρ::Real, log_σ::Real)
    d = 2
    ρ = exp(log_ρ)
    σ = exp(log_σ)
    # log π_ρ(ρ) + log ρ  (Jacobian log ρ converts to density on log ρ):
    #   log(d/2) + log λ_ρ - (d/2 + 1) log ρ - λ_ρ · ρ^(-d/2)  +  log ρ
    # = log(d/2) + log λ_ρ - (d/2) log ρ - λ_ρ · ρ^(-d/2)
    lp_range = log(d / 2) + log(pc.λ_ρ) - (d / 2) * log_ρ - pc.λ_ρ * ρ^(-d / 2)
    # log π_σ(σ) + log σ
    lp_sigma = log(pc.λ_σ) - pc.λ_σ * σ + log_σ
    return lp_range + lp_sigma
end

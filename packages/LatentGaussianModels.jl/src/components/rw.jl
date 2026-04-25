"""
    RW1(n; hyperprior = PCPrecision(), cyclic = false)

First-order random walk of length `n`. One hyperparameter on internal
scale `θ = log(τ)`. The component is intrinsic (rank deficient by 1)
and carries a sum-to-zero constraint so that the intercept is
identifiable.

NOTE: Sørbye-Rue (2014) `scale.model = TRUE` behaviour is not yet
applied here — the unit-τ geometric-mean marginal variance is not
forced to 1. Tracked in `plans/defaults-parity.md`.
"""
struct RW1{P <: AbstractHyperPrior} <: AbstractLatentComponent
    n::Int
    hyperprior::P
    cyclic::Bool
end
function RW1(n::Integer; hyperprior::AbstractHyperPrior = PCPrecision(),
             cyclic::Bool = false)
    n ≥ 2 || throw(ArgumentError("RW1: n must be ≥ 2"))
    return RW1(Int(n), hyperprior, cyclic)
end

Base.length(c::RW1) = c.n
nhyperparameters(::RW1) = 1
initial_hyperparameters(::RW1) = [0.0]

function gmrf(c::RW1, θ)
    return GMRFs.RW1GMRF(c.n; τ = exp(θ[1]), cyclic = c.cyclic)
end

precision_matrix(c::RW1, θ) = GMRFs.precision_matrix(gmrf(c, θ))
log_hyperprior(c::RW1, θ) = log_prior_density(c.hyperprior, θ[1])
GMRFs.constraints(c::RW1) = GMRFs.constraints(gmrf(c, [0.0]))

# Intrinsic (rank `n - 1`); R-INLA convention drops the structural
# `½ log|R̃|_+`. log NC = ½ (n - 1) log(τ).
function log_normalizing_constant(c::RW1, θ)
    return 0.5 * (c.n - 1) * θ[1]
end

"""
    RW2(n; hyperprior = PCPrecision(), cyclic = false)

Second-order random walk. Open version has rank deficiency 2 (linear
trend); cyclic has rank 1. Defaults match `RW1`; the one-hyperparameter
contract is identical.
"""
struct RW2{P <: AbstractHyperPrior} <: AbstractLatentComponent
    n::Int
    hyperprior::P
    cyclic::Bool
end
function RW2(n::Integer; hyperprior::AbstractHyperPrior = PCPrecision(),
             cyclic::Bool = false)
    n ≥ 3 || throw(ArgumentError("RW2: n must be ≥ 3"))
    return RW2(Int(n), hyperprior, cyclic)
end

Base.length(c::RW2) = c.n
nhyperparameters(::RW2) = 1
initial_hyperparameters(::RW2) = [0.0]

function gmrf(c::RW2, θ)
    return GMRFs.RW2GMRF(c.n; τ = exp(θ[1]), cyclic = c.cyclic)
end

precision_matrix(c::RW2, θ) = GMRFs.precision_matrix(gmrf(c, θ))
log_hyperprior(c::RW2, θ) = log_prior_density(c.hyperprior, θ[1])
GMRFs.constraints(c::RW2) = GMRFs.constraints(gmrf(c, [0.0]))

# Intrinsic; rank `n - 1` (cyclic) or `n - 2` (open). Structural
# `½ log|R̃|_+` dropped per R-INLA convention.
function log_normalizing_constant(c::RW2, θ)
    r = c.cyclic ? 1 : 2
    return 0.5 * (c.n - r) * θ[1]
end

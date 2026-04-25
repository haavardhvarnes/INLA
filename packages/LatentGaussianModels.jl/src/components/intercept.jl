"""
    Intercept(; prec = 1.0e-3)

Scalar intercept with a fixed Normal prior: `α ~ N(0, prec⁻¹)`. No
hyperparameters. `prec` defaults to `1e-3`, matching R-INLA's default
weak prior on fixed effects.

The latent field of this component has length 1.
"""
struct Intercept{T <: Real} <: AbstractLatentComponent
    prec::T
end
Intercept(; prec::Real = 1.0e-3) = Intercept{typeof(float(prec))}(float(prec))

Base.length(::Intercept) = 1
nhyperparameters(::Intercept) = 0
initial_hyperparameters(::Intercept) = Float64[]

function precision_matrix(c::Intercept, θ)
    return sparse([1], [1], [float(c.prec)], 1, 1)
end

log_hyperprior(::Intercept, θ) = zero(eltype(θ))

# Proper N(0, prec⁻¹) prior on a scalar: log NC = -½ log(2π) + ½ log(prec).
function log_normalizing_constant(c::Intercept, θ)
    return -0.5 * log(2π) + 0.5 * log(float(c.prec))
end

"""
    FixedEffects(p; prec = 1.0e-3)

`p`-dimensional block of fixed effects β with iid Normal prior
`β_j ~ N(0, prec⁻¹)`. No hyperparameters. `prec = 1e-3` matches
R-INLA's default `control.fixed = list(prec = 0.001)`.
"""
struct FixedEffects{T <: Real} <: AbstractLatentComponent
    p::Int
    prec::T
end
function FixedEffects(p::Integer; prec::Real = 1.0e-3)
    p > 0 || throw(ArgumentError("FixedEffects: p must be positive"))
    return FixedEffects{typeof(float(prec))}(Int(p), float(prec))
end

Base.length(c::FixedEffects) = c.p
nhyperparameters(::FixedEffects) = 0
initial_hyperparameters(::FixedEffects) = Float64[]

function precision_matrix(c::FixedEffects, θ)
    return spdiagm(0 => fill(float(c.prec), c.p))
end

log_hyperprior(::FixedEffects, θ) = zero(eltype(θ))

# Proper N(0, prec⁻¹) prior on each of `p` independent components.
function log_normalizing_constant(c::FixedEffects, θ)
    return -0.5 * c.p * log(2π) + 0.5 * c.p * log(float(c.prec))
end

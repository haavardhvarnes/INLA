"""
    Intercept(; prec = 1.0e-3, improper = true)

Scalar intercept. With `improper = true` (default, matching R-INLA's
`prec.intercept = 0`), the prior is treated as flat (improper Lebesgue);
`prec` is then only a numerical regulariser added to the joint precision
to keep it invertible, and is dropped from the log normalising constant
of the joint Gaussian. With `improper = false` the prior is the proper
Normal `α ~ N(0, prec⁻¹)`.

The latent field of this component has length 1.
"""
struct Intercept{T <: Real} <: AbstractLatentComponent
    prec::T
    improper::Bool
end
function Intercept(; prec::Real=1.0e-3, improper::Bool=true)
    return Intercept{typeof(float(prec))}(float(prec), improper)
end

Base.length(::Intercept) = 1
nhyperparameters(::Intercept) = 0
initial_hyperparameters(::Intercept) = Float64[]

function precision_matrix(c::Intercept, θ)
    return sparse([1], [1], [float(c.prec)], 1, 1)
end

log_hyperprior(::Intercept, θ) = zero(eltype(θ))

# log NC of the (regularised) joint-Gaussian block contributed by the
# intercept. Proper: -½ log(2π) + ½ log(prec). Improper: -½ log(2π)
# only — the ½ log(prec) term is dropped to match R-INLA's `prec.intercept = 0`
# convention, where `prec` is just an `idiag`-style numerical regulariser.
function log_normalizing_constant(c::Intercept, θ)
    nc = -0.5 * log(2π)
    return c.improper ? nc : nc + 0.5 * log(float(c.prec))
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
function FixedEffects(p::Integer; prec::Real=1.0e-3)
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

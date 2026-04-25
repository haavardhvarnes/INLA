"""
    IID(n; hyperprior = PCPrecision())

IID Gaussian random effect of length `n` with precision `τ`. One
hyperparameter on the internal scale `θ = log(τ)`; the user-facing
parameter is `τ`.

Composes `GMRFs.IIDGMRF` — the precision is `τ I_n`. No linear
constraint.
"""
struct IID{P <: AbstractHyperPrior} <: AbstractLatentComponent
    n::Int
    hyperprior::P
end
function IID(n::Integer; hyperprior::AbstractHyperPrior = PCPrecision())
    n > 0 || throw(ArgumentError("IID: n must be positive"))
    return IID(Int(n), hyperprior)
end

Base.length(c::IID) = c.n
nhyperparameters(::IID) = 1
initial_hyperparameters(::IID) = [0.0]            # log τ = 0

function precision_matrix(c::IID, θ)
    τ = exp(θ[1])
    return spdiagm(0 => fill(τ, c.n))
end

log_hyperprior(c::IID, θ) = log_prior_density(c.hyperprior, θ[1])

# Proper N(0, τ⁻¹ I) prior on `n` independent components.
# log NC = -½ n log(2π) + ½ n log(τ).
function log_normalizing_constant(c::IID, θ)
    return -0.5 * c.n * log(2π) + 0.5 * c.n * θ[1]
end

function gmrf(c::IID, θ)
    return GMRFs.IIDGMRF(c.n, exp(θ[1]))
end

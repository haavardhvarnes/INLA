"""
    Seasonal(n; period, hyperprior = PCPrecision())

Intrinsic seasonal-variation component of length `n` and period `s =
period`. Thin LGM wrapper around [`GMRFs.SeasonalGMRF`](@ref): the
precision is `Q = τ · B' B` penalising every `s`-consecutive sum, and
the null space (dimension `s - 1`) is handled via a `(s - 1) × n` set
of linear constraints whose row span matches `null(Q)`.

One hyperparameter on the internal scale `θ = log(τ)`.

Matches R-INLA's `model = "seasonal"` up to the constraint
parameterisation (R-INLA permits a user-chosen alternative; we default
to the Rue-Held §3.4.3 null-basis constraint).
"""
struct Seasonal{P <: AbstractHyperPrior} <: AbstractLatentComponent
    n::Int
    period::Int
    hyperprior::P
end

function Seasonal(n::Integer;
                  period::Integer,
                  hyperprior::AbstractHyperPrior = PCPrecision())
    return Seasonal(Int(n), Int(period), hyperprior)
end

Base.length(c::Seasonal) = c.n
nhyperparameters(::Seasonal) = 1
initial_hyperparameters(::Seasonal) = [0.0]

function precision_matrix(c::Seasonal, θ)
    return GMRFs.precision_matrix(GMRFs.SeasonalGMRF(c.n; period = c.period,
                                                      τ = exp(θ[1])))
end

log_hyperprior(c::Seasonal, θ) = log_prior_density(c.hyperprior, θ[1])

# Intrinsic; null space dim = period - 1, so rank = n - (period - 1).
# Structural `½ log|R̃|_+` dropped per R-INLA convention.
function log_normalizing_constant(c::Seasonal, θ)
    return 0.5 * (c.n - c.period + 1) * θ[1]
end

function gmrf(c::Seasonal, θ)
    return GMRFs.SeasonalGMRF(c.n; period = c.period, τ = exp(θ[1]))
end

function GMRFs.constraints(c::Seasonal)
    return GMRFs.constraints(GMRFs.SeasonalGMRF(c.n; period = c.period))
end

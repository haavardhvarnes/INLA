"""
    Seasonal(n; period, hyperprior = PCPrecision())

Intrinsic seasonal-variation component of length `n` and period `s =
period`. Thin LGM wrapper around [`GMRFs.SeasonalGMRF`](@ref): the
precision is `Q = τ · B' B` penalising every `s`-consecutive sum, and
the null space (dimension `s - 1`) is handled via a `(s - 1) × n`
linear constraint whose row span equals `null(Q)` (basis from Rue & Held
2005, §3.4.3).

One hyperparameter on the internal scale `θ = log(τ)`.

Matches R-INLA's `model = "seasonal"` up to the constraint
parameterisation (R-INLA permits a user-chosen alternative; we default
to the period-s zero-sum null basis).
"""
struct Seasonal{P <: AbstractHyperPrior} <: AbstractLatentComponent
    n::Int
    period::Int
    hyperprior::P
end

function Seasonal(n::Integer;
                  period::Integer,
                  hyperprior::AbstractHyperPrior = PCPrecision())
    period ≥ 2 || throw(ArgumentError("Seasonal: period must be ≥ 2, got $period"))
    n > period || throw(ArgumentError("Seasonal: n must be > period, got n=$n, period=$period"))
    return Seasonal(Int(n), Int(period), hyperprior)
end

Base.length(c::Seasonal) = c.n
nhyperparameters(::Seasonal) = 1
initial_hyperparameters(::Seasonal) = [0.0]

function gmrf(c::Seasonal, θ)
    return GMRFs.SeasonalGMRF(c.n; period = c.period, τ = exp(θ[1]))
end

precision_matrix(c::Seasonal, θ) = GMRFs.precision_matrix(gmrf(c, θ))
log_hyperprior(c::Seasonal, θ) = log_prior_density(c.hyperprior, θ[1])
GMRFs.constraints(c::Seasonal) = GMRFs.constraints(gmrf(c, [0.0]))

# Intrinsic with rank deficiency s-1; R-INLA drops the structural
# ½ log|R̃|_+. log NC = ½ (n - (s-1)) log τ.
function log_normalizing_constant(c::Seasonal, θ)
    return 0.5 * (c.n - (c.period - 1)) * θ[1]
end

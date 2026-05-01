"""
    Seasonal(n; period, hyperprior = PCPrecision())

Intrinsic seasonal-variation component of length `n` and period `s =
period`. Thin LGM wrapper around [`GMRFs.SeasonalGMRF`](@ref): the
precision is `Q = τ · B' B` penalising every `s`-consecutive sum. The
null space has dimension `s - 1`; the default constraint is a single
`1 × n` sum-to-zero row (matching R-INLA's `model = "seasonal"`), and
the remaining `s - 2` null directions are identified by the likelihood.

One hyperparameter on the internal scale `θ = log(τ)`.
"""
struct Seasonal{P <: AbstractHyperPrior} <: AbstractLatentComponent
    n::Int
    period::Int
    hyperprior::P
end

function Seasonal(n::Integer;
        period::Integer,
        hyperprior::AbstractHyperPrior=PCPrecision())
    period ≥ 2 || throw(ArgumentError("Seasonal: period must be ≥ 2, got $period"))
    n > period ||
        throw(ArgumentError("Seasonal: n must be > period, got n=$n, period=$period"))
    return Seasonal(Int(n), Int(period), hyperprior)
end

Base.length(c::Seasonal) = c.n
nhyperparameters(::Seasonal) = 1
initial_hyperparameters(::Seasonal) = [0.0]

function gmrf(c::Seasonal, θ)
    return GMRFs.SeasonalGMRF(c.n; period=c.period, τ=exp(θ[1]))
end

precision_matrix(c::Seasonal, θ) = GMRFs.precision_matrix(gmrf(c, θ))
log_hyperprior(c::Seasonal, θ) = log_prior_density(c.hyperprior, θ[1])
GMRFs.constraints(c::Seasonal) = GMRFs.constraints(gmrf(c, [0.0]))

# Per-component log NC for `F_SEASONAL`: the unconstrained prior has
# rank deficiency `s - 1`, but the sum-to-zero constraint hits
# `range(Q)` (the all-ones vector is *not* in `null(Q)` because the
# rolling-period sum of `1` equals `s ≠ 0`). One PD direction is
# therefore consumed by the constraint, so the effective τ-scaled
# prior dimension on the constraint surface is `n - s`, not `n - (s−1)`.
# Without this correction the marginal grows as `+½ log τ` for large
# τ (extra null cancelled against the Marriott-Van Loan log-det), which
# shifts the τ_seas posterior mode by an order of magnitude. Compare
# to F_GENERIC0/F_BYM2 where `range(C^T) ⊂ null(Q)` and no correction
# is needed (the constraint kills only null directions, leaving
# `(n - rd)` PD dimensions intact).
function log_normalizing_constant(c::Seasonal, θ)
    rd_eff = c.period
    return -0.5 * (c.n - rd_eff) * log(2π) + 0.5 * (c.n - rd_eff) * θ[1]
end

"""
    IID(n; hyperprior = PCPrecision(),
        τ_init = 0.0,
        fix_τ::Bool = false)

IID Gaussian random effect of length `n` with precision `τ`. One
hyperparameter on the internal scale `θ = log(τ)`; the user-facing
parameter is `τ`.

Composes `GMRFs.IIDGMRF` — the precision is `τ I_n`. No linear
constraint.

# Arguments

- `n`: number of independent slots.
- `hyperprior`: scalar prior on `log τ`. Default `PCPrecision()`.
- `τ_init`: initial value of `log τ` on the internal scale. Used as
  the optimisation start, and as the held-fixed value when
  `fix_τ = true`.
- `fix_τ`: when `true`, drop the precision from the θ vector and hold
  it at `τ = exp(τ_init)`. Useful for the multinomial-via-Poisson
  reformulation (ADR-024), where the per-row α-intercepts carry as a
  fixed-precision IID nuisance block (`τ_init = -10`, `fix_τ = true`,
  matching R-INLA's `prec = list(initial = -10, fixed = TRUE)`).
"""
struct IID{P <: AbstractHyperPrior} <: AbstractLatentComponent
    n::Int
    hyperprior::P
    τ_init::Float64
    fix_τ::Bool
end
function IID(n::Integer; hyperprior::AbstractHyperPrior=PCPrecision(),
        τ_init::Real=0.0,
        fix_τ::Bool=false)
    n > 0 || throw(ArgumentError("IID: n must be positive"))
    return IID{typeof(hyperprior)}(Int(n), hyperprior, Float64(τ_init), fix_τ)
end

Base.length(c::IID) = c.n
nhyperparameters(c::IID) = c.fix_τ ? 0 : 1
initial_hyperparameters(c::IID) = c.fix_τ ? Float64[] : [c.τ_init]

_iid_log_τ(c::IID, θ) = c.fix_τ ? c.τ_init : θ[1]

function precision_matrix(c::IID, θ)
    τ = exp(_iid_log_τ(c, θ))
    return spdiagm(0 => fill(τ, c.n))
end

log_hyperprior(c::IID, θ) = c.fix_τ ? 0.0 : log_prior_density(c.hyperprior, θ[1])

# Proper N(0, τ⁻¹ I) prior on `n` independent components.
# log NC = -½ n log(2π) + ½ n log(τ).
function log_normalizing_constant(c::IID, θ)
    log_τ = _iid_log_τ(c, θ)
    return -0.5 * c.n * log(2π) + 0.5 * c.n * log_τ
end

function gmrf(c::IID, θ)
    return GMRFs.IIDGMRF(c.n, exp(_iid_log_τ(c, θ)))
end

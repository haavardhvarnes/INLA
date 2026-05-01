"""
    PCAlphaW(λ = 5.0)

Penalised-complexity prior on the Weibull shape parameter `α > 0`
(Sørbye-Rue 2017; Simpson et al. 2017). The reference model is the
exponential `α = 1`, and the prior is exponential on the
Kullback-Leibler distance

    d(α) = √(2 · KL(Weibull(1, α) ‖ Exp(1)))
         = √(2 · [Γ(1 + 1/α) + log α - γ + γ/α - 1])

with rate `λ`. The corresponding density on `α > 0` is

    π(α) = ½ λ exp(-λ d(α)) |dd/dα|,

with the factor of ½ because `α = 1` is interior to the parameter
space and the unsigned distance `d` corresponds to two values of `α`
either side of 1. On the internal scale `θ = log α` (where the Weibull
likelihood carries `α`), the Jacobian `|dα/dθ| = α` cancels with the
`1/α` in `|dd/dlogα| = α |dd/dα|`, so

    log π_θ(θ) = log(½ λ) - λ d(α) + log|dd/dlogα|.

Mirrors R-INLA's `inla.pc.dalphaw` exactly (cross-checked at
`α ∈ {0.5, 1, 2, 5}` to ≈1e-12 nat).

The default `λ = 5` matches R-INLA's `pc.alphaw` default — a
moderately informative prior on shrinkage toward the exponential.

# Example

```julia
hyper = PCAlphaW(5.0)
log_prior_density(hyper, 0.0)   # log π_θ(0) at α = 1 (the reference)
```
"""
struct PCAlphaW{T <: Real} <: AbstractHyperPrior
    λ::T

    function PCAlphaW{T}(λ::T) where {T <: Real}
        λ > 0 || throw(ArgumentError("PCAlphaW: λ must be > 0, got λ=$λ"))
        return new{T}(λ)
    end
end
PCAlphaW(λ::Real=5.0) = PCAlphaW{typeof(float(λ))}(float(λ))

prior_name(::PCAlphaW) = :pc_alphaw

user_scale(::PCAlphaW, θ) = exp(θ)   # α = exp(θ)

# K''(0) where K(θ) = KL(Weibull(1, exp(θ)) ‖ Exp(1)). Both d(α=1) and
# dd/dθ(α=1) vanish, so the prior density at the reference takes its
# value as a Taylor limit. Closed-form:
#
#     K''(0) = (1 - γ_E)² + π²/6.
#
# Numerically ≈ 1.8237 nats; √K''(0) ≈ 1.3504 is the limiting
# |dd/dlogα| at α = 1.
const _PC_ALPHAW_K2_AT_0 = (1 - MathConstants.eulergamma)^2 + π^2 / 6

# Threshold below which we switch to the Taylor limit. KL(α) ≈
# ½ K''(0) θ² near θ = 0, so KL < 1e-12 means |θ| < ~1e-6 — well
# inside floating-point noise on KL.
const _PC_ALPHAW_KL_TOL = 1.0e-12

function log_prior_density(p::PCAlphaW, θ)
    γ = MathConstants.eulergamma
    α = exp(θ)
    invα = 1 / α

    # When 1/α is so large that `Γ(1 + 1/α)` overflows Float64, the
    # PC density is super-exponentially small (`d ∼ √Γ → ∞`). Return
    # `-Inf` rather than letting `Inf - Inf` propagate to NaN.
    log_Γ = Distributions.loggamma(1 + invα)
    if !isfinite(log_Γ) || log_Γ > 700
        return typemin(typeof(log_Γ))
    end

    Γ_term = exp(log_Γ)
    KL = Γ_term + θ - γ + γ * invα - 1
    KL = max(KL, zero(KL))   # numerical safety: KL is non-negative

    if KL < _PC_ALPHAW_KL_TOL
        # Taylor limit at θ = 0: |dd/dlogα| → √K''(0).
        d = sqrt(2 * KL)
        log_dd = 0.5 * log(_PC_ALPHAW_K2_AT_0)
    else
        d = sqrt(2 * KL)
        ψ_term = Distributions.digamma(1 + invα)
        # dKL/dlogα = α · dKL/dα = -Γ(1+1/α) ψ(1+1/α)/α + 1 - γ/α.
        dKL_dlogα = -Γ_term * ψ_term * invα + 1 - γ * invα
        log_dd = log(abs(dKL_dlogα)) - log(d)
    end

    return log(p.λ) - log(2) - p.λ * d + log_dd
end

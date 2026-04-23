"""
    DEFAULT_BYM2_PHI_ALPHA

Default right-tail probability `α` for the PC prior on BYM2's mixing
parameter `φ`, given `P(φ < U) = α` with `U = 0.5`.

Set to `2/3` from Riebler et al. (2016). **Current R-INLA may ship a
different default** — some releases use `α = 0.5`. This constant must be
reconciled against the running R-INLA in
`scripts/verify-defaults/bym2_phi.R` before v0.1 release; until then the
divergence is documented in `plans/defaults-parity.md`.
"""
const DEFAULT_BYM2_PHI_ALPHA = 2 / 3

"""
    PCBYM2Phi{T}(U, α, γ)

PC prior on the BYM2 mixing parameter `φ ∈ [0, 1]`, specified by
`P(φ < U) = α`. The base model is `φ = 0` (pure IID random effects);
the alternative is `φ > 0` (increasing weight on the scaled spatial
component). See Riebler, Sørbye, Simpson & Rue (2016).

`γ` is the vector of non-zero eigenvalues of the **scaled** Besag
structure matrix `R_scaled = c · L`, where `L` is the combinatorial
graph Laplacian and `c` is the Sørbye-Rue (2014) scaling constant. The
length of `γ` equals `n - r`, where `n` is the graph size and `r` is
the number of connected components.

The prior is obtained from the exponential-on-distance PC construction,
with

    d(φ) = sqrt(2 · KLD(φ))
    KLD(φ) = (1/2) Σ_k [ φ(1/γ_k - 1) - log(1 + φ(1/γ_k - 1)) ]

on the non-null subspace. Since `d` is bounded on `φ ∈ [0, 1]` (i.e.
`d(1) < ∞`), the exponential density on `d` is **truncated** to
`[0, d(1)]` and renormalised by `Z = 1 - exp(-λ · d(1))`. The rate
`λ > 0` is solved numerically so that

    P(φ < U) = (1 - exp(-λ · d(U))) / (1 - exp(-λ · d(1))) = α.

Internal scale is `θ = logit(φ)`; the Jacobian `φ(1-φ)` is included in
`log_prior_density`.

Defaults `U = 0.5, α = DEFAULT_BYM2_PHI_ALPHA` match Riebler et al.
(2016). See `plans/defaults-parity.md` for the ⚠ unverified-default
note.
"""
struct PCBYM2Phi{T <: Real, V <: AbstractVector{T}} <: AbstractHyperPrior
    U::T
    α::T
    λ::T          # exponential rate on d
    d_max::T      # d(1): upper bound of d(φ) on φ ∈ [0, 1]
    log_Z::T      # log-normaliser: log(1 - exp(-λ·d_max))
    γ::V          # non-zero eigenvalues of scaled structure matrix
end

function PCBYM2Phi(U::Real, α::Real, γ::AbstractVector{<:Real})
    0 < U < 1 || throw(ArgumentError("PCBYM2Phi: U must be in (0, 1), got U=$U"))
    0 < α < 1 || throw(ArgumentError("PCBYM2Phi: α must be in (0, 1), got α=$α"))
    isempty(γ) && throw(ArgumentError("PCBYM2Phi: γ must be non-empty"))
    any(<=(0), γ) && throw(ArgumentError("PCBYM2Phi: γ entries must be strictly positive"))

    T = typeof(float(U * α * first(γ)))
    γT = convert(AbstractVector{T}, γ)
    dU = _bym2_phi_distance(T(U), γT)
    iszero(dU) && throw(ArgumentError("PCBYM2Phi: d(U) is zero; check γ (all eigenvalues equal 1?)"))
    # d is bounded on [0, 1]; compute d_max = d(1) and solve for λ in
    # the truncated/renormalised exponential so that the quantile
    # equation P(φ < U) = α holds.
    d_max = _bym2_phi_distance(one(T), γT)
    λ = _solve_bym2_phi_rate(T(α), dU, d_max)
    log_Z = log1p(-exp(-λ * d_max))
    return PCBYM2Phi{T, typeof(γT)}(T(U), T(α), λ, d_max, log_Z, γT)
end

PCBYM2Phi(γ::AbstractVector{<:Real}; U::Real = 0.5,
          α::Real = DEFAULT_BYM2_PHI_ALPHA) = PCBYM2Phi(U, α, γ)

"""
Internal: solve `(1 - exp(-λ·d_U)) / (1 - exp(-λ·d_max)) = α` for
`λ > 0`. This is strictly monotone in `λ`: as `λ → 0⁺` the ratio
approaches `d_U/d_max`; as `λ → ∞` it approaches `1`. So a solution
with `λ > 0` exists iff `α > d_U/d_max`; when `α ≤ d_U/d_max` we
return the limit `λ = 0`, which corresponds to a uniform prior on `d`.
"""
function _solve_bym2_phi_rate(α::T, d_U::T, d_max::T) where {T <: Real}
    # Small-λ limit: ratio → d_U/d_max
    if α ≤ d_U / d_max
        # Uniform-on-d limit; return λ ≈ 0 (density becomes 1/d_max on d).
        return zero(T)
    end
    # Bisection on λ — numerically robust and doesn't need Roots.jl.
    f(λ) = (one(T) - exp(-λ * d_U)) / (one(T) - exp(-λ * d_max)) - α
    # Bracket: f is decreasing in λ. At λ→0, f → d_U/d_max - α < 0 (when we got here). Wait — we have α > d_U/d_max, so at λ→0 the ratio is d_U/d_max < α, so f < 0. At λ=∞, ratio → 1 > α, so f > 0. So bracket [ε, λ_hi] with f(ε) < 0 < f(λ_hi).
    lo = T(1.0e-8)
    hi = one(T)
    while f(hi) < 0 && hi < T(1.0e8)
        hi *= 2
    end
    for _ in 1:100
        mid = (lo + hi) / 2
        fm = f(mid)
        if abs(fm) < T(1.0e-12) || hi - lo < T(1.0e-12) * max(one(T), hi)
            return mid
        end
        if fm < 0
            lo = mid
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end

prior_name(::PCBYM2Phi) = :pc_bym2_phi

user_scale(::PCBYM2Phi, θ) = inv(one(θ) + exp(-θ))   # logistic σ(θ)

"""
Internal: KLD on the non-null subspace between the alternative BYM2
covariance `(1-φ) I + φ R_scaled^{-1}` and the base `I`, in terms of
the eigenvalues `γ_k > 0` of `R_scaled`.

    KLD(φ) = (1/2) Σ_k [ φ(1/γ_k - 1) - log1p(φ(1/γ_k - 1)) ]

Uses `log1p` for numerical accuracy when `u = φ(1/γ_k - 1)` is small.
"""
function _bym2_phi_kld(φ::Real, γ::AbstractVector{<:Real})
    T = promote_type(typeof(φ), eltype(γ))
    s = zero(T)
    @inbounds for γk in γ
        u = φ * (inv(γk) - one(T))
        s += u - log1p(u)
    end
    return s / 2
end

"""
Distance `d(φ) = sqrt(2 · KLD(φ))`.
"""
function _bym2_phi_distance(φ::Real, γ::AbstractVector{<:Real})
    k = _bym2_phi_kld(φ, γ)
    # KLD is non-negative up to floating-point noise; clamp below to 0.
    k = max(k, zero(k))
    return sqrt(2 * k)
end

"""
Internal: derivative `d'(φ) / φ` — pulling out the leading factor of `φ`
so the ratio is well-defined at `φ → 0`. From

    KLD'(φ) = (φ/2) Σ_k (1/γ_k - 1)^2 / (1 + φ(1/γ_k - 1))
    d'(φ)   = KLD'(φ) / d(φ)

we get, at finite `φ > 0`,

    d'(φ) = KLD'(φ)/d(φ).

At `φ → 0`, `d(φ) ~ A φ` with
`A^2 = (1/2) Σ_k (1/γ_k - 1)^2`, and `d'(φ) → A`.
"""
function _bym2_phi_log_abs_dφ(φ::Real, γ::AbstractVector{<:Real})
    T = promote_type(typeof(φ), eltype(γ))
    # Σ_k (1/γ_k - 1)^2 / (1 + φ(1/γ_k - 1)) — proportional to KLD'(φ)/φ
    s = zero(T)
    @inbounds for γk in γ
        u = inv(γk) - one(T)
        s += u^2 / (one(T) + φ * u)
    end
    # KLD'(φ) = (φ/2) s
    # d(φ)    = sqrt(2 KLD)
    # d'(φ)   = KLD'(φ) / d(φ)   when d > 0
    kld = _bym2_phi_kld(φ, γ)
    if kld > 0
        dφ = sqrt(2 * kld)
        dprime = (φ * s / 2) / dφ
        return log(dprime)
    else
        # φ → 0 limit: d'(0) = sqrt(Σ_k (1/γ_k - 1)^2 / 2)
        A2 = s / 2    # at φ=0 this equals (1/2) Σ (1/γ_k - 1)^2
        return log(sqrt(A2))
    end
end

"""
    log_prior_density(p::PCBYM2Phi, θ) -> Real

Log-density of the PC prior on `φ = σ(θ)` transformed to the internal
scale `θ = logit(φ)`. Comprises

- the PC density on `φ`:
    `log π_φ(φ) = log λ - λ d(φ) + log|d'(φ)|`
- the Jacobian `|dφ/dθ| = φ(1-φ)`.

At `θ → ±∞`, the prior mass goes to zero: `log π(θ) → -∞`.
"""
function log_prior_density(p::PCBYM2Phi, θ)
    φ = inv(one(θ) + exp(-θ))
    dφ = _bym2_phi_distance(φ, p.γ)
    log_dprime = _bym2_phi_log_abs_dφ(φ, p.γ)
    # Renormalised exponential on the bounded support [0, d_max]:
    # log π_φ(φ) = log λ - λ d(φ) - log(1 - exp(-λ d_max)) + log|d'(φ)|.
    # λ = 0 corresponds to uniform-on-d: log π_φ = -log d_max + log|d'(φ)|.
    if iszero(p.λ)
        log_πφ = -log(p.d_max) + log_dprime
    else
        log_πφ = log(p.λ) - p.λ * dφ - p.log_Z + log_dprime
    end
    # Jacobian dφ/dθ = φ(1-φ).
    log_jac = log(φ) + log1p(-φ)
    return log_πφ + log_jac
end

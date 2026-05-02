"""
    PCCor0(U = 0.5, α = 0.5)

Penalised-complexity prior on a correlation `ρ ∈ (-1, 1)` with reference
model at `ρ = 0` (independence). Mirrors R-INLA's `pc.cor0` — used for
the bivariate-IID correlation slot in `IIDND_Sep{2}` (and recursively for
each pairwise correlation in `IIDND_Sep{N}` for `N ≥ 3`).

The Kullback-Leibler distance from a bivariate normal with correlation
`ρ` and unit marginals to its `ρ = 0` counterpart is

    KLD(ρ) = -½ log(1 - ρ²),

so the natural PC distance is

    d(ρ) = √(2 · KLD(ρ)) = √(-log(1 - ρ²)).

The PC prior is exponential on `d(ρ)` with rate `λ`; because `d` is
symmetric under `ρ ↔ -ρ`, the density on `ρ ∈ (-1, 1)` carries a factor
of ½:

    π_ρ(ρ) = ½ λ exp(-λ d(ρ)) · |dd/dρ|,
    |dd/dρ| = |ρ| / ((1 - ρ²) d(ρ))    (for ρ ≠ 0).

The user-facing parameters are `(U, α)` with `P(|ρ| > U) = α`,
i.e. `α = exp(-λ · d(U))`, hence `λ = -log(α) / √(-log(1 - U²))`.

Internal scale is `θ = atanh(ρ)`, so `ρ = tanh(θ)` and
`|dρ/dθ| = 1 - ρ²`. The Jacobian cancels the explicit `1 - ρ²` in
`|dd/dρ|`, leaving

    log π_θ(θ) = log(λ) - log(2) - λ d(ρ)
                 - ½ log( -log(1 - ρ²) / ρ² )

where the bracketed ratio is well-defined at `ρ = 0` (limit equals
`1`) — handled by a Taylor short-circuit when `ρ²` is below a small
threshold.

Defaults `U = 0.5, α = 0.5` match R-INLA's `pc.cor0(0.5, 0.5)`
default.
"""
struct PCCor0{T <: Real} <: AbstractHyperPrior
    U::T
    α::T
    λ::T   # rate of the Exponential prior on d(ρ)

    function PCCor0{T}(U::T, α::T) where {T <: Real}
        0 < U < 1 || throw(ArgumentError("PCCor0: U must be in (0, 1), got U=$U"))
        0 < α < 1 || throw(ArgumentError("PCCor0: α must be in (0, 1), got α=$α"))
        d_U = sqrt(-log1p(-U * U))
        λ = -log(α) / d_U
        return new{T}(U, α, λ)
    end
end
function PCCor0(; U::Real=0.5, α::Real=0.5)
    T = typeof(float(U * α))
    return PCCor0{T}(T(U), T(α))
end
PCCor0(U::Real, α::Real) = PCCor0(; U=U, α=α)

prior_name(::PCCor0) = :pc_cor0

user_scale(::PCCor0, θ) = tanh(θ)

# Threshold below which we replace `log(-log1p(-ρ²)/ρ²)` with its
# Taylor expansion `ρ²/2 + ρ⁴/12 + …` to avoid the formal `0/0` at
# ρ = 0. The `d = sqrt(-log1p(-ρ²))` factor is computed exactly in
# both branches — only the ratio term needs the limit.
const _PC_COR0_RHO2_TOL = 1.0e-7

function log_prior_density(p::PCCor0, θ)
    ρ = tanh(θ)
    ρ² = ρ * ρ
    nlog = -log1p(-ρ²)            # = d²; well-defined and exact at ρ = 0
    d = sqrt(nlog)
    if ρ² < _PC_COR0_RHO2_TOL
        # Taylor: log(nlog/ρ²) = ρ²/2 + ρ⁴/12 + O(ρ⁶). Truncate at ρ²/2;
        # absolute error is bounded by ρ⁴/12 ≤ 1e-15 at the threshold.
        log_ratio = ρ² / 2
    else
        log_ratio = log(nlog / ρ²)
    end
    return log(p.λ) - log(2) - p.λ * d - 0.5 * log_ratio
end

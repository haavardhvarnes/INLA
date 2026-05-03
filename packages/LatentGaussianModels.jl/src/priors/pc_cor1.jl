"""
    PCCor1(U = 0.7, α = 0.7)

Penalised-complexity prior on a correlation `ρ ∈ (-1, 1)` with reference
model at `ρ = 1` (perfect positive correlation). Mirrors R-INLA's
`pc.cor1` — the textbook PC prior on the lag-1 correlation of an AR(1)
process, where `ρ = 1` is the random-walk limit.

Density on the user scale (R-INLA `inla.pc.dcor1`):

    π_ρ(ρ) = λ exp(-λ √(1-ρ)) / (1 - exp(-√2 λ)) · 1/(2√(1-ρ)),  ρ ∈ (-1, 1).

The "complexity distance" is `μ(ρ) = √(1-ρ)`, monotonically decreasing
from `√2` (at `ρ = -1`) to `0` (at `ρ = 1`); the density is exponential
in `μ` truncated to `μ ∈ (0, √2)`. Calibration is via `(U, α)` with
`P(ρ > U) = α`, i.e. `λ` is the unique positive root of

    (1 - exp(-λ √(1-U))) / (1 - exp(-√2 λ)) = α,    α > √((1-U)/2).

The lower bound on `α` enforces `λ > 0` — at `α = √((1-U)/2)` the rate
collapses to zero and the prior becomes uniform on `μ`.

Internal scale is `θ = atanh(ρ)` (matching `AR1`'s lag-1 correlation
parameterisation), so `ρ = tanh(θ)` and `|dρ/dθ| = 1 - ρ²`. Combining
the density and the Jacobian:

    log π_θ(θ) = log(λ) - λ √(1-ρ) - log(1 - exp(-√2 λ))
                 - log 2 + ½ log(1-ρ) + log(1+ρ).

Computations of `log(1-ρ)`, `log(1+ρ)`, and `√(1-ρ)` route through
`softplus` of `±2θ` so that `1 - tanh(θ)` underflowing to `0` for
`θ ≳ 19` does not break the prior at saturation.

# Defaults

`U = 0.7, α = 0.7` matches R-INLA's textbook example for `pc.cor1`
(also the AR(1) ρ-prior default suggested in INLA tutorials when the
user opts out of the built-in `Normal` on the logit scale).

# Notes

- Not a default for `IID2D` / `IID3D` / `2diid`: the bivariate-IID
  correlation slot uses `PCCor0` (reference at `ρ = 0`) — see ADR-022.
- R-INLA's actual `f(., model="ar1")` default is a Normal prior on
  `2 atanh(ρ)` with precision 0.15, *not* `pc.cor1`. Pass
  `ρprior = PCCor1(...)` to `AR1` to opt into this prior.
"""
struct PCCor1{T <: Real} <: AbstractHyperPrior
    U::T
    α::T
    λ::T   # rate of the truncated-exponential prior on μ(ρ) = √(1-ρ)
    log_norm::T  # cached `log(1 - exp(-√2 λ))` (η-independent)

    function PCCor1{T}(U::T, α::T) where {T <: Real}
        -1 < U < 1 || throw(ArgumentError("PCCor1: U must be in (-1, 1), got U=$U"))
        α_min = sqrt((1 - U) / 2)
        α_min < α < 1 ||
            throw(ArgumentError("PCCor1: α must satisfy √((1-U)/2) = $α_min < α < 1, got α=$α"))
        λ = _pc_cor1_solve_lambda(U, α)
        log_norm = log(-expm1(-sqrt(T(2)) * λ))
        return new{T}(U, α, λ, log_norm)
    end
end
function PCCor1(; U::Real=0.7, α::Real=0.7)
    T = typeof(float(U * α))
    return PCCor1{T}(T(U), T(α))
end
PCCor1(U::Real, α::Real) = PCCor1(; U=U, α=α)

prior_name(::PCCor1) = :pc_cor1
user_scale(::PCCor1, θ) = tanh(θ)

# Stable `log(1 + exp(x))` for any finite `x`. Required to keep the
# `1 - tanh(θ)` factor finite for `|θ| ≳ 19` where the user-scale
# expression underflows.
function _softplus(x::Real)
    return x > zero(x) ? x + log1p(exp(-x)) : log1p(exp(x))
end

# Numerical bisection for `λ > 0` such that
#   F(λ) = (1 - exp(-a λ)) / (1 - exp(-b λ)) = α,
# with `a = √(1-U)`, `b = √2`. `F` is monotonically increasing in `λ`
# on `(0, ∞)` from `α_min = a/b = √((1-U)/2)` to `1`, so bisection on
# the geometric mean (log-bisection) converges in O(log₂(λ_hi/λ_lo))
# halvings to floating-point precision.
function _pc_cor1_solve_lambda(U::Real, α::Real)
    a = sqrt(1 - U)
    b = sqrt(oftype(a, 2))
    F(λ) = -expm1(-a * λ) / -expm1(-b * λ)
    lo = oftype(a, 1.0e-10)
    hi = oftype(a, 1.0e10)
    for _ in 1:200
        mid = sqrt(lo * hi)  # geometric-mean = log-bisection step
        if F(mid) < α
            lo = mid
        else
            hi = mid
        end
        hi <= lo * (1 + oftype(a, 1.0e-14)) && break
    end
    return sqrt(lo * hi)
end

function log_prior_density(p::PCCor1, θ)
    sp_pos = _softplus(2θ)        # log(1 + exp(2θ))
    sp_neg = _softplus(-2θ)       # log(1 + exp(-2θ))
    log2 = log(oftype(sp_pos, 2))
    log_1m_ρ = log2 - sp_pos      # log(1 - tanh θ); stable for θ ≳ 19
    log_1p_ρ = log2 - sp_neg      # log(1 + tanh θ); stable for θ ≲ -19
    sqrt_1m_ρ = sqrt(oftype(sp_pos, 2)) * exp(-sp_pos / 2)  # √(1 - tanh θ)
    return log(p.λ) - p.λ * sqrt_1m_ρ - p.log_norm - log2 +
           oftype(sp_pos, 0.5) * log_1m_ρ + log_1p_ρ
end

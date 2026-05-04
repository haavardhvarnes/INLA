"""
    GEVLikelihood(; link = IdentityLink(),
                  precision_prior = GammaPrecision(1.0, 5.0e-5),
                  shape_prior = GaussianPrior(0.0, 2.0),
                  xi_scale = 0.1,
                  weights = nothing)

Generalised Extreme Value (GEV) observation model in R-INLA's
`family = "gev"` parameterisation. The CDF is

    F(y; η, τ, ξ) = exp(− [1 + ξ √(τ s) (y − η)]^(−1/ξ))

defined on the half-line `1 + ξ √(τ s) (y − η) > 0`, where `η`
is the linear predictor (location), `τ > 0` the precision (so the
GEV scale is `σ = 1/√(τ s)`), `s > 0` a fixed per-observation weight
(`weights[i]`, default 1), and `ξ ∈ ℝ` the shape parameter.

Two hyperparameters on the internal scale:
- `θ[1] = log τ`
- `θ[2] = ξ / xi_scale` — internal scaling of the shape parameter
  (default `xi_scale = 0.1`, matching R-INLA's `gev.scale.xi`).

Defaults match R-INLA's `family = "gev"`:
- `precision_prior = GammaPrecision(1, 5e-5)` ↔ `loggamma(1, 5e-5)` on `log τ`.
- `shape_prior = GaussianPrior(0, 2.0)`. R-INLA's documented default is
  `gaussian(0, prec = 25)`, applied to the **user-scale** ξ, not to the
  internal `θ[2]`. Reparameterising (since `ξ = xi_scale · θ[2]`) gives a
  Gaussian on `θ[2]` with `prec = 25 · xi_scale² = 0.25` for the default
  `xi_scale = 0.1`, i.e. σ = 2.0. If you change `xi_scale`, scale this
  σ by `0.1/xi_scale` to keep the user-scale prior identical.
- `xi_scale = 0.1`.

R-INLA's `gev` family is marked "disabled" in current releases in
favour of `bgev`. The two share the body parameterisation; bgev adds
a smooth tail blend to guarantee finite support. Use this likelihood
for body-of-distribution fits or replicating older R-INLA results.

Note: GEV is *not* globally log-concave. The inner Newton step may
require damping when the initial η falls near the support boundary;
ensure `initial_η` is well inside `1 + ξ √(τ s)(y − η) > 0` for all
observations.
"""
struct GEVLikelihood{L <: AbstractLinkFunction,
    Pτ <: AbstractHyperPrior,
    Pξ <: AbstractHyperPrior,
    W <: Union{Nothing, AbstractVector{<:Real}}} <: AbstractLikelihood
    link::L
    precision_prior::Pτ
    shape_prior::Pξ
    xi_scale::Float64
    weights::W
end

function GEVLikelihood(; link::AbstractLinkFunction=IdentityLink(),
        precision_prior::AbstractHyperPrior=GammaPrecision(1.0, 5.0e-5),
        shape_prior::AbstractHyperPrior=GaussianPrior(0.0, 2.0),
        xi_scale::Real=0.1,
        weights::Union{Nothing, AbstractVector{<:Real}}=nothing)
    link isa IdentityLink ||
        throw(ArgumentError("GEVLikelihood: only IdentityLink is supported, got $(typeof(link))"))
    xi_scale > 0 ||
        throw(ArgumentError("GEVLikelihood: xi_scale must be > 0, got $xi_scale"))
    if weights !== nothing
        all(>(0), weights) ||
            throw(ArgumentError("GEVLikelihood: weights must all be > 0"))
    end
    return GEVLikelihood(link, precision_prior, shape_prior, Float64(xi_scale), weights)
end

link(ℓ::GEVLikelihood) = ℓ.link
nhyperparameters(::GEVLikelihood) = 2
# Start above the Gumbel cusp at θ[2] = 0: the closed-form gradient
# branch flips there and AD through the runtime conditional reads as
# ∂/∂θ[2] = 0, which would let the outer optimizer terminate prematurely.
# θ[2] = 1 ⇒ ξ ≈ 0.1 (default xi_scale), well inside the smooth branch.
initial_hyperparameters(::GEVLikelihood) = [4.0, 1.0]   # τ ≈ 54.6, ξ ≈ 0.1
function log_hyperprior(ℓ::GEVLikelihood, θ)
    return log_prior_density(ℓ.precision_prior, θ[1]) +
           log_prior_density(ℓ.shape_prior, θ[2])
end

@inline _gev_weight(ℓ::GEVLikelihood, i) =
    ℓ.weights === nothing ? 1.0 : Float64(ℓ.weights[i])

# --- identity-link closed forms ---------------------------------------
# Let z_i = √(τ s_i) (y_i − η_i), A_i = 1 + ξ z_i (must be > 0),
# B_i = A_i^(−1/ξ). For |ξ| > eps:
#   log p_i = ½ log(τ s_i) − (1 + 1/ξ) log A_i − B_i
#   ∂   log p_i / ∂η_i  =  √(τ s_i) · (1 + ξ − B_i) / A_i
#   ∂²  log p_i / ∂η_i² = (τ s_i) · (1 + ξ) · (ξ − B_i) / A_i²
# Gumbel (ξ → 0) limit: log p = ½ log(τ s) − z − exp(−z), with
#   ∂log p/∂η = √(τ s) (1 − exp(−z))
#   ∂²log p/∂η² = − τ s · exp(−z)
# implemented via a small-|ξ| guard so derivatives stay continuous.
# (∂³ inherits the abstract finite-difference fallback.)

const _GEV_XI_EPS = 1.0e-6

function log_density(ℓ::GEVLikelihood{IdentityLink}, y, η, θ)
    τ = exp(θ[1])
    ξ = ℓ.xi_scale * θ[2]
    s = zero(promote_type(eltype(η), eltype(y), Float64))
    use_gumbel = abs(ξ) < _GEV_XI_EPS
    @inbounds for i in eachindex(y)
        w = _gev_weight(ℓ, i)
        sqτw = sqrt(τ * w)
        z = sqτw * (y[i] - η[i])
        if use_gumbel
            s += 0.5 * log(τ * w) - z - exp(-z)
        else
            A = 1 + ξ * z
            A > 0 || return -Inf * one(s)
            B = A^(-1 / ξ)
            s += 0.5 * log(τ * w) - (1 + 1 / ξ) * log(A) - B
        end
    end
    return s
end

function ∇_η_log_density(ℓ::GEVLikelihood{IdentityLink}, y, η, θ)
    τ = exp(θ[1])
    ξ = ℓ.xi_scale * θ[2]
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    use_gumbel = abs(ξ) < _GEV_XI_EPS
    @inbounds for i in eachindex(y)
        w = _gev_weight(ℓ, i)
        sqτw = sqrt(τ * w)
        z = sqτw * (y[i] - η[i])
        if use_gumbel
            out[i] = sqτw * (1 - exp(-z))
        else
            A = 1 + ξ * z
            if A <= 0
                out[i] = zero(eltype(out))
            else
                B = A^(-1 / ξ)
                out[i] = sqτw * (1 + ξ - B) / A
            end
        end
    end
    return out
end

function ∇²_η_log_density(ℓ::GEVLikelihood{IdentityLink}, y, η, θ)
    τ = exp(θ[1])
    ξ = ℓ.xi_scale * θ[2]
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    use_gumbel = abs(ξ) < _GEV_XI_EPS
    @inbounds for i in eachindex(y)
        w = _gev_weight(ℓ, i)
        τw = τ * w
        sqτw = sqrt(τw)
        z = sqτw * (y[i] - η[i])
        if use_gumbel
            out[i] = -τw * exp(-z)
        else
            A = 1 + ξ * z
            if A <= 0
                # Outside support: zero curvature placeholder; the inner
                # Newton step ought to never visit such points provided
                # the initial η is well inside the support.
                out[i] = zero(eltype(out))
            else
                B = A^(-1 / ξ)
                out[i] = τw * (1 + ξ) * (ξ - B) / A^2
            end
        end
    end
    return out
end

# --- pointwise log-density (CPO/WAIC) ---------------------------------

function pointwise_log_density(ℓ::GEVLikelihood{IdentityLink}, y, η, θ)
    τ = exp(θ[1])
    ξ = ℓ.xi_scale * θ[2]
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    use_gumbel = abs(ξ) < _GEV_XI_EPS
    @inbounds for i in eachindex(y)
        w = _gev_weight(ℓ, i)
        sqτw = sqrt(τ * w)
        z = sqτw * (y[i] - η[i])
        if use_gumbel
            out[i] = 0.5 * log(τ * w) - z - exp(-z)
        else
            A = 1 + ξ * z
            if A <= 0
                out[i] = -Inf
            else
                B = A^(-1 / ξ)
                out[i] = 0.5 * log(τ * w) - (1 + 1 / ξ) * log(A) - B
            end
        end
    end
    return out
end

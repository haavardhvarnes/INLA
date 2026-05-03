"""
    AR1(n; precprior = PCPrecision(), ρprior = nothing)

Stationary AR(1) latent field of length `n`. Two hyperparameters:

- `τ` (marginal precision) on internal scale `θ₁ = log(τ)`.
- `ρ ∈ (-1, 1)` (lag-1 correlation) on internal scale
  `θ₂ = atanh(ρ) = ½ log((1 + ρ)/(1 - ρ))`.

`precprior` is a scalar prior on `θ₁` (default PC prior on τ).
`ρprior` defaults to a Normal(0, σ=1) on `θ₂` — the simple
diffuse-on-Fisher-z choice used in many INLA tutorials. R-INLA's
*built-in* `f(., model="ar1")` default differs: a Normal on
`logit(ρ) = 2 atanh(ρ)` with precision 0.15 (i.e. σ ≈ 2.582 on
`logit(ρ)`, equivalently σ ≈ 1.291 on `atanh(ρ)`) — close to but not
identical to our default. Pass `ρprior = PCCor1(U, α)` to opt into
the textbook PC prior on `ρ` with reference at `ρ = 1` (Sørbye-Rue
2017), or any other `AbstractHyperPrior` to override.

Parameterization follows Rue & Held (2005, Eq. 1.39) so that
`Var(x_i) = 1/τ` at every index.
"""
struct AR1{P1 <: AbstractHyperPrior, P2 <: AbstractHyperPrior} <: AbstractLatentComponent
    n::Int
    precprior::P1
    ρprior::P2
end

"""
Internal-scale Normal prior on the Fisher-transformed correlation.
Kept local to this file because the Normal-on-θ case recurs only here.
"""
struct _NormalAR1ρ <: AbstractHyperPrior
    μ::Float64
    σ::Float64
end
prior_name(::_NormalAR1ρ) = :normal_atanh_ρ
user_scale(::_NormalAR1ρ, θ) = tanh(θ)
function log_prior_density(p::_NormalAR1ρ, θ)
    return -0.5 * log(2π) - log(p.σ) - 0.5 * ((θ - p.μ) / p.σ)^2
end

function AR1(n::Integer;
        precprior::AbstractHyperPrior=PCPrecision(),
        ρprior::AbstractHyperPrior=_NormalAR1ρ(0.0, 1.0))
    n ≥ 2 || throw(ArgumentError("AR1: n must be ≥ 2"))
    return AR1(Int(n), precprior, ρprior)
end

Base.length(c::AR1) = c.n
nhyperparameters(::AR1) = 2
initial_hyperparameters(::AR1) = [0.0, 0.0]   # log τ = 0, atanh ρ = 0 (ρ = 0)

function gmrf(c::AR1, θ)
    τ = exp(θ[1])
    ρ = tanh(θ[2])
    return GMRFs.AR1GMRF(c.n; ρ=ρ, τ=τ)
end

precision_matrix(c::AR1, θ) = GMRFs.precision_matrix(gmrf(c, θ))

function log_hyperprior(c::AR1, θ)
    return log_prior_density(c.precprior, θ[1]) +
           log_prior_density(c.ρprior, θ[2])
end

GMRFs.constraints(::AR1) = GMRFs.NoConstraint()

# Proper, full-rank Q with `log|Q| = n log(τ) - (n-1) log(1 - ρ²)` for
# the Rue-Held (2005, Eq. 1.39) parameterisation `Q = (τ/(1 - ρ²)) S`,
# where the AR1 correlation matrix has determinant `(1 - ρ²)^(n-1)`.
function log_normalizing_constant(c::AR1, θ)
    ρ = tanh(θ[2])
    return -0.5 * c.n * log(2π) +
           0.5 * (c.n * θ[1] - (c.n - 1) * log1p(-ρ^2))
end

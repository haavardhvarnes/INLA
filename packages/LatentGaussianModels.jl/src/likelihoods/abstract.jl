"""
    AbstractLikelihood

The observation model `y | η, θ`. `η` is the linear predictor — the
sum of latent-component contributions — and `θ` carries any
likelihood-specific hyperparameters (Gaussian precision, NegBinomial
size, Gamma shape). Likelihoods without hyperparameters ignore `θ`.

Required methods:

- `log_density(ℓ, y, η, θ) -> Real` — scalar `log p(y | η, θ)`.
- `∇_η_log_density(ℓ, y, η, θ) -> AbstractVector{<:Real}` — gradient
  w.r.t. `η`, same length as `y`.
- `∇²_η_log_density(ℓ, y, η, θ) -> AbstractVector{<:Real}` — diagonal
  of the Hessian w.r.t. `η`, same length as `y`. The likelihood
  factorises across observations so the Hessian is diagonal.
- `link(ℓ) -> AbstractLinkFunction`.
- `nhyperparameters(ℓ) -> Int` — number of scalar hyperparameters
  attached to the likelihood (default 0).

Return-type conventions are load-bearing for the inner Newton hot
path: `∇_η_log_density` and `∇²_η_log_density` return a plain
`AbstractVector{<:Real}`, **not** `Diagonal`, **not**
`SparseMatrixCSC`. The caller wraps with `Diagonal(...)` only where
algebraically required.
"""
abstract type AbstractLikelihood end

"""
    log_density(ℓ::AbstractLikelihood, y, η, θ) -> Real

Scalar log-density `log p(y | η, θ)`.
"""
function log_density end

"""
    ∇_η_log_density(ℓ::AbstractLikelihood, y, η, θ) -> AbstractVector

Gradient of `log p(y | η, θ)` w.r.t. `η`.
"""
function ∇_η_log_density end

"""
    ∇²_η_log_density(ℓ::AbstractLikelihood, y, η, θ) -> AbstractVector

Diagonal of the Hessian of `log p(y | η, θ)` w.r.t. `η`. Length equals
`length(y)`.
"""
function ∇²_η_log_density end

"""
    ∇³_η_log_density(ℓ::AbstractLikelihood, y, η, θ) -> AbstractVector

Diagonal of the third derivative tensor of `log p(y | η, θ)` w.r.t.
`η`. Length equals `length(y)`. The likelihood factorises across
observations, so only the diagonal entry `∂³ log p(y_i|η_i) / ∂η_i^3`
is non-zero.

Needed for Rue-Martino-Chopin (2009) simplified Laplace / skew
correction of posterior marginals `π(x_i | y)`. Concrete likelihoods
should override with closed forms; the default falls back on a
central finite-difference of `∇²_η_log_density` per coordinate.
"""
function ∇³_η_log_density(ℓ::AbstractLikelihood, y, η, θ)
    T = promote_type(eltype(η), Float64)
    n = length(y)
    out = Vector{T}(undef, n)
    # Per-coordinate central difference because the likelihood factorises —
    # each observation's third derivative only depends on its own η_i.
    h = cbrt(eps(T))
    η_p = copy(η)
    η_m = copy(η)
    @inbounds for i in 1:n
        step = max(h * abs(η[i]), h)
        η_p[i] = η[i] + step
        η_m[i] = η[i] - step
        h²p = ∇²_η_log_density(ℓ, y, η_p, θ)[i]
        h²m = ∇²_η_log_density(ℓ, y, η_m, θ)[i]
        out[i] = (h²p - h²m) / (2 * step)
        η_p[i] = η[i]
        η_m[i] = η[i]
    end
    return out
end

"""
    link(ℓ::AbstractLikelihood) -> AbstractLinkFunction

The link function attached to this likelihood instance.
"""
function link end

"""
    nhyperparameters(ℓ::AbstractLikelihood) -> Int

Number of scalar hyperparameters attached to the likelihood. Defaults
to `0`; Gaussian/NegBinomial/Gamma override.
"""
nhyperparameters(::AbstractLikelihood) = 0

"""
    initial_hyperparameters(ℓ::AbstractLikelihood) -> Vector

Initial values on the *internal* (unconstrained real-valued) scale. A
likelihood with a Gaussian precision τ uses `log(τ)` internally; this
function returns `[0.0]` (i.e. `τ = 1` at init).
"""
initial_hyperparameters(::AbstractLikelihood) = Float64[]

"""
    log_hyperprior(ℓ::AbstractLikelihood, θ) -> Real

Log-prior density for the likelihood's hyperparameters, evaluated on
the internal scale. Zero when `nhyperparameters(ℓ) == 0`.
"""
log_hyperprior(::AbstractLikelihood, θ) = zero(eltype(θ))

"""
    pointwise_log_density(ℓ, y, η, θ) -> AbstractVector{<:Real}

Per-observation log-density `log p(y_i | η_i, θ)`, length `length(y)`.
Likelihoods factorise across observations, so the sum of this vector
equals `log_density(ℓ, y, η, θ)`.

Needed by WAIC, CPO, DIC, and PIT diagnostics. Default implementation
falls back on `log_density` evaluated on singletons — correct but
inefficient; concrete likelihoods should override with a vectorised
form.
"""
function pointwise_log_density(ℓ::AbstractLikelihood, y, η, θ)
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        out[i] = log_density(ℓ, @view(y[i:i]), @view(η[i:i]), θ)
    end
    return out
end

"""
    pointwise_cdf(ℓ, y, η, θ) -> AbstractVector{<:Real}

Per-observation CDF `F(y_i | η_i, θ) = P(Y_i ≤ y_i | η_i, θ)`, length
`length(y)`. Needed for PIT diagnostics. Not all likelihoods have a
closed-form CDF — the default raises, and concrete likelihoods that
support PIT implement this method.
"""
function pointwise_cdf(ℓ::AbstractLikelihood, y, η, θ)
    throw(ArgumentError("pointwise_cdf not implemented for $(typeof(ℓ)); " *
                        "needed for PIT diagnostics"))
end

"""
    add_copy_contributions!(η_block, ℓ::AbstractLikelihood, x, θ_ℓ) -> η_block

Add this likelihood's `Copy`-component contributions to its slice of the
linear predictor, in place, and return `η_block`.

`Copy` (R-INLA `f(..., copy=...)`) shares a latent component across
linear predictors with an estimated scaling β: the receiving block
gets `η_target[i] += β * x_source[k(i)]`. ADR-021 places β on the
*receiving* likelihood (rather than on the projection mapping), so
each likelihood that opts in stores its `(source_indices, β_index)`
pairs and reads β from its own `θ_ℓ` slice.

The default no-op covers every likelihood without copies — Gaussian,
Poisson, Binomial, NegBinomial, Gamma, the survival likelihoods on
arms that don't share latents — and leaves `η_block` unchanged. The
inner Newton loop calls this hook after every `η = mapping * x`
evaluation, before computing likelihood derivatives.

Implementing likelihoods receive an `η_block` view spanning their
observation rows, the *full* latent vector `x` (so they can index any
source component), and their hyperparameter view `θ_ℓ` (where β
lives).
"""
add_copy_contributions!(η::AbstractVector, ::AbstractLikelihood, ::AbstractVector, θ_ℓ) = η

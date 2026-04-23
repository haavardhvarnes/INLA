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

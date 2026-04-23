"""
    AbstractHyperPrior

Prior on a scalar hyperparameter. All priors are expressed on the
*internal* unconstrained scale θ ∈ ℝ; the transformation from the
user-facing parameter (precision τ > 0, correlation ρ ∈ (-1, 1),
mixing φ ∈ [0, 1]) to θ is fixed by the component that owns the
hyperparameter (see `plans/defaults-parity.md`).

Concrete subtypes implement:

- `log_prior_density(prior, θ)` — `log π(θ)` on the internal scale,
  **including the Jacobian** of the internal-to-user transformation.
- `user_scale(prior, θ)` — convert internal `θ` to the user-facing
  parameter for reporting.
- `prior_name(prior) -> Symbol` — short identifier for printing.

This type is for *scalar* priors. Multi-dimensional priors (e.g. the
PC prior on (σ, range) for SPDE) live in `INLASPDE.jl` because they
are inherently coupled.
"""
abstract type AbstractHyperPrior end

"""
    log_prior_density(prior::AbstractHyperPrior, θ) -> Real

Log-prior density at internal-scale value `θ`, including Jacobian.
"""
function log_prior_density end

"""
    user_scale(prior::AbstractHyperPrior, θ) -> Real

Map internal-scale θ to the user-facing parameter (e.g. `exp(θ)` for
a log-precision prior).
"""
function user_scale end

"""
    prior_name(prior::AbstractHyperPrior) -> Symbol
"""
function prior_name end

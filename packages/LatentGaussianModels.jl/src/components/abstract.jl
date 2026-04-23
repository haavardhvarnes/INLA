"""
    AbstractLatentComponent

The Bayesian wrapper around a GMRF — the LGM equivalent of R-INLA's
`f(...)` term. A concrete component composes a `GMRFs.AbstractGMRF`
(or builds one lazily from θ) with an `AbstractHyperPrior` and any
constraint metadata.

Required methods:

- `length(c)` — dimension of the component's latent field.
- `initial_hyperparameters(c) -> Vector` — internal-scale start values.
- `nhyperparameters(c) -> Int` — convenience: `length(initial_hyperparameters(c))`.
- `precision_matrix(c, θ)` — sparse precision matrix as a function of
  this component's slice of θ.
- `log_hyperprior(c, θ)` — log-prior density at θ (on internal scale).

Optional (with default implementations):

- `prior_mean(c, θ)` — defaults to zeros.
- `constraints(c)` — defaults to `NoConstraint()`.
- `graph(c)` — the underlying GMRF graph if any.
- `gmrf(c, θ)` — the `AbstractGMRF` at θ, default built from
  `precision_matrix(c, θ)` via a `Generic0GMRF` wrapper.
"""
abstract type AbstractLatentComponent end

Base.length(::AbstractLatentComponent) = error("length must be implemented for concrete AbstractLatentComponent")

"""
    nhyperparameters(c::AbstractLatentComponent) -> Int
"""
nhyperparameters(c::AbstractLatentComponent) = length(initial_hyperparameters(c))

"""
    precision_matrix(c::AbstractLatentComponent, θ)

Sparse symmetric precision matrix at `θ` (this component's slice).
"""
function precision_matrix end

"""
    log_hyperprior(c::AbstractLatentComponent, θ)

Log-prior density on the internal scale; includes any Jacobians from
the user-to-internal transformation. Zero-hyperparameter components
return `0`.
"""
log_hyperprior(::AbstractLatentComponent, θ) = zero(eltype(θ))

"""
    initial_hyperparameters(c::AbstractLatentComponent)

Initial values on the internal unconstrained scale.
"""
function initial_hyperparameters end

"""
    prior_mean(c::AbstractLatentComponent, θ) -> AbstractVector

Prior mean vector. Default: zeros of length `length(c)`.
"""
prior_mean(c::AbstractLatentComponent, θ) = zeros(Float64, length(c))

"""
    GMRFs.constraints(c::AbstractLatentComponent) -> AbstractConstraint

Default hard linear constraint attached to the component. Intrinsic
components override; proper ones inherit `NoConstraint`.
"""
GMRFs.constraints(::AbstractLatentComponent) = NoConstraint()

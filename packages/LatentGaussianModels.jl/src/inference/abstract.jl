"""
    AbstractInferenceStrategy

Dispatch type for the `fit` entry point. Concrete strategies:

- `Laplace` — fit at fixed `θ`, return the Gaussian approximation
  `x | θ, y ≈ N(x̂, (Q + A' D A)⁻¹)` (`D` from the likelihood Hessian).
- `EmpiricalBayes` — plug-in estimate `θ̂ = argmax π(θ | y)` via the
  outer Laplace log-marginal, then Laplace at `θ̂`.
- `INLA` — the full thing (deferred).
"""
abstract type AbstractInferenceStrategy end

"""
    AbstractInferenceResult

Return type for `fit`.
"""
abstract type AbstractInferenceResult end

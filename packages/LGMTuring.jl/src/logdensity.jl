# LogDensityProblems bridge.
#
# The actual `INLALogDensity` struct lives in core LatentGaussianModels.jl
# (`src/inference/log_density.jl`) so it can be consumed by *any*
# LogDensityProblems sampler — Turing, Pathfinder, MCMCTempering — without
# pulling LGMTuring into the dependency graph. This file is a thin
# re-export so users who only know about LGMTuring still find it.
#
# A LGMTuring-specific constructor is provided as syntactic convenience —
# it forwards to the upstream constructor unchanged.

"""
    inla_log_density(model::LatentGaussianModel, y; laplace = Laplace())
        -> INLALogDensity

Build a `LogDensityProblems`-conforming wrapper around the INLA
hyperparameter posterior `log π(θ | y) ∝ log p(y | θ) + log π(θ)`.

Equivalent to calling `INLALogDensity(model, y)` from
`LatentGaussianModels`. Re-exposed here so HMC users don't have to import
two packages.

The returned object satisfies `LogDensityProblems.LogDensityOrder{1}()` —
both `logdensity` and `logdensity_and_gradient` are defined. The gradient
is computed by central finite differences (each evaluation is already
`O(Laplace)`-bounded), which is enough for NUTS at typical hyperparameter
dimensions.
"""
inla_log_density(model::LatentGaussianModel, y; laplace::Laplace=Laplace()) = INLALogDensity(
    model, y; laplace=laplace)

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

"""
    AbstractMarginalStrategy

Dispatch type for the per-coordinate posterior marginal density of
`x_i | y` and, in the INLA integration stage, for the per-θ
approximation of `x_mean` / `x_var`. Mirrors R-INLA's
`control.inla\$strategy`.

Concrete strategies (see ADR-026):

- [`Gaussian`](@ref) — Gaussian centred at the Newton mode, with
  constraint-corrected Laplace marginal variance. R-INLA's
  `strategy = "gaussian"`.
- [`SimplifiedLaplace`](@ref) — Newton mode plus the Rue-Martino mean
  shift in the integration stage; Edgeworth first-order skewness
  correction in the per-coordinate density. Reduces to `Gaussian`
  when the likelihood third derivative `∇³_η log p` is zero. R-INLA's
  `strategy = "simplified.laplace"`. Phase L will add `FullLaplace`
  (R-INLA's `strategy = "laplace"`) as a third subtype.

The mean-shift facet of `SimplifiedLaplace` (integration-stage) and
its density-skew facet (per-coordinate marginals) are independent —
they happen at different points in the pipeline and are dispatched on
the same type but through different internal hooks. See ADR-016 and
ADR-026.
"""
abstract type AbstractMarginalStrategy end

"""
    Gaussian()

Marginal strategy: per-θ Gaussian centred at the Newton mode, with
constraint-corrected Laplace marginal variance. R-INLA's
`strategy = "gaussian"`. Default for both
[`INLA`](@ref) (integration stage) and
[`posterior_marginal_x`](@ref) (per-coordinate density).

See [`AbstractMarginalStrategy`](@ref).
"""
struct Gaussian <: AbstractMarginalStrategy end

"""
    SimplifiedLaplace()

Marginal strategy: Rue-Martino mean shift in the integration stage,
plus Edgeworth first-order skewness correction in the per-coordinate
density. Reduces to [`Gaussian`](@ref) when the likelihood third
derivative `∇³_η log p` is zero. R-INLA's
`strategy = "simplified.laplace"`.

See [`AbstractMarginalStrategy`](@ref) and ADR-016.
"""
struct SimplifiedLaplace <: AbstractMarginalStrategy end

"""
    _resolve_marginal_strategy(s) -> AbstractMarginalStrategy

Backwards-compatibility shim accepting either an
[`AbstractMarginalStrategy`](@ref) instance (returned as-is) or a
symbol from the legacy whitelist (`:gaussian`, `:simplified_laplace`).
Mirrors `_resolve_scheme(::Symbol, ::Int)` for the integration-scheme
keyword.

Throws `ArgumentError` for unknown symbols.
"""
_resolve_marginal_strategy(s::AbstractMarginalStrategy) = s
function _resolve_marginal_strategy(s::Symbol)
    s === :gaussian && return Gaussian()
    s === :simplified_laplace && return SimplifiedLaplace()
    throw(ArgumentError("unknown marginal strategy :$s; " *
                        "use Gaussian(), SimplifiedLaplace(), " *
                        "or :gaussian / :simplified_laplace"))
end

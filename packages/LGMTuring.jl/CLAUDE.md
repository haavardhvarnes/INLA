# Guidance for Claude Code in LGMTuring.jl

Extends [`/CLAUDE.md`](../../CLAUDE.md). Scoped to the HMC/NUTS bridge.

## Scope

This package owns:
- The `sample(lgm::LatentGaussianModel, ::NUTS, n; kwargs...)` bridge
  built on AdvancedHMC / Turing.
- The INLA-within-MCMC loop (Gómez-Rubio Ch. 5–7 pattern) that uses
  LGM's public `sample_conditional(lgm, θ, y)`.
- `compare(inla_fit, nuts_chain)` — posterior-summary diff for
  triangulation-tier validation.
- `MCMCChains.Chains` output so the result plays with the rest of the
  Turing ecosystem.

Out of scope:
- Anything that should live in core LGM (the `LogDensityProblems`
  conformance is core's responsibility; we consume it).
- Non-HMC samplers. Those can live in their own sub-packages later if
  needed (e.g., `LGMPathfinder.jl`).
- Model construction. Users pass in an already-constructed
  `LatentGaussianModel`.

## Dependencies allowed

Core:
- `LatentGaussianModels` — the host; we dispatch on its types.
- `Turing` — top-level `sample` interface.
- `AdvancedHMC` — NUTS implementation.
- `MCMCChains` — output type.
- `LogDensityProblems`, `LogDensityProblemsAD` — gradient path.

Nothing else without an ADR.

## Key design decisions

- **`init_from_inla` default is `true`.** If the user has an `INLAResult`
  handy, use its posterior mean as HMC init; otherwise fall back to
  `initial_hyperparameters`. This avoids users reporting "NUTS didn't
  converge" when the real issue was cold-starting from a bad θ.
- **`rng` is always a kwarg.** No global RNG state.
- **AD backend defaults to ForwardDiff for dim(θ) < 20, ReverseDiff
  otherwise.** Configurable via `ADTypes`.

## Testing conventions

- `test/regression/` — on tiny known-posterior models (`IID` with
  conjugate prior), HMC recovers the analytic posterior within MC error.
- `test/oracle/` — for Scotland BYM2 and Meuse SPDE, pre-baked NUTS
  chains as JLD2. The actual chain sampling happens in
  `scripts/generate-fixtures/lgmturing/`, not at test time.
- `test/triangulation/` — the actual Stan/NIMBLE/Turing cross-validation
  lives here, not in core LGM.

## What not to do

- Do not reimplement HMC. AdvancedHMC is the right layer.
- Do not leak Turing types into core LGM's API surface. The seam is
  `LogDensityProblems`.
- Do not commit to a specific `MCMCChains` internal representation —
  Turing has broken this historically.

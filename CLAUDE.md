# Guidance for Claude Code in this repository

This file is read automatically by Claude Code sessions working in this
repository. It sets ecosystem-wide conventions. Per-package `CLAUDE.md` files
override and extend these for their scope.

## What this project is

A Julia-native reimplementation of the R-INLA latent Gaussian model / INLA
stack. The reference implementation is [`hrue/r-inla`](https://github.com/hrue/r-inla).
Canonical method references are in `references/papers.md`. Ecosystem-level
design lives in `plans/`; each package has its own `plans/plan.md`.

**Supported Julia version: 1.12+ only** (current stable). Julia 1.10 LTS
is *not* supported — see ADR-020 in `plans/decisions.md`. Don't add
back-compat shims for older versions; assume `Returns`, public marker,
and other 1.11+ features are available.

## Architectural principles — treat as load-bearing

1. **Dispatch over macros.** Multiple dispatch is the primary extension
   mechanism. A new latent component is a struct plus a handful of methods —
   never a macro-defined trace. Macros are permitted only as surface sugar
   over an explicit constructor API that works without them. See
   `plans/macro-policy.md`.

2. **Composability at the seams.** Consume ecosystem interfaces rather than
   owning types: GeoInterface.jl for geometry, Graphs.jl for topology,
   Meshes.jl for meshes, Tables.jl for data, Distributions.jl for priors and
   likelihoods, LinearSolve.jl for factorization backends.

3. **SciML style.** Follow the [SciML Style Guide](https://github.com/SciML/SciMLStyle).
   CamelCase for types, snake_case for functions. No globals. Abstract types
   named `AbstractX`. Prefer `Base.@kwdef` over long positional constructors.

4. **Weakdeps for optional integrations.** Anything not needed for the core
   numerical path goes in `[weakdeps]` + extensions: Rasters, Integrals,
   HCubature, MakieCore, Turing bridge. See `plans/dependencies.md`.

5. **Validate against R-INLA.** The four-tier testing strategy in
   `plans/testing-strategy.md` is how we prove correctness. Regression tests,
   R-INLA oracle fixtures, triangulation against Stan/NIMBLE, textbook
   end-to-end. Every non-trivial component needs at least tier 1 and tier 2.

## Things to avoid

- **`@model`-style macro APIs as the primary interface.** The LGM class is
  structural, not procedural; don't hide it inside a trace.
- **Owning geometry or graph types.** Use the ecosystem abstractions.
- **Heavy dependencies for convenience.** DataFrames, MLJ, Turing, Makie,
  Rasters — none of these belong in `[deps]` of the core packages. They may
  appear in `[weakdeps]` or test deps.
- **Silently different defaults from R-INLA.** If our BYM2 scaling, PC prior
  parameterization, or constraint convention differs, say so loudly in
  docstrings and flag it in `plans/defaults-parity.md`.
- **Reimplementing SuiteSparse.** Use LinearSolve.jl.
- **Globals, including random number state.** Take an `AbstractRNG`
  explicitly in any function that samples.

## Code conventions

- Type parameters where they matter for performance; no abstract field types
  in hot-path structs.
- `AbstractGMRF`, `AbstractLatentComponent`, `AbstractLikelihood`,
  `AbstractHyperPrior`, `AbstractInferenceStrategy`,
  `AbstractIntegrationScheme` — these are the load-bearing abstract types.
  Each concrete type's required methods are specified at the abstract type's
  docstring.
- Public API goes through each package's top-level module. Internal helpers
  live in submodules or unexported functions.
- Every exported function has a docstring with at least a one-line summary
  and an example.
- Errors use domain-specific types (`GMRFSingularError`, `PriorConstraintError`)
  rather than raw `error()` strings.

## Testing conventions

- `Test.jl` with `@testset`, no TestItems magic. Tests run with
  `Pkg.test()`.
- Per-package `test/` structure:
  - `test/regression/` — closed-form or hand-computed fixtures.
  - `test/oracle/` — R-INLA reference runs stored as JLD2 fixtures.
  - `test/triangulation/` — cross-checks against Stan/NIMBLE where applicable.
- Fixture generation scripts live in top-level `scripts/generate-fixtures/`
  with pinned R/R-INLA versions documented.
- Tolerances: 1e-10 for closed-form, 1% for means and 5% for hyperparameters
  against R-INLA, wider for triangulation.

## Dependency policy

See `plans/dependencies.md`. Do not add a new `[deps]` entry without
updating that document and recording the rationale in `plans/decisions.md`.

## When asked to write code

- Check the relevant package's `CLAUDE.md` and `plans/plan.md` first.
- If the task isn't in scope for a package, say so; do not expand scope.
- Prefer minimal, composable pieces. If a function is doing three things,
  split it.
- If a design decision is needed, write the ADR entry in
  `plans/decisions.md` before implementing.

## When asked about R-INLA

- Use `references/papers.md` for method references.
- Canonical datasets for validation: Scotland lip cancer, Germany oral
  cavity, Sardinia (disconnected components), Meuse (geostatistics/SPDE),
  Pennsylvania lung cancer, Tokyo rainfall, Seeds, Epil.
- Cross-validation repos: `spatialstatisticsupna/Comparing-R-INLA-and-NIMBLE`,
  `ConnorDonegan/Stan-IAR`.

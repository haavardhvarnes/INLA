# Julia INLA Ecosystem

A Julia-native reimplementation of the latent Gaussian model / INLA stack
originally provided by [R-INLA](https://www.r-inla.org/). The goal is not
line-by-line port but a composable, dispatch-based, SciML-aligned alternative
that covers the mainstream use cases of R-INLA with native performance and
genuine extensibility.

## Status

**Planning.** No source code yet. This repository currently holds design
documents, architectural decisions, and package scaffolding. See
[`ROADMAP.md`](ROADMAP.md) for phased milestones and
[`plans/`](plans/) for design decisions.

## Packages

### Core ecosystem

| Package | Purpose | Depends on |
|---|---|---|
| [`GMRFs.jl`](packages/GMRFs.jl/) | Sparse Gaussian Markov random fields: graph, precision, sampling, conditioning, log-density. Standalone-useful. | SparseArrays, LinearSolve, Graphs, Distributions, ChainRulesCore |
| [`LatentGaussianModels.jl`](packages/LatentGaussianModels.jl/) | LGM abstraction, latent components (IID/RW/AR1/Besag/BYM2/…), likelihoods, INLA inference. `LogDensityProblems` seam for downstream samplers. | GMRFs, NonlinearSolve, Optimization, LogDensityProblems |
| [`INLASPDE.jl`](packages/INLASPDE.jl/) | SPDE–Matérn finite-element assembly on Meshes.jl triangulations. | LatentGaussianModels, Meshes, DelaunayTriangulation, CoordRefSystems |

### Optional sub-packages (install separately)

Heavy integrations live outside the core to keep load times small and
release cadences independent. Install whichever you need.

| Package | Purpose | Adds dep on |
|---|---|---|
| [`LGMFormula.jl`](packages/LGMFormula.jl/) | Tier-2 `@lgm` formula sugar over `LatentGaussianModel(...)`. | StatsModels |
| [`LGMTuring.jl`](packages/LGMTuring.jl/) | HMC/NUTS bridge for cross-validation and INLA-within-MCMC flows. | Turing, AdvancedHMC |
| [`GMRFsPardiso.jl`](packages/GMRFsPardiso.jl/) | MKL / Panua Pardiso factorization backend for GMRFs.jl. License-gated. | Pardiso |
| [`INLASPDERasters.jl`](packages/INLASPDERasters.jl/) | Covariate extraction from rasters + raster prediction surfaces. | Rasters |

An umbrella package (tentatively `INLA.jl`) may be added once the core
three stabilize.

## Directory map

- `plans/` — ecosystem-level design documents (architecture, dependencies, testing, macro policy, ADR log).
- `references/` — annotated bibliography and notes on upstream INLA source.
- `benchmarks/` — cross-package validation runs against R-INLA, Stan, NIMBLE.
- `scripts/` — fixture generation and utilities.
- `packages/` — the Julia packages themselves, each with its own plan and `CLAUDE.md`.

## Working principles

See [`CLAUDE.md`](CLAUDE.md) for the full set. Short version:

1. Multiple dispatch is the primary extension mechanism. Macros are optional sugar, never semantics.
2. Compose existing ecosystem packages rather than owning types — GeoInterface, Graphs, Meshes, Distributions, LinearSolve.
3. Validation against R-INLA on canonical datasets is a first-class test tier, not an afterthought.
4. SciML code style; weakdeps/extensions for optional integrations.

## License

MIT (each package licensed independently, but all packages in the ecosystem use MIT).

# Julia INLA Ecosystem

A Julia-native reimplementation of the latent Gaussian model / INLA stack
originally provided by [R-INLA](https://www.r-inla.org/). The goal is not
line-by-line port but a composable, dispatch-based, SciML-aligned alternative
that covers the mainstream use cases of R-INLA with native performance and
genuine extensibility.

## Status

**`v0.1.0-rc1`.** The four `src/`-bearing packages
(`GMRFs.jl`, `LatentGaussianModels.jl`, `INLASPDE.jl`,
`INLASPDERasters.jl`) cover the canonical R-INLA datasets within the
testing-strategy tolerances. See [`CHANGELOG.md`](CHANGELOG.md) for
what landed and where R-INLA parity is known to be loose.

## Relationship to IntegratedNestedLaplace.jl

[`IntegratedNestedLaplace.jl`](https://github.com/JuliaApproxInference/IntegratedNestedLaplace.jl)
is a separate Julia package by a different author. It started earlier
and is best read as a research scratchpad: a small set of hand-written
LGMs (BivariateIID, NonStationarySPDE, …) used to explore Julia-native
INLA approximations. Several of those numerical kernels are excellent
references and we port them deliberately — the replan calls out
`BivariateIIDModel` and `NonStationarySPDEModel` for graduation in
Phase H/J, with attribution.

This ecosystem (`INLA.jl` + the four core packages above) is the
**production line**: the goal is breadth-of-R-INLA-parity, oracle-tested
on canonical datasets, with the dispatch-based component / likelihood /
inference-strategy seams documented in [`CLAUDE.md`](CLAUDE.md) and
[`plans/`](plans/). If you want to fit a model and trust the answer
against R-INLA, install from here.

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

The umbrella package [`INLA.jl`](packages/INLA.jl) re-exports
`GMRFs`, `LatentGaussianModels`, and `INLASPDE` so that `using INLA`
brings the inference stack into scope in one import.

## Installation

The packages are not yet on the General registry. They are registered
in a personal Julia registry. From a fresh Julia REPL:

```julia
using Pkg
Pkg.Registry.add(RegistrySpec(url = "https://github.com/haavardhvarnes/JuliaRegistry"))
Pkg.Registry.add("General")  # idempotent if already added
Pkg.add("INLA")              # umbrella: GMRFs + LatentGaussianModels + INLASPDE
```

For a leaner install, replace `"INLA"` with any individual core
package — `"GMRFs"`, `"LatentGaussianModels"`, `"INLASPDE"`, or
`"INLASPDERasters"`. `INLASPDERasters` is registered but its API is
scaffolding only in v0.1.0-rc1; see its
[README](packages/INLASPDERasters.jl/README.md).

The optional sub-packages `LGMTuring.jl`, `LGMFormula.jl`, and
`GMRFsPardiso.jl` are **not** registered yet. Install them by
`Pkg.develop`-ing this repo's subdir directly:

```julia
Pkg.develop(url = "https://github.com/haavardhvarnes/INLA.jl",
            subdir = "packages/LGMTuring.jl")
```

General-registry submission is planned; until then the personal
registry is the install path.

## What ships in v0.1.0-rc1

- **Latent components**: `Intercept`, `FixedEffects`, `IID`, `RW1`,
  `RW2`, `AR1`, `Seasonal`, `Besag`, `BYM`, `BYM2`, `Leroux`,
  `Generic0`, `Generic1`, `SPDE2` (α = 2 Matérn).
- **Likelihoods**: `Gaussian`, `Poisson`, `Binomial`,
  `NegativeBinomial`, `Gamma` — with closed-form gradients/Hessians
  for the inner Newton loop and a ForwardDiff fallback for
  user-defined cases.
- **Hyperpriors**: `PCPrecision`, `GammaPrecision`,
  `LogNormalPrecision`, `WeakPrior`, `PCBYM2Phi`, `LogitBeta`.
- **Inference strategies**: `EmpiricalBayes` (Laplace at θ̂), `INLA`
  (Laplace + θ-integration), and a `LogDensityProblems` seam for
  external samplers — `LGMTuring.jl` provides the NUTS bridge.
- **θ-integration schemes**: `Grid`, `GaussHermite`, `CCD` —
  `int_strategy = :auto` chooses CCD for dim θ > 2, Grid otherwise.
- **Diagnostics**: DIC, WAIC, CPO, PIT.

## Reproducing R-INLA parity

Eleven R-INLA oracle fixtures (Scotland and Pennsylvania BYM2,
classical BYM, synthetic Gamma / Negative Binomial / Generic0 /
Generic1 / Seasonal / Leroux / disconnected Besag, Meuse SPDE) are
checked into the test suite, each with the R-INLA posterior summaries
and `cpu.used` wall-clock embedded.

The reproducer script at
[`bench/oracle_compare.jl`](bench/oracle_compare.jl) runs every problem
end-to-end, prints a markdown table of relative errors and side-by-side
wall-clock seconds, and writes the full per-quantity comparison to
JSON. From the repo root:

```julia
julia --project=bench bench/oracle_compare.jl
```

See [`bench/README.md`](bench/README.md) for column meanings, expected
runtime (~3-5 minutes), and the JSON schema.

## Directory map

- `plans/` — ecosystem-level design documents (architecture, dependencies, testing, macro policy, ADR log).
- `references/` — annotated bibliography and notes on upstream INLA source.
- `bench/` — R-INLA parity reproducer (`oracle_compare.jl`) and its env.
- `benchmarks/` — placeholder for cross-package perf runs vs Stan/NIMBLE (Phase 0.2).
- `docs/` — Documenter site source (`docs/src/`); built site lives at `docs/build/` (gitignored).
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

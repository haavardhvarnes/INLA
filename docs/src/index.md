# Julia INLA Ecosystem

A Julia-native reimplementation of the latent Gaussian model / INLA
stack originally provided by [R-INLA](https://www.r-inla.org/). The
goal is not a line-by-line port but a composable, dispatch-based,
SciML-aligned alternative covering the mainstream R-INLA workflows
with native performance and genuine extensibility.

## Status

`v0.1.0-rc1`. Phase A through D of the [replan](https://github.com/HaavardHvarnes/INLA/blob/main/plans/replan-2026-04.md)
have landed: the four `src/`-bearing packages
([`GMRFs.jl`](packages/gmrfs.md),
[`LatentGaussianModels.jl`](packages/lgm.md),
[`INLASPDE.jl`](packages/inlaspde.md),
[`INLASPDERasters.jl`](packages/inlaspderasters.md)) cover the
canonical R-INLA datasets — Scotland and Pennsylvania BYM2, Meuse
zinc SPDE, the synthetic Negative Binomial / Gamma / Generic /
Seasonal / Leroux / disconnected-Besag suite — within the testing
strategy's tolerances. `LGMTuring.jl` provides the NUTS bridge for
INLA-vs-MCMC triangulation.

## What's here

```@contents
Pages = [
    "getting-started.md",
    "vignettes/scotland-bym2.md",
    "vignettes/tokyo-rainfall.md",
    "vignettes/meuse-spde.md",
    "packages/gmrfs.md",
    "packages/lgm.md",
    "packages/inlaspde.md",
    "packages/inlaspderasters.md",
    "references.md",
]
Depth = 2
```

## Installing

`v0.1.0` is not yet on the General registry. For the rc1 line, develop
the packages directly from a clone:

```julia
using Pkg
Pkg.develop([
    Pkg.PackageSpec(path = "packages/GMRFs.jl"),
    Pkg.PackageSpec(path = "packages/LatentGaussianModels.jl"),
    Pkg.PackageSpec(path = "packages/INLASPDE.jl"),
    Pkg.PackageSpec(path = "packages/INLASPDERasters.jl"),
])
```

Optional sub-packages (`LGMTuring.jl`, `LGMFormula.jl`,
`GMRFsPardiso.jl`) install separately — see the dependency notes in
each package's section.

## How to read this site

- **[Getting started](getting-started.md)** — the smallest possible
  Poisson + spatial random effect fit, top to bottom.
- **Vignettes** — three canonical R-INLA datasets, end-to-end:
  - [Scotland BYM2](vignettes/scotland-bym2.md) (areal Poisson with
    BYM2 spatial random effect),
  - [Tokyo rainfall](vignettes/tokyo-rainfall.md) (Bernoulli with a
    cyclic RW2 seasonal),
  - [Meuse zinc SPDE](vignettes/meuse-spde.md) (Gaussian on point-referenced
    data via the SPDE–Matérn link).
- **Packages** — per-package overviews, exported API, and required
  contracts for extending them.
- **[References](references.md)** — the canonical method papers and
  validation datasets we test against.

## Differences from R-INLA

The full list lives in
[`plans/defaults-parity.md`](https://github.com/HaavardHvarnes/INLA/blob/main/plans/defaults-parity.md).
Highlights:

- **Single dispatch table.** Every latent component, likelihood,
  prior, and integration scheme is a struct + a handful of methods.
  Adding a new component is "subtype `AbstractLatentComponent` and
  implement five methods" — no formula-DSL gatekeeping.
- **`LogDensityProblems` seam.** The posterior is a
  `LogDensityProblems`-conformant object; downstream samplers
  (Turing, AdvancedHMC, custom) plug in without touching this
  package.
- **R-INLA-equivalent BYM2 scaling and PC priors** by default. Where
  we differ, the docstring says so.
- **No `f(...)` / `inla.formula` macro magic.** A formula sugar layer
  (`@lgm`) exists in `LGMFormula.jl` for users coming from R; the
  underlying constructor API is the source of truth.

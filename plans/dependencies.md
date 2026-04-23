# Dependencies

Every dependency is a commitment. This document records what we depend on
and why, and what we deliberately do not.

Three hosting modes:

- **Core `[deps]`** — always loaded; every user pays the cost.
- **Weakdep + extension** — loaded only when the user brings the trigger
  package. The extension cannot export new symbols, only extend methods
  on types owned by the core package (Julia 1.9+ rule).
- **Sub-package** — separately-installed package in `packages/` that
  depends on the core. Can freely export new symbols; has its own release
  cadence, test matrix, and Project.toml.

When a heavy integration (Turing, Rasters, Pardiso) is needed, a
sub-package is preferred over a weakdep because it gates the cost
behind an explicit `Pkg.add`, makes failure modes visible, and does not
inflate the core's test matrix. Extensions are kept for *lightweight*
convenience integrations (Makie recipes, GeoInterface acceptance,
ChainRules AD rules for Zygote/Enzyme users).

## Core dependencies by package

### `GMRFs.jl`

| Package | Purpose | Notes |
|---|---|---|
| SparseArrays | Sparse matrix representation of Q | stdlib |
| LinearAlgebra | Dense linear algebra, BLAS calls | stdlib |
| Random | RNG interface | stdlib |
| Graphs | Graph structure, connected components, adjacency | JuliaGraphs |
| LinearSolve | Swappable sparse factorization backend (CHOLMOD / KLU / Pardiso) | SciML |
| Distributions | Priors as first-class distribution objects | standard |
| Statistics | Statistical summaries | stdlib |
| ChainRulesCore | AD rules (promoted from weakdep — near-zero mass, high convenience for Zygote/Enzyme users) | standard |
| SelectedInversion | `diag(Q⁻¹)` on the sparse Cholesky pattern; default `marginal_variances` path | see ADR-012 |

Weakdeps (extensions):

| Package | Extension | Purpose |
|---|---|---|
| MakieCore | `GMRFsMakieExt` | Plotting recipes (MakieCore only — the recipes half, lightweight) |

### `LatentGaussianModels.jl`

| Package | Purpose | Notes |
|---|---|---|
| GMRFs | This ecosystem | core dep |
| LinearAlgebra, Random, SparseArrays, Statistics | stdlib | |
| Distributions | Likelihoods and priors | standard |
| LogDensityProblems | Standard interface for posterior log-density — **the seam** for downstream samplers | standard |
| NonlinearSolve | Inner Newton for mode of x \| θ, y | SciML |
| Optimization, OptimizationOptimJL | Outer θ-mode finding | SciML |
| Roots | PC prior construction, 1D root-finding | standard |
| FastGaussQuadrature | Gauss-Hermite nodes | standard |
| QuadGK | 1D marginal integration | standard |
| ADTypes | AD backend selection | SciML |

Weakdeps (extensions):

| Package | Extension | Purpose |
|---|---|---|
| MakieCore | `LGMMakieExt` | Posterior plot recipes |
| HCubature | `LGMHCubatureExt` | Adaptive cubature for posterior expectations |
| Integrals | `LGMIntegralsExt` | SciML-style quadrature backend selection |

### `INLASPDE.jl`

| Package | Purpose | Notes |
|---|---|---|
| LatentGaussianModels | This ecosystem | core dep |
| Meshes | 2D/3D mesh representation, topology | JuliaEarth |
| DelaunayTriangulation | Constrained Delaunay mesh generation | standard |
| SciMLOperators | Lazy projector operators | SciML |
| CoordRefSystems | CRS-aware distances, great-circle Matérn on sphere | JuliaEarth |

Weakdeps (extensions):

| Package | Extension | Purpose |
|---|---|---|
| GeoInterface | `INLASPDEGeoInterfaceExt` | Accept any GeoInterface geometry |
| MakieCore | `INLASPDEMakieExt` | Mesh + field plotting |

## Sub-packages (optional companion packages)

Each has its own `CLAUDE.md`, `plans/plan.md`, and Project.toml under
`packages/`. Users install only what they need.

| Sub-package | Depends on (ecosystem + outside) | Role |
|---|---|---|
| `LGMFormula.jl` | LatentGaussianModels + StatsModels | Tier-2 `@lgm` formula sugar. Defer to Phase 3 tail. See ADR-008. |
| `LGMTuring.jl` | LatentGaussianModels + Turing + AdvancedHMC | HMC/NUTS bridge, INLA-within-MCMC. Defer to Phase 3 tail / Phase 5. See ADR-009. |
| `GMRFsPardiso.jl` | GMRFs + Pardiso | MKL Pardiso / Panua Pardiso factorization backend. License-gated; explicit install makes failure modes visible. |
| `INLASPDERasters.jl` | INLASPDE + Rasters | Covariate extraction + prediction surfaces. Rasters transitively pulls GDAL_jll/Proj_jll (~hundreds of MB) — too heavy for a weakdep. |

### Why sub-packages, not weakdeps, for these four?

- **Export new symbols.** `@lgm`, `NUTS` convenience constructors, a
  `Pardiso.PardisoFactorization()` re-export — none of these can be
  exported from a weakdep extension.
- **Heavy transitive deps.** Turing (20–40 s TTFX), Rasters
  (GDAL_jll, Proj_jll), Pardiso (license-gated MKL or Panua) all inflate
  even an unused weakdep's install footprint. A sub-package gates the
  cost behind an explicit `Pkg.add`.
- **Independent release cadence.** Turing and Rasters evolve on their own
  schedules. Pinning core LGM's CI to their master branches would be
  churn.
- **Clear failure modes.** If `LGMTuring.jl` is broken on a given Julia
  version, core INLA still works. Weakdep extensions can fail at load
  time in ways that look like core-package bugs.

## Forbidden or deferred

- **DynamicPPL / Turing as core deps.** Turing is a downstream consumer
  via `LogDensityProblems`. `LGMTuring.jl` bridges it; core LGM has no
  Turing dep of any kind.
- **MLJ.** Out of scope. A separate `MLJLatentGaussianModels.jl` could
  provide a bridge later.
- **DataFrames anywhere in the ecosystem.** Tables.jl-compatible inputs
  are fine; DataFrames adds no affordance over Tables. An earlier draft
  had `LGMDataFramesExt`; dropped.
- **Reimplementing SuiteSparse / CHOLMOD.** LinearSolve.jl already
  abstracts this.
- **fmesher wrapping in core.** Native Julia mesh generation via
  DelaunayTriangulation.jl is the Phase-4 plan. A BinaryBuilder-based
  wrapper of `fmesher` is a possible sub-package if DelaunayTriangulation
  proves inadequate. See ADR-007.
- **Makie (full) in core deps.** MakieCore weakdep only. Users who plot
  bring Makie themselves.
- **StatsModels as a core dep of LGM.** It lives in `LGMFormula.jl`,
  nowhere else.

## Version policy

- Target Julia 1.10 LTS and current stable minor. Nightly in CI.
- Use `[compat]` bounds on **every** `[deps]` entry. Pin major versions;
  bump minor when we test against them. Inter-package deps within this
  monorepo (`GMRFs = "0.1"`, etc.) also carry compat bounds — a breaking
  change in one should not ripple silently into the others.
- LinearSolve, NonlinearSolve, Optimization, Meshes, Turing, Rasters all
  evolve faster than our cadence. CI includes a nightly job against the
  master branches of the core-dep ones to catch breakages early; for
  sub-package deps (Turing, Rasters), each sub-package runs its own
  nightly matrix.

## Adding a dependency

Before adding a new entry to `[deps]` anywhere:

1. Check this file — is it explicitly forbidden or deferred?
2. Is the right host a weakdep extension, or a new sub-package?
3. If a sub-package, scaffold `packages/<Name>.jl/` with its own
   CLAUDE.md + plans/plan.md + Project.toml.
4. Write an ADR in `plans/decisions.md` with the rationale and link the
   PR.

## Why not Integrals.jl in core LGM

Earlier drafts had it as a core dep. It is not, because the θ-integration
in INLA exploits the Gaussian shape of `π(θ ∣ y)` around its mode — CCD,
grid, and Gauss-Hermite in the eigenbasis of −H(θ*) are *not* adaptive
cubature, and a generic black-box integrator would waste Cholesky
evaluations. CCD and grid are hand-written on top of FastGaussQuadrature
plus raw arrays. Integrals is useful for user-facing posterior
expectations, which is a weakdep.

## Why Graphs.jl even though Q has the adjacency

For connected-component detection (disconnected ICAR sum-to-zero
correctness), Kronecker graph products, user-facing adjacency
construction, and interop with the rest of JuliaGraphs. At extreme
scales (SPDE meshes with 10⁶+ nodes) we may provide a lazy
`AbstractGraph` wrapper on top of `SparseMatrixCSC` to avoid
materializing `SimpleGraph`; the interface is open enough.

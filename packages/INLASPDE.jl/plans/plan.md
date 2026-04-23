# INLASPDE.jl — package plan

## Goal

SPDE–Matérn machinery for `LatentGaussianModels.jl`. Equivalent to
R-INLA's `inla.spde2.matern` + `fmesher` stack, but native-Julia and built
on `Meshes.jl` / `DelaunayTriangulation.jl`.

## Module layout

```
src/
├── INLASPDE.jl                 # main module
├── mesh/
│   ├── inla_mesh.jl            # inla_mesh_2d wrapper around DT
│   ├── refinement.jl           # edge-length / angle constraints
│   └── boundary.jl             # convex / nonconvex hull construction
├── assembly/
│   ├── fem.jl                  # C (mass), G₁, G₂ assembly per-element
│   ├── lumping.jl              # diagonal C lumping
│   └── precision.jl            # Q(τ, κ) from C, G for α ∈ {1, 2}
├── components/
│   ├── spde2.jl                # SPDE2 <: AbstractLatentComponent
│   └── spde_fractional.jl      # deferred to v0.3
├── projector.jl                # MeshProjector, barycentric interpolation
├── priors/
│   └── pc_matern.jl            # PC priors on range + σ
└── utils.jl

ext/
├── INLASPDEGeoInterfaceExt.jl
├── INLASPDEMakieExt.jl
└── INLASPDERastersExt.jl

test/
├── runtests.jl
├── regression/
│   ├── test_fem_small_mesh.jl       # hand-computed reference
│   ├── test_mass_lumping.jl
│   ├── test_precision_alpha1.jl
│   ├── test_precision_alpha2.jl
│   ├── test_matern_reproduction.jl  # Q⁻¹ ≈ Matérn on fine mesh
│   ├── test_projector.jl            # barycentric correctness
│   └── test_pc_matern_prior.jl
└── oracle/
    ├── fixtures/                     # R-INLA SPDE fits as JLD2
    └── test_against_inla.jl
```

## Milestones

### M1 — FEM assembly (3 weeks)

- [ ] Per-element mass `C` and stiffness `G₁` assembly.
- [ ] Diagonal lumping of `C` for α = 2.
- [ ] `G₂ = G₁ C̃⁻¹ G₁` construction.
- [ ] Precision `Q(τ, κ)` for α ∈ {1, 2}:
    - α=1: `Q = τ² (κ² C + G₁)`
    - α=2: `Q = τ² (κ⁴ C + 2κ² G₁ + G₂)`
- [ ] Regression tests against hand-computed matrices on small meshes.
- [ ] Matérn covariance reproduction test on fine mesh.

### M2 — Mesh generation (3 weeks)

- [ ] `inla_mesh_2d(points; max_edge, cutoff, extend)` using
      DelaunayTriangulation.jl.
- [ ] Convex / nonconvex hull helpers.
- [ ] Boundary refinement.
- [ ] Optional extension buffer (matches R-INLA's `offset` argument).
- [ ] Compare mesh statistics against `fmesher` on identical input
      boundary.

### M3 — SPDE component (2 weeks)

- [ ] `SPDE2` struct implementing `AbstractLatentComponent`.
- [ ] `graph(spde)` — mesh topology as `GMRFGraph`.
- [ ] `precision_matrix(spde, θ)` — Q from stored C, G₁, G₂.
- [ ] Internal parameterization `θ = (log τ, log κ)`.
- [ ] `log_hyperprior(spde, θ)` with PC prior.

### M4 — Projector + PC priors (2 weeks)

- [ ] `MeshProjector(mesh, locations)` — sparse barycentric matrix.
- [ ] `SciMLOperators.AbstractSciMLOperator` wrapper for lazy application.
- [ ] `PCMatern(range_prior, sigma_prior)` per Fuglstad et al. 2019.
- [ ] Integration with `LatentGaussianModel.projector` field.

### M5 — Meuse vignette (1 week)

- [ ] End-to-end: Meuse zinc data → mesh → SPDE → inla fit → prediction.
- [ ] Oracle test: posterior summaries within tolerance of R-INLA.
- [ ] Docs: first SPDE vignette in Documenter site.

### M6 — Extensions (2 weeks)

- [ ] `INLASPDERastersExt`: covariate extraction from rasters to mesh
      vertices, prediction returned as `Raster`.
- [ ] `INLASPDEGeoInterfaceExt`: accept any GeoInterface geometry for
      boundary / observation points.
- [ ] `INLASPDEMakieExt`: mesh plot, posterior field with uncertainty.

## Deferred to v0.3+

- **Fractional α** via Bolin-Kirchner 2020 rational approximations.
- **Non-stationary SPDE** with spatially-varying κ, τ.
- **Non-separable space-time SPDE** (Lindgren et al. 2024).
- **SPDE on the sphere** using `CoordRefSystems` great-circle distances.
- **3D SPDE** (brain connectivity, subsurface).

## Risk items

- **Mesh quality.** If `DelaunayTriangulation.jl` doesn't produce meshes
  comparable to `fmesher`, fallback is wrapping `fmesher` via
  BinaryBuilder. ADR needed before that decision.
- **Projector performance at scale.** For 10⁶-node meshes × 10⁵
  observations, sparse barycentric projector must be assembled quickly.
  Benchmark on realistic sizes in M4.

## Validation datasets

- **Meuse zinc** (Gaussian, standard geostatistics).
- **North Carolina SIDS** (Poisson, SPDE on lat-lon).
- **Synthetic 2D Matérn** with known `(range, σ)` — recover them within
  CI.

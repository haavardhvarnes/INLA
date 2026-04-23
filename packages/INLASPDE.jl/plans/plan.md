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

### M1 — FEM assembly (3 weeks) — DONE

- [x] Per-element mass `C` and stiffness `G₁` assembly.
- [x] Diagonal lumping of `C` for α = 2.
- [x] `G₂ = G₁ C̃⁻¹ G₁` construction.
- [x] Precision `Q(τ, κ)` for α ∈ {1, 2}:
    - α=1: `Q = τ² (κ² C + G₁)`
    - α=2: `Q = τ² (κ⁴ C̃ + 2κ² G₁ + G₂)` (R-INLA-style, lumped C̃
      in the κ⁴ term for sparsity — Lindgren–Rue–Lindström 2011, App. C).
- [x] Regression tests against hand-computed matrices on small meshes.
- [x] Matérn covariance reproduction test on fine mesh.

### M2 — SPDE component + PC-Matérn prior (2 weeks) — DONE

Implementation reordered ahead of mesh generation so the component contract
can be validated on hand-made meshes before `inla_mesh_2d` lands.

- [x] `SPDE2` struct implementing `AbstractLatentComponent`.
- [x] Mesh topology exposed as `GMRFs.GMRFGraph` (field `spde.graph`,
      derived from off-diagonal pattern of `C`).
- [x] `precision_matrix(spde, θ)` — Q from stored FEM matrices.
- [x] Internal parameterization `θ = (log τ, log κ)` per ADR-013.
- [x] `(log τ, log κ) ↔ (ρ, σ)` user-scale mapping (α = 2, d = 2).
- [x] `PCMatern` prior struct + `pc_matern_log_density` evaluator with
      change-of-variables Jacobian onto `(log ρ, log σ)`.
- [x] `log_hyperprior(spde, θ)` compositing the PC prior through the
      internal-to-user-scale map (Jacobian determinant 1, ADR-013).
- [x] `GMRFs.constraints(::SPDE2) = NoConstraint()`.
- [x] Regression tests: rate derivation, argument validation, tail
      probabilities, closed-form density, round-trip user ↔ internal,
      precision agreement with FEM, SPD on a fine mesh.

### M3 — Mesh generation (3 weeks) — v0.1 core DONE, fmesher parity deferred

v0.1 core (this commit):

- [x] `inla_mesh_2d(loc; max_edge, offset, cutoff, min_angle)` and
      `inla_mesh_2d(; boundary, ...)` using DelaunayTriangulation.jl.
- [x] Convex-hull helper `convex_hull_polygon`.
- [x] Convex-polygon outward expansion `expand_polygon` (R-INLA `offset`).
- [x] Point-cloud `cutoff_dedup` for near-duplicate collapse.
- [x] `INLAMesh` wrapper (points / triangles matrices + DT object).
- [x] `FEMMatrices(mesh)` and `SPDE2(mesh; …)` convenience constructors.
- [x] Regression tests: hull orientation, polygon offset correctness on
      square / triangle, cutoff behaviour, argument validation,
      post-refinement min-angle guarantee, mesh domain extension under
      `offset`, and end-to-end SPDE2 assembly on a refined mesh.

Deferred (for a later M3.x):

- [ ] Nonconvex hull helpers (alpha shape / α-concave).
- [ ] Pre-subdivision of boundary edges to enforce a strict `max_edge`
      bound rather than the current area-based soft bound.
- [ ] Two-region mesh with separate inner/outer `max_edge` (R-INLA's
      `max.edge = c(inner, outer)`).
- [ ] Compare mesh statistics against `fmesher` on identical input
      boundary (requires fixture infrastructure).

Quantitative exit criterion (the ADR-007 quality gate): on each of three
reference boundaries — unit square, L-shape, NC-coastline subset — the
Julia mesh must satisfy, versus the `fmesher` baseline fixture:
- `|n_vertices_J - n_vertices_R| / n_vertices_R ≤ 0.05`
- minimum triangle angle ≥ `max(20°, 0.95 · fmesher_min_angle)`
- maximum edge length ≤ `1.05 · fmesher_max_edge`

Failure on any boundary triggers the `INLASPDEFmesher.jl` fallback per
ADR-007.

### M4 — Projector (1 week) — DONE

- [x] `MeshProjector(mesh, locations)` — `n_obs × n_vertices` sparse
      barycentric matrix. Enclosing-triangle lookup delegates to
      `DelaunayTriangulation.find_triangle`; barycentric coordinates
      computed in double precision and stored as the row weights.
- [x] `outside = :error | :zero` policy for locations outside the mesh
      domain, with optional `atol` for near-boundary roundoff.
- [x] `SciMLOperators.MatrixOperator` wrapper via `scimloperator(P)`.
- [x] Interop with `LatentGaussianModel(likelihood, component, A)` —
      `P.A` / `sparse(P)` plug straight into the model's projector
      slot.
- [x] Regression tests: row-sum = 1 partition of unity, ≤ 3 nonzeros
      per row, exact reproduction of linear fields, identity-like rows
      at mesh-vertex queries, outside-policy behaviour, `scimloperator`
      agreement, and `LatentGaussianModel` end-to-end wiring.

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

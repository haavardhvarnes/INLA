# Guidance for Claude Code in INLASPDE.jl

Extends [`/CLAUDE.md`](../../CLAUDE.md). Scoped to SPDE machinery.

## Scope

This package owns:
- FEM assembly of SPDE matrices `C` (mass), `G₁`, `G₂` (stiffness) on a
  triangulated mesh.
- SPDE–Matérn link: precision `Q(τ, κ)` from FEM matrices for α ∈ {1, 2}.
- Fractional-α SPDE via rational approximations (Bolin-Kirchner 2020) —
  deferred to v0.3.
- `SPDE2` (and later `SPDEFractional`) as concrete
  `AbstractLatentComponent`s.
- Mesh generation wrappers around `DelaunayTriangulation.jl` producing
  INLA-compatible meshes (equivalent of `inla.mesh.2d`).
- `MeshProjector` — the A-matrix mapping mesh vertices to observation
  points.
- PC priors on range and σ (Fuglstad-Simpson-Lindgren-Rue 2019).

Out of scope:
- Observation likelihoods → `LatentGaussianModels.jl`.
- Sparse Q manipulation → `GMRFs.jl`.
- Raster prediction surfaces → `INLASPDERastersExt` weakdep.

## Dependencies

Core:
- `LatentGaussianModels` (hence `GMRFs` transitively) — this package
  extends the component contract.
- `Meshes` (JuliaEarth) — mesh representation.
- `DelaunayTriangulation` — constrained Delaunay mesh generation.
- `SciMLOperators` — lazy projector operators.
- `CoordRefSystems` — CRS-aware distances.

Weakdeps:
- `GeoInterface` — accept any GeoInterface-compatible geometry as input.
- `MakieCore` — mesh + posterior field visualization.

`Rasters` is **not** a weakdep here — it lives in the separate
`packages/INLASPDERasters.jl/` sub-package. The transitive closure of
Rasters (GDAL_jll, Proj_jll, NetCDF_jll) is too heavy for a weakdep
that most users will never trigger.

## Key correctness tests

The FEM assembly is the single most error-prone piece of this package. Get
these right or nothing else matters:

1. **Mass matrix `C` diagonal lumping.** R-INLA lumps C to its diagonal
   for α = 2. We do the same. Test: lumped vs full C on small meshes.
2. **Stiffness matrix `G₁`.** Element-by-element assembly against a hand-
   computed reference on a 3-triangle mesh.
3. **`G₂ = G₁ C⁻¹ G₁`.** Direct vs sparse-formula construction.
4. **Matérn covariance reproduction.** On a fine mesh, `Q⁻¹` should
   approximate the Matérn covariance `Σ(r; κ, τ)` within finite-element
   error. Test on a few lag distances.

## Mesh generation

`inla_mesh_2d` should produce meshes comparable to R-INLA's `inla.mesh.2d`:
- Constrained Delaunay triangulation with domain boundary + inner points.
- Minimum angle constraint.
- Refinement near boundary.
- Optional extension buffer outside the observation region.

Match `fmesher`'s output quality on the same input points. Cross-check
against `inlabru-org/fmesher` meshes where possible (same boundary, compare
number of vertices, minimum angle, maximum edge length).

## Performance

For realistic SPDE meshes (10⁴–10⁶ vertices), performance matters:
- Sparse G-matrix assembly should be O(triangles), not O(vertices²).
- Prefer building `SparseMatrixCSC` via `sparse(I, J, V, n, n)` over
  element-by-element `setindex!`.
- Use `CartesianIndices` for mesh loops where possible.

## Style

Same as ecosystem-wide. Type parameters on mesh-holding structs so the
mesh type doesn't erase to `Any`.

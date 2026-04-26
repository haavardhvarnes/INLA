# INLASPDE.jl

SPDE–Matérn finite-element assembly. The point-referenced /
geostatistics machinery for the ecosystem: triangulate a domain,
assemble the FEM matrices, wire the resulting precision into a
`LatentGaussianModel` as an `SPDE2` component.

## What's here

- **FEM assembly**: `fem_matrices(mesh)` returns the mass matrix `C`
  (lumped to its diagonal for α = 2 per R-INLA convention) and
  stiffness matrices `G₁`, `G₂`, with `Q(τ, κ)` derived from those.
- **`SPDE2`** — concrete `AbstractLatentComponent` that wraps the
  FEM precision and the PC-Matérn hyperprior. α = 2 is the v0.1
  default; fractional α via Bolin–Kirchner rational approximation is
  deferred to v0.3.
- **`PCMatern`** — the Penalised Complexity prior on (range, σ) of
  Fuglstad–Simpson–Lindgren–Rue (2019). Two scalar tail probabilities
  `(ρ₀, p_ρ, σ₀, p_σ)` parameterise it.
- **Mesh generation**: `inla_mesh_2d` wraps DelaunayTriangulation.jl
  to produce R-INLA-style triangulations (constrained Delaunay,
  minimum-angle, optional outer extension buffer). The companion
  `fmesher_parity` test suite checks vertex counts and minimum-angle
  bounds against R-INLA's `fmesher` outputs.
- **`MeshProjector`** — sparse barycentric A-matrix mapping mesh
  vertices to observation points; the SPDE counterpart of the
  identity-block in areal projectors.

## Required correctness tests

The FEM assembly is the single most error-prone piece. The package
ships closed-form regression tests for:

1. Mass-matrix lumping (lumped vs full C on small meshes).
2. Stiffness matrix `G₁` against hand-computed references on a
   3-triangle mesh.
3. `G₂ = G₁ C⁻¹ G₁` direct vs sparse-formula construction.
4. Matérn covariance reproduction: on a fine mesh, `Q⁻¹` matches the
   analytic Matérn covariance within FEM error.

See the [Meuse SPDE vignette](../vignettes/meuse-spde.md) for a real
end-to-end fit.

## API

```@autodocs
Modules = [INLASPDE]
```

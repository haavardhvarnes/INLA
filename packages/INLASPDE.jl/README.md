# INLASPDE.jl

SPDE–Matérn Gaussian random fields on triangulated meshes, for use as
`AbstractLatentComponent` in the [`LatentGaussianModels.jl`](../LatentGaussianModels.jl/)
framework.

Implements the stochastic partial differential equation approach of
Lindgren, Rue & Lindström (2011), which links Matérn Gaussian fields to
sparse Gaussian Markov random fields via finite-element projections on
a mesh. This is the native-Julia equivalent of R-INLA's SPDE +
fmesher functionality.

## Status

`v0.1.0-rc1`. Shipped:

- `SPDE2` — α = 2 SPDE-Matérn `AbstractLatentComponent`.
- `PCMatern` — joint PC prior on (range, σ).
- `inla_mesh_2d` — DT.jl-native constrained-Delaunay mesh generator
  matching R-INLA's `inla.mesh.2d` on convex domains.
- `MeshProjector` — A-matrix mapping mesh vertices to observation
  points, exposed as a `SciMLOperators.AbstractSciMLOperator`.

Validated against R-INLA on the Meuse zinc dataset (see
[`test/oracle/test_meuse_spde.jl`](test/oracle/test_meuse_spde.jl)).
Higher-α and fractional SPDE (Bolin-Kirchner 2020) are deferred to
v0.3.

## Quick example — Meuse zinc

The actual API exercised by the Meuse oracle test:

```julia
using INLASPDE, LatentGaussianModels, SparseArrays

# `points :: Matrix{Float64}` (n_v × 2) — mesh vertex coordinates
# `tv     :: Matrix{Int}`     (n_t × 3) — triangle index list (1-based)
# `A_field :: SparseMatrixCSC` (n_obs × n_v) — projector to obs locations

spde = SPDE2(points, tv; α = 2,
    pc = PCMatern(
        range_U = 0.5, range_α = 0.5,   # P(range < 0.5) = 0.5
        sigma_U = 1.0, sigma_α = 0.5,   # P(σ > 1.0)     = 0.5
    ))

intercept = Intercept(prec = 1.0e-3)
beta_dist = FixedEffects(1; prec = 1.0e-3)

# Latent layout: x = [α, β_dist, u(field)]
A = hcat(ones(n_obs, 1),
        reshape(dist_cov, n_obs, 1),
        A_field)

like  = GaussianLikelihood(hyperprior = PCPrecision(1.0, 0.01))
model = LatentGaussianModel(like, (intercept, beta_dist, spde), A)

res = inla(model, y)
```

For mesh generation from a polygon, use:

```julia
mesh = inla_mesh_2d(boundary; max_edge = (0.05, 0.2), cutoff = 0.02)
points, tv = mesh.loc, mesh.tv
```

## Installation

Not yet on the General registry. Registered in a personal Julia
registry — add it once, then `Pkg.add` as usual:

```julia
using Pkg
Pkg.Registry.add(RegistrySpec(url = "https://github.com/haavardhvarnes/JuliaRegistry"))
Pkg.Registry.add("General")  # idempotent if already added
Pkg.add("INLASPDE")
```

## See also

- [`GMRFs.jl`](../GMRFs.jl/) — sparse precision substrate.
- [`LatentGaussianModels.jl`](../LatentGaussianModels.jl/) — LGM
  framework that consumes SPDE components.
- [`INLASPDERasters.jl`](../INLASPDERasters.jl/) — raster ↔ SPDE glue
  (planning).
- [`Meshes.jl`](https://github.com/JuliaGeometry/Meshes.jl) — mesh
  representation.
- [`DelaunayTriangulation.jl`](https://github.com/JuliaGeometry/DelaunayTriangulation.jl)
  — the constrained-Delaunay engine under the hood.

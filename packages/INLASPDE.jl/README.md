# INLASPDE.jl

SPDE–Matérn Gaussian random fields on triangulated meshes, for use as
`AbstractLatentComponent` in the [`LatentGaussianModels.jl`](../LatentGaussianModels.jl/)
framework.

Implements the stochastic partial differential equation approach of
Lindgren, Rue & Lindström (2011), which links Matérn Gaussian fields to
sparse Gaussian Markov random fields via finite-element projections on a
mesh. This is the native-Julia equivalent of R-INLA's SPDE + fmesher
functionality.

## Status

Planning. See [`plans/plan.md`](plans/plan.md).

## Quick example (target API)

```julia
using INLASPDE, LatentGaussianModels, Meshes, DelaunayTriangulation

# Build a mesh over the observation domain
points = [(x, y) for (x, y) in zip(coords_x, coords_y)]
mesh = inla_mesh_2d(points; max_edge = (0.05, 0.2), cutoff = 0.02)

# SPDE–Matérn component with PC prior on range and sigma
spde = SPDE2(
    mesh = mesh,
    α    = 2,
    prior = PCMatern(range_prior = (0.3, 0.5),     # P(range < 0.3) = 0.5
                     sigma_prior = (1.0, 0.01)),   # P(sigma > 1) = 0.01
)

# Use as a component in an LGM
model = LatentGaussianModel(
    likelihood = Gaussian(),
    components = (Intercept(), spde),
    projector  = MeshProjector(mesh, observation_points),
)

fit = inla(model, y)

# Predict on a grid
pred = predict(fit, prediction_grid)
```

## See also

- [`GMRFs.jl`](../GMRFs.jl/) — the sparse precision substrate.
- [`LatentGaussianModels.jl`](../LatentGaussianModels.jl/) — the LGM framework
  that consumes SPDE components.
- [`Meshes.jl`](https://github.com/JuliaGeometry/Meshes.jl) — mesh
  representation.
- [`DelaunayTriangulation.jl`](https://github.com/JuliaGeometry/DelaunayTriangulation.jl) — mesh generation.

# INLASPDERasters.jl

Raster ↔ SPDE glue: extract covariate values from `Rasters.Raster`
sources onto SPDE mesh vertices, and return posterior-field predictions
as `Rasters.Raster` objects.

Companion package to [`INLASPDE.jl`](../INLASPDE.jl/).

## Status

Planning. Activates after INLASPDE.jl M5 (Meuse vignette) lands.

## Quick example

```julia
using INLASPDE, INLASPDERasters, Rasters

# Extract elevation at mesh vertices
elev_raster = Raster("elevation.tif")
elev_at_vertices = extract_at_mesh(elev_raster, mesh)

# Build SPDE model with this covariate, fit, predict back to raster
model = LatentGaussianModel(
    likelihood = Gaussian(),
    components = (Intercept(),
                  FixedEffect(elev_at_vertices),
                  spde),
    projector = MeshProjector(mesh, observation_points),
)
fit = inla(model, y)

# Predict on a grid defined by the raster's extent and resolution
pred = predict_raster(fit, mesh, elev_raster)  # -> Raster
```

## Why a separate package, not a weakdep of INLASPDE

- `Rasters.jl` pulls `GDAL_jll`, `Proj_jll`, `NetCDF_jll`, and other
  binary artifacts totalling hundreds of MB. A weakdep triggered by
  `using Rasters` would inflate the install / precompile cost for
  everyone who wants Rasters in any of their projects, regardless of
  whether they care about SPDE.
- Raster-specific covariate extraction belongs in one dedicated place,
  not sprinkled across an extension.

## See also

- [`INLASPDE.jl`](../INLASPDE.jl/) — the SPDE framework itself.
- [`Rasters.jl`](https://github.com/rafaqz/Rasters.jl).

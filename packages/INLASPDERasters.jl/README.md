# INLASPDERasters.jl

Raster ↔ SPDE glue: extract covariate values from `Rasters.Raster`
sources onto SPDE mesh vertices, and return posterior-field predictions
as `Rasters.Raster` objects.

Companion package to [`INLASPDE.jl`](../INLASPDE.jl/).

## Status

`v0.1.0-rc1`. **Scaffolding only** — package skeleton, dependency
declaration, and test stubs are in place, but the raster-glue API
(`extract_at_mesh`, `predict_raster`) is not yet implemented. Activates
in v0.2.

The Meuse SPDE vignette has landed in the
[main docs site](../../docs/src/vignettes/meuse-spde.md), but the
example there constructs the SPDE A-matrix and covariate vectors
manually rather than going through this package. Once the raster glue
ships, the vignette will be rewritten to use it directly.

## Planned API

The following sketch shows the intended call shapes; none of these
functions are exported yet.

```julia
using INLASPDE, INLASPDERasters, Rasters

# Extract elevation at mesh vertices
elev_raster      = Raster("elevation.tif")
elev_at_vertices = extract_at_mesh(elev_raster, mesh)

# Predict on a grid defined by the raster's extent and resolution
pred_raster = predict_raster(fit, mesh, elev_raster)  # -> Raster
```

CRS handling will be done via `CoordRefSystems.jl`; mismatches between
mesh CRS and raster CRS will be detected at the API boundary.

## Why a separate package, not a weakdep of INLASPDE

- `Rasters.jl` pulls `GDAL_jll`, `Proj_jll`, `NetCDF_jll`, and other
  binary artifacts totalling hundreds of MB. A weakdep triggered by
  `using Rasters` would inflate the install / precompile cost for
  everyone who happens to load Rasters in the same session, regardless
  of whether they care about SPDE.
- Raster-specific covariate extraction belongs in one dedicated place,
  not sprinkled across an extension.

## See also

- [`INLASPDE.jl`](../INLASPDE.jl/) — the SPDE framework itself.
- [`Rasters.jl`](https://github.com/rafaqz/Rasters.jl) — the raster
  abstraction.

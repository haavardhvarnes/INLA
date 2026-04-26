# INLASPDERasters.jl

Raster glue for `INLASPDE.jl`: extract covariates at mesh vertices
from a `Rasters.Raster`, and project an SPDE posterior back onto a
raster grid. Lives in its own sub-package because Rasters transitively
pulls GDAL_jll / Proj_jll (~hundreds of MB) — too heavy for a weakdep
that most users will never trigger.

## What's here

- **`extract_at_mesh(raster, mesh; crs)`** — barycentric / nearest-
  neighbour sampling of a raster at the mesh vertex coordinates,
  with explicit CRS handling. Mismatched CRSs raise at the API
  boundary rather than silently mis-locating points.
- **`predict_raster(fit, mesh, template; quantity = :mean)`** — push a
  fitted SPDE posterior through `INLASPDE.MeshProjector` onto the
  `template`'s grid. Returns a `Raster` with the same extent /
  resolution / CRS as the template, ready for further GIS use.
- **`quantile_rasters(fit, mesh, template; q = (0.025, 0.5, 0.975))`** —
  return a NamedTuple of `Raster`s for posterior quantiles.

## Why a sub-package, not a weakdep

Rasters' transitive closure (GDAL_jll, Proj_jll, NetCDF_jll) is large
and license-fragile across platforms. Gating the cost behind an
explicit `Pkg.add("INLASPDERasters")` makes the load-time and install-
size impact visible to users, and means CI for `INLASPDE.jl` does not
have to include the Rasters matrix.

## API

```@autodocs
Modules = [INLASPDERasters]
```

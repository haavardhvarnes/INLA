# Guidance for Claude Code in INLASPDERasters.jl

Extends [`/CLAUDE.md`](../../CLAUDE.md). Scoped to raster-SPDE glue.

## Scope

This package owns:
- `extract_at_mesh(raster, mesh)` — sample a `Raster` at mesh vertex
  coordinates; handle CRS transformation via CoordRefSystems.
- `predict_raster(fit, mesh, template_raster)` — project a fitted
  SPDE posterior back onto a raster grid matching the template's
  extent/resolution.
- Type piracy is forbidden — do not add methods on `Raster` or
  `INLASPDE` types that could live in one of those packages.

Out of scope:
- Prediction *uncertainty* aggregation (separate question; handled by
  core LGM's marginal machinery).
- GDAL-specific IO — `Rasters.jl` abstracts that.

## Dependencies allowed

Core:
- `INLASPDE` — the host.
- `Rasters` — the raster abstraction.

That's it. No others without an ADR.

## Testing

- `test/regression/` — synthetic raster + synthetic mesh, check that
  `extract_at_mesh` recovers the analytic raster value at each mesh
  vertex to machine precision (no interpolation ambiguity).
- `test/regression/` — CRS mismatch: mesh in UTM, raster in WGS84;
  extraction respects CRS.
- `test/oracle/` — Meuse zinc fit, compare predicted raster to
  R-INLA's `predict.inla` on the same grid.

## Style

- CRS handling is the subtle part. Every CRS-crossing operation must
  assert equality (or explicit conversion) at the API boundary with a
  helpful error if the CRSs do not match.
- Prediction rasters have the same CRS / extent / resolution as the
  template; callers expecting a different grid must provide a
  different template.

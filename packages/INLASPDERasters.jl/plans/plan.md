# INLASPDERasters.jl — package plan

## Goal

Raster ↔ SPDE glue for `INLASPDE.jl`. Extract covariates from
`Rasters.Raster` at mesh vertices; project posterior fields back onto
raster grids. Not a standalone package; useless without `INLASPDE.jl`.

## Module layout

```
src/
├── INLASPDERasters.jl     # main module
├── extract.jl             # extract_at_mesh, with CRS handling
├── predict.jl             # predict_raster: fit → Raster
└── crs.jl                 # CRS matching / conversion helpers

test/
├── runtests.jl
├── regression/
│   ├── test_extract_synthetic.jl
│   ├── test_extract_crs.jl
│   └── test_predict_shape.jl
└── oracle/
    ├── fixtures/           # JLD2 from R-INLA SPDE + predict.inla
    └── test_meuse_predict.jl
```

## Milestones

### M1 — Extraction (1 week, after INLASPDE.jl M5) — DONE

- [x] `extract_at_mesh(raster, mesh; method = :bilinear | :nearest,
      outside = :error | :missing, missingval)` sampling a 2D `Raster`
      at `INLAMesh` vertex coordinates.
- [x] Ascending- and descending-coordinate raster axes both supported
      via `_bracket(xs, x)` linear scan returning `(i, t)` bracketing
      index + interpolation fraction.
- [x] Outside-extent policy: `:error` throws `ArgumentError`;
      `:missing` substitutes a user-supplied sentinel.
- [x] Regression tests: bilinear reproduces affine fields exactly to
      machine precision; exact at cell corners; nearest snaps to the
      closest cell; descending Y handled; outside policy enforced;
      argument validation.
- [ ] CRS-aware assertion: deferred until `INLAMesh` carries CRS
      metadata. Docstring states the caller must pre-project.

### M2 — Prediction to raster (1 week)

- [ ] `predict_raster(fit, mesh, template_raster)` using the SPDE
      projector and the posterior mean / quantiles.
- [ ] Output matches template CRS, extent, resolution.
- [ ] Meuse zinc oracle test vs R-INLA's `predict.inla`.

### M3 — Uncertainty surfaces (1 week)

- [ ] Quantile rasters (`:mean`, `:sd`, `:q025`, `:q975`).
- [ ] Per-pixel credible interval width as a `Raster` for diagnostic
      plotting.

## Risk items

- **Rasters.jl breaking changes.** Rasters has had a churny API.
  Narrow compat, nightly CI against its master.
- **CRS correctness.** Silent CRS mismatches are a classic geo
  foot-gun. Make the API force explicit CRS agreement.

## Out of scope for v0.1

- Non-raster geographic outputs (GeoPackages, Shapefiles). Those go
  through GeoInterface in core INLASPDE.
- Raster ingest optimization for multi-GB inputs. If someone needs it,
  a separate `INLASPDEZarr.jl` or similar.

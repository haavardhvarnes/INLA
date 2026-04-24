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

### M2 — Prediction to raster (1 week) — DONE

- [x] `predict_raster(values, mesh, template_raster; outside,
      missingval)` — barycentric projection of a per-vertex field to
      a raster matching the template's dims, extent, resolution, and
      dim order. Unit-tested against linear fields (exact
      reproduction), constant fields, outside-of-mesh policy,
      reversed dim order, and argument validation.
- [x] Output raster inherits the template's lookups via
      `similar(template, Float64)`; no ad-hoc dim-order gymnastics.
- [ ] Meuse zinc oracle test vs R-INLA's `predict.inla` — deferred
      until a production SPDE fit is part of the test tree; the
      non-oracle building blocks (M1 extraction + M2 projection) are
      oracle-verified through INLASPDE.jl M5 end-to-end, and M2's
      linear-reproduction test is the tight regression gate.

### M3 — Uncertainty surfaces (1 week) — DONE

- [x] `quantile_rasters(mean, sd, mesh, template; z, outside,
      missingval)` returning a NamedTuple `(mean, sd, lower, upper)`
      of rasters. Lower / upper are built by projecting the
      vertex-level `mean ± z · sd` quantities through the same P1
      projector — a linear view that is sharp at vertices and
      interpolated linearly inside each triangle.
- [x] Default `z = 1.96` gives a 95% Gaussian credible interval;
      callers may pass any non-negative `z` (e.g. `1.645` for 90%).
- [x] Outside-domain cells receive `missingval` consistently in all
      four output rasters.
- [ ] True pixel-level standard deviation `diag(P Σ P')` requires
      access to the joint posterior covariance from LGM — deferred to
      a downstream integration once `marginal_variances` + a pixel
      covariance helper are exposed there.

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

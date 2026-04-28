# Changelog

All notable changes to this repository are documented here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/spec/v2.0.0.html).

## [v0.1.0-rc1] — 2026-04-28

First publicly-usable release line of the Julia INLA ecosystem. The four
`src/`-bearing packages —
[`GMRFs.jl`](packages/GMRFs.jl/),
[`LatentGaussianModels.jl`](packages/LatentGaussianModels.jl/),
[`INLASPDE.jl`](packages/INLASPDE.jl/),
[`INLASPDERasters.jl`](packages/INLASPDERasters.jl/) — cover the
canonical R-INLA datasets within the testing-strategy tolerances.

### Added

- **GMRFs.jl** — sparse Gaussian Markov random field core. Concrete
  types: `IIDGMRF`, `RW1GMRF`, `RW2GMRF`, `AR1GMRF`, `SeasonalGMRF`,
  `BesagGMRF`, `Generic0GMRF`. `GMRFGraph` wraps any sparse adjacency
  for `Graphs.jl` interop. Sampling, log-density, log-determinant,
  marginal variances via selected inversion, sparse-Cholesky
  factor caching (`FactorCache`).
- **LatentGaussianModels.jl** — LGM stack on top of GMRFs.
  Components: `Intercept`, `FixedEffects`, `IID`, `RW1`, `RW2`, `AR1`,
  `Seasonal`, `Besag`, `BYM`, `BYM2`, `Leroux`, `Generic0`,
  `Generic1`. Likelihoods: `Gaussian`, `Poisson`, `Binomial`,
  `NegativeBinomial`, `Gamma` (closed-form gradients/Hessians;
  ForwardDiff fallback for user-defined). Inference strategies:
  `Laplace`, `EmpiricalBayes`, `INLA`. θ-integration schemes: `Grid`,
  `GaussHermite`, `CCD` (`int_strategy = :auto` chooses CCD for
  dim θ > 2, Grid otherwise). Diagnostics: DIC, WAIC, CPO, PIT.
  Hyperpriors: `PCPrecision`, `GammaPrecision`, `LogNormalPrecision`,
  `WeakPrior`, `PCBYM2Phi`, `LogitBeta`. `LogDensityProblems` seam for
  external samplers.
- **INLASPDE.jl** — SPDE–Matérn FEM on triangulated meshes. `SPDE2`
  component for α = 2. `PCMatern` joint PC prior on (range, σ).
  `inla_mesh_2d` constrained-Delaunay mesh generator (DT.jl-native;
  fmesher-equivalent on convex domains). `MeshProjector` A-matrix as
  a `SciMLOperators.AbstractSciMLOperator`.
- **INLASPDERasters.jl** — package scaffolding only; raster ↔ SPDE
  glue (`extract_at_mesh`, `predict_raster`) is planning. Activates
  in v0.2.
- **Oracle test suite.** Eleven R-INLA-derived JLD2 fixtures across
  Scotland and Pennsylvania BYM2, classical BYM, synthetic Gamma /
  Negative Binomial / Generic0 / Generic1 / Seasonal / Leroux /
  disconnected Besag, and Meuse SPDE. R-INLA wall-clock timings
  (`fit$cpu.used`) are stored alongside posteriors so reproductions
  are fully self-contained.
- **`bench/oracle_compare.jl`.** Reproducible parity benchmark over
  all eleven oracle problems; emits a markdown table of relative
  errors and side-by-side wall-clock seconds vs the stored R-INLA
  timing. See [`bench/README.md`](bench/README.md).
- **Documenter site** with three vignettes (Scotland BYM2, Tokyo
  rainfall, Meuse SPDE) and per-package API pages under
  [`docs/src/`](docs/src/).
- **LGMTuring.jl** sub-package providing the NUTS bridge for
  INLA-vs-MCMC triangulation.

### Changed

- **Seasonal log-NC and constraint convention**
  ([`4020589`](https://github.com/HaavardHvarnes/INLA/commit/4020589)).
  `SeasonalGMRF` declares a single sum-to-zero constraint matching
  R-INLA's `model = "seasonal"`. Per-component
  `log_normalizing_constant` uses `rd_eff = period` (not `period − 1`)
  because the constraint hits `range(Q)` rather than `null(Q)`,
  consuming one PD direction. Closes the τ\_seas / mlik gap to R-INLA.
- **BYM log-NC**
  ([`7d1cab7`](https://github.com/HaavardHvarnes/INLA/commit/7d1cab7)).
  `BYM` per-component `log_normalizing_constant` matches R-INLA's
  `extra()` for `F_BYM`: `−¼(2n − K) log(2π)` where `K` is the number
  of connected components. Closes the Scotland BYM mlik gap.
- **Generic0 / Generic1 log-NC**
  ([`3e28604`](https://github.com/HaavardHvarnes/INLA/commit/3e28604)).
  Both match R-INLA's shared `F_GENERIC0` `extra()` branch
  (`inla.c:2986-2987`), with the Gaussian normaliser
  `−½(n − rd) log(2π) + ½(n − rd) θ` applied per component.
- **`Intercept()` is improper by default**
  ([`41c986b`](https://github.com/HaavardHvarnes/INLA/commit/41c986b)),
  matching R-INLA's `prec.intercept = 0`. Closes the constant
  ½ log(prec) shift in BYM2 / BYM joint Gaussian normalising
  constants. Pass `Intercept(prec = …)` for the proper-Normal
  variant.
- **Phase-B feature scope trimmed to MVP**
  ([`ebf8b42`](https://github.com/HaavardHvarnes/INLA/commit/ebf8b42))
  ahead of the rc1 cut.

### Fixed

- **Per-component Sørbye-Rue scaling on disconnected graphs**
  ([`c6547a4`](https://github.com/HaavardHvarnes/INLA/commit/c6547a4)).
  `BesagGMRF` and `BYM2` now scale each connected component
  independently per Freni-Sterrantino, Ventrucci & Rue (2018), and
  emit one sum-to-zero constraint per component. Was the most common
  silent failure mode in disease-mapping models on disconnected
  regions.
- **SPDE2 log-normalizing-constant** for Meuse-class meshes
  ([`bd70f40`](https://github.com/HaavardHvarnes/INLA/commit/bd70f40)).

### Known limitations

Honest list of cases where the rc1 line knowingly diverges from R-INLA:

- **Scotland classical-BYM `τ_b` weakly identified.** Posterior mean
  diverges from R-INLA's by ≈ 60 % at n = 56; the `b`-vs-`u` split
  is not data-identified. Marked `@test_broken` in
  [`test/oracle/test_scotland_bym.jl:101`](packages/LatentGaussianModels.jl/test/oracle/test_scotland_bym.jl).
  Mean of `b + u` and the marginal log-likelihood agree to 1 %.
- **`disconnected_besag` τ posterior mean is heavy-tailed** at n = 12
  (R-INLA's mean ≈ 7587 with sd ≈ 102906; median ≈ 48). The oracle
  test asserts only that the Julia mlik is finite. Use the median or
  smaller fixed-θ grids for tight comparisons here.
- **BYM2 / Leroux φ and ρ are weakly identified by design** at the
  sample sizes in the oracle suite. Reported residual errors of
  ~10–20 % on these are expected, not regressions.
- **Performance regressions vs R-INLA on two cases.**
  `pennsylvania_bym2` runs in ~17.5 s vs R-INLA's 7.4 s (≈ 2.4×
  slower; suspected θ-grid envelope from a wider Hessian at θ̂),
  and `meuse_spde` runs in ~140 s vs R-INLA's 5.3 s (≈ 27× slower;
  R-INLA uses GMRFLib's tuned sparse Cholesky on the mesh-scale
  precision). Every other oracle problem is 10×–1230× faster than
  R-INLA — see the bench harness output for full numbers.

### Validated against

R-INLA `25.x` (see fixture `inla_version` field), R 4.5.x. Fixtures
are regenerated via the scripts under
[`scripts/generate-fixtures/`](scripts/generate-fixtures/).

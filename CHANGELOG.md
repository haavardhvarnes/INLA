# Changelog

All notable changes to this repository are documented here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/spec/v2.0.0.html).

## [v0.1.4] — 2026-05-02

Phase I-A PR-1a. Patch release on `LatentGaussianModels.jl` and the
`INLA.jl` umbrella; `GMRFs.jl`, `INLASPDE.jl`, and `INLASPDERasters.jl`
are unchanged at v0.1.1. First multivariate-IID building block lands:
the bivariate slot for joint-longitudinal-survival random effects,
paired-areal disease mapping, and bivariate meta-analysis. No public
API changed for existing components.

### Added

- **`PCCor0` PC prior on a correlation `ρ ∈ (-1, 1)`** with reference at
  `ρ = 0` (independence). Mirrors R-INLA's `pc.cor0` — used by `2diid`
  and `iid3d`. User-facing parameters `(U, α)` with `P(|ρ| > U) = α`
  give `λ = -log(α) / √(-log(1 - U²))`. Internal scale is
  `θ = atanh(ρ)`; the Jacobian cancels exactly with the
  Kullback-Leibler `|dd/dρ|` factor, leaving a closed-form log-density
  with a Taylor short-circuit at `|ρ|² < 1.0e-7` to avoid the formal
  `0/0`. Implementation:
  [`src/priors/pc_cor0.jl`](packages/LatentGaussianModels.jl/src/priors/pc_cor0.jl);
  regression suite covers symmetry, branch-boundary continuity,
  `λ`-monotonicity, and integral-to-1 over `θ ∈ ℝ`
  ([`test/regression/test_priors.jl`](packages/LatentGaussianModels.jl/test/regression/test_priors.jl)).
- **`IIDND_Sep{N}` family of multivariate IID random effects** with N=2
  shipped (PR-1a). The latent vector is `n·N` slots laid out as N
  consecutive `n`-blocks; joint precision is `Q = Λ ⊗ I_n` with `Λ` the
  inverse of the marginal covariance. For N=2 the constructor
  parameterises via `(τ_1, τ_2, ρ)` on internal scale `(log τ_1, log
  τ_2, atanh ρ)`, with PC priors on the marginal precisions and a
  `PCCor0` on the correlation by default — matches R-INLA's `2diid`
  default exactly. Implementation:
  [`src/components/iidnd.jl`](packages/LatentGaussianModels.jl/src/components/iidnd.jl).
- **`IID2D(n; …)` ergonomic alias** for `IIDND_Sep{2}` with sensible
  default priors (`PCPrecision()` × 2 + `PCCor0()`); accepts a Gaussian
  prior on Fisher-z if the user wants R-INLA's alternate
  `loggamma + atanh-ρ-Gaussian` form. PR-1b territory (`IID3D` +
  Cholesky/LKJ stick-breaking) and PR-1c (`Wishart`/`InvWishart` joint
  prior path) are scoped but not in this release.
- **Argument-validation tests** for `IIDND` reject `n ≤ 0`, `N = 1`,
  `N ≥ 3` (PR-1b territory), and conflicting `hyperprior_corr` /
  `hyperprior_corrs` kwargs
  ([`test/regression/test_iidnd.jl`](packages/LatentGaussianModels.jl/test/regression/test_iidnd.jl)).

### Changed

- **ADR-022 rename `PCCor1` → `PCCor0`** in
  [`plans/decisions.md`](plans/decisions.md). R-INLA's `pc.cor0`
  reserves the reference-at-`ρ = 0` name for the independence-anchored
  prior used by `2diid` / `iid3d`; `pc.cor1` is the
  reference-at-`ρ = 1` companion used by AR(1)'s lag-1 correlation.
  The ADR update was caught and corrected before any code shipped, so
  no migration impact for users — but the wrong name has now been
  burned into PR-1a's public API by the right one.

## [v0.1.3] — 2026-05-02

Phase Q PR-1. Patch release on `LatentGaussianModels.jl` and the
`INLA.jl` umbrella; `GMRFs.jl`, `INLASPDE.jl`, and `INLASPDERasters.jl`
are unchanged at v0.1.1. Closes both Phase Q v0.1 performance regressions
with a single LBFGS-tuning fix; quality numbers unchanged byte-for-byte.

### Changed

- **Default `g_tol = 1.0e-4`** in the outer θ-mode LBFGS for both
  [`INLA`](packages/LatentGaussianModels.jl/src/inference/inla.jl) and
  [`EmpiricalBayes`](packages/LatentGaussianModels.jl/src/inference/empirical_bayes.jl).
  The `Optimization.jl + AutoFiniteDiff` FD-gradient noise floor sits
  near `√eps ≈ 1.0e-8` — exactly Optim.jl's default `g_tol`, so LBFGS
  exhausted the 1000-iteration limit chasing noise. Raising `g_tol` to
  `1.0e-4` recovers the same θ̂ to ≲ 2e-4 in θ-space (well under any
  oracle test tolerance) at a fraction of the work. Users who want the
  prior tolerance can pass `optim_options = (; g_tol = 1.0e-8)` to
  override.
- **Pennsylvania BYM2 wall-clock**: 18.3 s → 0.15 s (119× faster),
  flipping the v0.1.x regression from `0.47×` of R-INLA to **48.8×
  faster** than R-INLA on `bench/oracle_compare.jl`. Phase Q's
  ≤ 1.2× R-INLA acceptance criterion met with 40× headroom.
- **Meuse SPDE wall-clock**: 142 s → 1.32 s (108× faster) under the
  default SuiteSparse backend, flipping the v0.1.x regression from
  27× *slower* than R-INLA to **4.46× faster**. Phase Q's
  ≤ 2× R-INLA-under-Pardiso acceptance criterion met under SuiteSparse
  alone, without needing the `GMRFsPardiso.jl` backend.
- **Quality unchanged.** `bench/oracle_compare_julia.md` Quality table
  is byte-identical to v0.1.2: every `fixed_max_rel`, `hyperpar_max_rel`,
  `mlik_rel` matches to four significant figures across all 11 oracle
  problems.

### Diagnostic

- [`bench/diagnostics/pa_bym2_hessian.jl`](bench/diagnostics/pa_bym2_hessian.jl)
  — eight-stage diagnostic that pinned the regression to the outer
  LBFGS rather than the integration grid or the FD Hessian. Refuted the
  replan-2026-04-28 hypothesis ("wider Hessian at θ̂ → wider grid
  envelope"); FD Hessian was correct, mode-finding was the bottleneck.

## [v0.1.2] — 2026-05-02

Phase F.5 close. Patch release on `LatentGaussianModels.jl` and the
`INLA.jl` umbrella; `GMRFs.jl`, `INLASPDE.jl`, and `INLASPDERasters.jl`
are unchanged at v0.1.1.

### Added

- **`synthetic_baghfalaki` R-INLA oracle** for the joint longitudinal-
  Gaussian + Weibull-PH survival model
  ([`test/oracle/test_synthetic_baghfalaki.jl`](packages/LatentGaussianModels.jl/test/oracle/test_synthetic_baghfalaki.jl)).
  Promotes `test/regression/test_inla_joint_baghfalaki.jl` to oracle
  parity: Julia and R-INLA fit the same dataset via
  `y_resp = list(y_gauss, inla.surv(...))` with `family = c("gaussian",
  "weibullsurv")` and `f(b_surv_idx, copy = "b_long_idx", fixed = FALSE)`.
  Asserts fixed-effects (10%), hyperparameters (20%), β-Copy (15% mean /
  50% sd, wide on the asymmetric posterior), and per-subject `b̂`
  correlation > 0.99. mlik kept as `isfinite` only — joint inherits the
  polynomial-form Laplace gap from the Weibull arm.
- **Joint longitudinal + survival vignette** at
  [`docs/src/vignettes/joint-longitudinal-survival.md`](docs/src/vignettes/joint-longitudinal-survival.md).
  End-to-end Baghfalaki et al. (2024)-style synthetic recovery via
  `StackedMapping`, `Copy`, and the multi-likelihood
  `LatentGaussianModel` (the marquee Phase G PR3 deliverable that was
  missing from the docs site at v0.1.1).

### Changed

- **Phase F.5 calibration excavation.**
  Survival oracle headers
  ([`weibullsurv`](packages/LatentGaussianModels.jl/test/oracle/test_synthetic_weibull_survival.jl),
  [`lognormalsurv`](packages/LatentGaussianModels.jl/test/oracle/test_synthetic_lognormal_survival.jl),
  [`gammasurv`](packages/LatentGaussianModels.jl/test/oracle/test_synthetic_gamma_survival.jl))
  rewritten to record the polynomial-form-Laplace finding that closed
  the 2-week excavation: R-INLA's `GMRFLib` differs from Julia's
  textbook formula at three points (cubic contribution `+⅙ x0³ dddf` vs
  `−⅙`, `η̂·dddf`-corrected Hessian, `*logdens` evaluated at sample = 0
  rather than at the mode). Closure requires modifying
  `src/inference/laplace.jl`; deferred to v0.3 / Phase Q.
- **Documentation: registry policy.** READMEs and `docs/src/index.md`
  drop "not yet on the General registry / submission planned" framing;
  the personal registry at `haavardhvarnes/JuliaRegistry` is the
  documented install path.

## [v0.1.1] — 2026-05-02

Second release line. Multi-likelihood `LatentGaussianModel`, censoring
infrastructure, five new survival likelihoods, the zero-inflated count
pack, the `Copy` component, and the `AbstractObservationMapping` seam.
Julia 1.12+ requirement.

### Added

- **Multi-likelihood `LatentGaussianModel`** (Phase G PR2,
  [`8d66bc9`](https://github.com/HaavardHvarnes/INLA.jl/commit/8d66bc9)).
  A single `LatentGaussianModel` mounts more than one likelihood block
  over a stacked observation vector via `StackedMapping`. Block-diagonal
  observation mappings compose with per-block likelihoods, so joint
  Gauss-Poisson, joint longitudinal-survival, and similar mixed-family
  models share one latent and one hyperparameter posterior. Covered by
  the new `synthetic_joint_gauss_pois` oracle.
- **`AbstractObservationMapping` seam** (Phase G PR1, ADR-017,
  [`fb46f71`](https://github.com/HaavardHvarnes/INLA.jl/commit/fb46f71)).
  The projector from latent `x` to linear predictor `η` is now a typed
  abstraction with three concrete implementations: `IdentityMapping`,
  `LinearProjector` (sparse `A`-matrix; the existing default),
  `StackedMapping` (multi-likelihood block-row stack), plus a
  `KroneckerMapping` stub reserved for Phase M space-time SPDE.
- **`Copy` component** (Phase G PR3, ADR-021,
  [`f20bbfd`](https://github.com/HaavardHvarnes/INLA.jl/commit/f20bbfd) →
  [`2113623`](https://github.com/HaavardHvarnes/INLA.jl/commit/2113623)).
  Joint-effect sharing à la R-INLA's `f(., copy = "name")`. The
  β-scaling lives on the receiving likelihood (not on the projection
  mapping), preserving the separation between observation maps and
  likelihood logic. Backed by an `add_copy_contributions!` hook on
  `AbstractLikelihood`, a closed-form fixed-β oracle, and a joint
  longitudinal + Weibull survival regression test.
- **Censoring infrastructure** (ADR-018,
  [`b97b84f`](https://github.com/HaavardHvarnes/INLA.jl/commit/b97b84f)).
  Per-observation `Censoring` enum (`NONE`, `LEFT`, `RIGHT`,
  `INTERVAL`); survival likelihoods accept a
  `censoring::Vector{Censoring}` field and dispatch internally for
  log-density and η-derivatives.
- **Five new survival likelihoods** (ADR-018):
  - `ExponentialLikelihood`
    ([`b97b84f`](https://github.com/HaavardHvarnes/INLA.jl/commit/b97b84f)).
  - `WeibullLikelihood` — PH parameterisation, shape `α_w` as a
    hyperparameter
    ([`5071990`](https://github.com/HaavardHvarnes/INLA.jl/commit/5071990));
    `PCAlphaW` PC prior (Sørbye–Rue 2017) alongside the
    `loggamma(1, 0.001)` default
    ([`ec18458`](https://github.com/HaavardHvarnes/INLA.jl/commit/ec18458)).
  - `LognormalSurvLikelihood` — AFT parameterisation, precision `τ` on
    `log T`
    ([`1b6f54c`](https://github.com/HaavardHvarnes/INLA.jl/commit/1b6f54c)).
  - `GammaSurvLikelihood` — mean parameterisation, shape `φ`
    ([`5a7c327`](https://github.com/HaavardHvarnes/INLA.jl/commit/5a7c327)).
  - `WeibullCureLikelihood` — Weibull mixture-cure with logistic cure
    fraction
    ([`1501296`](https://github.com/HaavardHvarnes/INLA.jl/commit/1501296)).
- **Cox proportional-hazards via data augmentation** (ADR-018 PR4,
  [`6876b3a`](https://github.com/HaavardHvarnes/INLA.jl/commit/6876b3a)).
  `inla_coxph(time, event)` produces the Holford / Laird-Olivier
  piecewise-exponential-as-Poisson augmentation; `coxph_design` builds
  the matching design matrix.
- **Zero-inflated likelihood pack** (ADR-019,
  [`925d853`](https://github.com/HaavardHvarnes/INLA.jl/commit/925d853)).
  Three R-INLA parameterisations (types 0, 1, 2) × three base count
  families (Poisson, Binomial, NegativeBinomial) = nine new
  likelihoods. ZIP1 oracle vs R-INLA's
  `family = "zeroinflatedpoisson1"`
  ([`b1ab680`](https://github.com/HaavardHvarnes/INLA.jl/commit/b1ab680)).
- **Opt-in simplified-Laplace mean-shift** (ADR-016,
  [`fbe9b50`](https://github.com/HaavardHvarnes/INLA.jl/commit/fbe9b50)).
  `inla(...; latent_strategy = :simplified_laplace)` applies a per-row
  mean-shift correction at the cost of one extra Newton step per
  integration node. The variance correction remains deferred to v0.3
  (Phase Q in the rolling plan). `pennsylvania_bym2` oracle covers the
  new pathway.
- **Survival vignettes.** CoxPH and Weibull survival under the new
  censoring infrastructure, published in
  [`docs/src/vignettes/coxph-weibull-survival.md`](docs/src/vignettes/coxph-weibull-survival.md)
  ([`fd5ac78`](https://github.com/HaavardHvarnes/INLA.jl/commit/fd5ac78)).
- **Seven new R-INLA oracle fixtures**:
  `synthetic_exponential_survival`, `synthetic_weibull_survival`,
  `synthetic_lognormal_survival`, `synthetic_gamma_survival`,
  `synthetic_coxph`, `synthetic_zip1`, plus
  `synthetic_joint_gauss_pois`.

### Changed

- **Drop Julia 1.10 LTS support** (ADR-020,
  [`4b90410`](https://github.com/HaavardHvarnes/INLA.jl/commit/4b90410)).
  All four `src/`-bearing packages now require Julia 1.12+. Back-compat
  shims for `Returns`, `public` markers, and similar 1.11+ features
  have been removed.
- **Project versions bumped to 0.1.1** across `GMRFs.jl`,
  `LatentGaussianModels.jl`, `INLASPDE.jl`, `INLASPDERasters.jl`, and
  the `INLA.jl` umbrella
  ([`700f218`](https://github.com/HaavardHvarnes/INLA.jl/commit/700f218));
  `[compat]` widened for fresh installs.
- **R-INLA fixture regen against `25.10.19`**
  ([`9f98a64`](https://github.com/HaavardHvarnes/INLA.jl/commit/9f98a64)
  + follow-up CI hardening through
  [`44093ab`](https://github.com/HaavardHvarnes/INLA.jl/commit/44093ab)).
  Tolerance comparator replaces the previous byte-level diff to
  accommodate floating-point drift across R-INLA point releases.

### Known limitations

- **Marginal log-likelihood gap on `weibullsurv`, `lognormalsurv`,
  `gammasurv`, and `coxph` oracles.** Phase F.5 excavation
  (2026-05-02) traced
  this to a polynomial-form Laplace approximation in R-INLA's
  `GMRFLib` that differs from Julia's textbook formula at three
  points: the cubic contribution to the centered polynomial
  (`+⅙ x0³ dddf` vs the strict-Taylor `−⅙`), a modified Hessian
  carrying an `η̂·dddf` correction, and `*logdens` evaluated at
  sample = 0 rather than at the posterior mode. Closure requires
  modifying
  [`src/inference/laplace.jl`](packages/LatentGaussianModels.jl/src/inference/laplace.jl);
  deferred to v0.3 per the Phase Q rolling plan. Fixed-effect and
  hyperparameter posteriors agree tightly with R-INLA on these
  oracles. Oracle tests assert `isfinite(log_marginal)` while the
  gap is being characterised.
- **Coxph augmentation `mlik` shifted by `Σ_events log E_{k_last,i}`**
  — the η-independent exposure of the interval the event lands in.
  Cancels in the posterior of `(γ, β)` so it does not affect
  inference. See the algebraic-equivalence regression test
  ([`test/regression/test_coxph_augmentation.jl`](packages/LatentGaussianModels.jl/test/regression/test_coxph_augmentation.jl)).

### Validated against

R-INLA `25.10.19` (CI fixture regen on
[`9f98a64`](https://github.com/HaavardHvarnes/INLA.jl/commit/9f98a64)),
R 4.5.x. Fixture generation scripts under
[`scripts/generate-fixtures/`](scripts/generate-fixtures/).

## [v0.1.0] — 2026-04-28

First tagged release on the user's personal Julia registry. No content
changes versus `v0.1.0-rc1`; release-prep cleanup only.

### Changed

- **Drop `[sources]` blocks from `INLASPDERasters.jl`** to enable
  registration
  ([`06df56a`](https://github.com/HaavardHvarnes/INLA.jl/commit/06df56a)).
- **Version bump** to `v0.1.0` across `GMRFs.jl`,
  `LatentGaussianModels.jl`, `INLASPDE.jl`, and `INLASPDERasters.jl`
  ([`dad9f17`](https://github.com/HaavardHvarnes/INLA.jl/commit/dad9f17)).

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
  ([`4020589`](https://github.com/HaavardHvarnes/INLA.jl/commit/4020589)).
  `SeasonalGMRF` declares a single sum-to-zero constraint matching
  R-INLA's `model = "seasonal"`. Per-component
  `log_normalizing_constant` uses `rd_eff = period` (not `period − 1`)
  because the constraint hits `range(Q)` rather than `null(Q)`,
  consuming one PD direction. Closes the τ\_seas / mlik gap to R-INLA.
- **BYM log-NC**
  ([`7d1cab7`](https://github.com/HaavardHvarnes/INLA.jl/commit/7d1cab7)).
  `BYM` per-component `log_normalizing_constant` matches R-INLA's
  `extra()` for `F_BYM`: `−¼(2n − K) log(2π)` where `K` is the number
  of connected components. Closes the Scotland BYM mlik gap.
- **Generic0 / Generic1 log-NC**
  ([`3e28604`](https://github.com/HaavardHvarnes/INLA.jl/commit/3e28604)).
  Both match R-INLA's shared `F_GENERIC0` `extra()` branch
  (`inla.c:2986-2987`), with the Gaussian normaliser
  `−½(n − rd) log(2π) + ½(n − rd) θ` applied per component.
- **`Intercept()` is improper by default**
  ([`41c986b`](https://github.com/HaavardHvarnes/INLA.jl/commit/41c986b)),
  matching R-INLA's `prec.intercept = 0`. Closes the constant
  ½ log(prec) shift in BYM2 / BYM joint Gaussian normalising
  constants. Pass `Intercept(prec = …)` for the proper-Normal
  variant.
- **Phase-B feature scope trimmed to MVP**
  ([`ebf8b42`](https://github.com/HaavardHvarnes/INLA.jl/commit/ebf8b42))
  ahead of the rc1 cut.

### Fixed

- **Per-component Sørbye-Rue scaling on disconnected graphs**
  ([`c6547a4`](https://github.com/HaavardHvarnes/INLA.jl/commit/c6547a4)).
  `BesagGMRF` and `BYM2` now scale each connected component
  independently per Freni-Sterrantino, Ventrucci & Rue (2018), and
  emit one sum-to-zero constraint per component. Was the most common
  silent failure mode in disease-mapping models on disconnected
  regions.
- **SPDE2 log-normalizing-constant** for Meuse-class meshes
  ([`bd70f40`](https://github.com/HaavardHvarnes/INLA.jl/commit/bd70f40)).

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

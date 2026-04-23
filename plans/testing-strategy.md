# Testing strategy

Four tiers, each answering a different correctness question. Tests live in
each package's `test/` subdirectory; cross-package integration tests live in
top-level `benchmarks/`.

## Tier 1 — Regression (closed-form / hand-computed)

Fast tests that run on every CI push. No external dependencies (no R).

- Q-matrix construction for known models (RW1, RW2, AR1, IID, Besag) against
  analytic precision matrices built densely.
- Sampling: covariance of many samples matches Q⁻¹ within Monte Carlo error.
- Log-determinant of sparse Cholesky matches dense Cholesky.
- PC-prior density values at known points (from Simpson et al. 2017).
- Linear constraint correction: conditional mean after constraint matches
  the analytical formula.

Tolerances:
- **Deterministic comparisons:** 1e-10.
- **Monte Carlo statistics:** 5σ on scalar summaries (per-moment, per-entry
  of mean/variance). 3σ was in an earlier draft; at a CI flake budget
  below 1 in 10⁴, 3σ is too narrow (0.3% per-test flake rate compounds
  over ~50 MC assertions). For joint covariance comparisons
  (`Cov(samples) ≈ Q⁻¹` across many entries), use **Hotelling T²** or
  the operator-norm Frobenius distance of `S − Q⁻¹` normalized by
  `‖Q⁻¹‖_F`, with a 5σ-equivalent threshold calibrated from the sample
  size `N`.
- **Every MC test uses a seeded `AbstractRNG`.** A flaky MC test is
  almost always an unseeded RNG or an undercount of samples; both are
  fixable. Retrying a flaky test is not an acceptable mitigation.

Location: `packages/<pkg>/test/regression/`.

## Tier 2 — R-INLA oracle fixtures

Pre-computed R-INLA outputs stored as JLD2 fixtures. Tests load the fixture
and compare Julia output.

Fixtures live in `packages/<pkg>/test/oracle/fixtures/`. The R scripts that
produced them live in `scripts/generate-fixtures/`, with:

- A `renv.lock` pinning R and R-INLA versions.
- A `Makefile` to regenerate all fixtures.
- A `README.md` documenting which fixture corresponds to which paper /
  example.

Tolerances:
- Posterior means of latent field: 1%.
- Posterior SDs of latent field: 2%.
- Hyperparameter posterior summaries: 5%.
- Marginal likelihood: 1% on log scale (0.01 on `log_mlik`).

When R-INLA releases a new version and a fixture shifts, the workflow is:
1. Rerun the `generate-fixtures` script.
2. Diff old and new fixtures.
3. If the change is explained by a documented R-INLA release note, update
   the fixture and pin the new version.
4. If unexplained, investigate before updating.

## Tier 3 — Triangulation with independent implementations

For the models we really care about being right (BYM2 disease mapping,
SPDE geostatistics), we compare against not just R-INLA but also Stan and
NIMBLE implementations. Our posterior should sit inside the envelope
defined by all three.

Reference repositories:
- `spatialstatisticsupna/Comparing-R-INLA-and-NIMBLE` — Spanish breast
  cancer mortality, ICAR and BYM spatio-temporal.
- `ConnorDonegan/Stan-IAR` — Scotland/other ICAR/BYM/BYM2 Stan
  implementations with correct disconnected-graph handling.
- `gkonstantinoudis/nimble` — Scottish lip cancer NIMBLE implementation.

Location: `benchmarks/triangulation/`. These are slower, run in nightly CI
only.

## Tier 4 — Textbook end-to-end reproduction

Each chapter of Moraga's *Geospatial Health Data* and
Blangiardo & Cameletti's *Spatial and Spatio-Temporal Bayesian Models with
R-INLA* is an implicit specification. For each of the canonical datasets we
reproduce the book's analysis and compare figures/tables.

Canonical datasets:

| Dataset | Model | Book / source |
|---|---|---|
| Scottish lip cancer | Poisson BYM2 + AFF covariate | Moraga Ch. 6 |
| Germany oral cavity | Poisson BYM | R-INLA demodata |
| Sardinia (disconnected) | BYM2 | Freni-Sterrantino et al. 2018 |
| Pennsylvania lung cancer | Poisson BYM2 + smoking | Moraga Ch. 5 |
| Ohio lung cancer | Spatio-temporal BYM2 | Moraga Ch. 7 |
| Meuse zinc | Gaussian SPDE | Gómez-Rubio Ch. 7 |
| NY leukemia | Poisson geostatistical | Gómez-Rubio |
| Tokyo rainfall | Binomial RW2 | R-INLA demodata |
| Seeds germination | Binomial GLMM (BUGS classic) | R-INLA demodata |
| Epil | Poisson GLMM (BUGS classic) | R-INLA demodata |

Location: `benchmarks/books/`. Failure policy: flagged in CI as a warning
rather than a hard failure, because book figures depend on plotting
libraries that we treat as optional. Numerical agreement is still checked.

## Running tests

Per-package:
```
cd packages/GMRFs.jl
julia --project -e 'using Pkg; Pkg.test()'
```

Full ecosystem (from the root):
```
julia --project=. -e 'include("scripts/test-all.jl")'
```

## Per-package test layout

```
packages/<pkg>/test/
├── runtests.jl                 # top-level, @safetestset per file
├── regression/
│   ├── test_Q_construction.jl
│   ├── test_sampling.jl
│   └── …
├── oracle/
│   ├── fixtures/               # committed JLD2 files
│   │   ├── scotland_bym2.jld2
│   │   └── …
│   └── test_against_fixtures.jl
└── triangulation/              # only in packages with non-INLA cross-checks
    └── …
```

## No test is better than a flaky test

- No randomness without a seeded `AbstractRNG`.
- No dependency on machine BLAS details at tight tolerances.
- No downloading datasets at test time — use `Artifacts.toml` or commit
  small ones directly.

## What's explicitly NOT tested

- Bit-level agreement with R-INLA. Different AMD/METIS orderings produce
  different-but-equivalent numerical paths.
- Timing / performance regressions. A separate benchmark suite under
  `benchmarks/performance/` runs on tagged releases, not in PR CI.

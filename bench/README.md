# Oracle compare — Julia INLA vs R-INLA fixtures

Reproducible parity benchmark for the Julia INLA ecosystem. Iterates
over every R-INLA oracle fixture in
[`packages/LatentGaussianModels.jl/test/oracle/fixtures/`](../packages/LatentGaussianModels.jl/test/oracle/fixtures/)
and
[`packages/INLASPDE.jl/test/oracle/fixtures/`](../packages/INLASPDE.jl/test/oracle/fixtures/),
runs the Julia model end-to-end, and prints a markdown summary plus
writes a structured JSON file with per-quantity comparisons.

## Running

From the repo root:

```bash
julia --project=bench bench/oracle_compare.jl
```

First run compiles the world; expect ~3-5 minutes wall-clock total
(single thread). Subsequent runs in the same Julia session are
substantially faster — most of the time is in the `meuse_spde` SPDE
fit and Pennsylvania BYM2 grid integration.

The harness loads `bench/Project.toml`, which `Pkg.develop`s the four
ecosystem packages from `packages/`. `bench/Manifest.toml` is
gitignored — see the comment block at the top of `bench/Project.toml`
for the one-liner that reproduces the environment from a clean clone.

`bench/oracle_compare_julia.json` and `bench/oracle_compare_run.log`
are also gitignored: they're produced fresh on every run.

## Problems

Eleven oracle problems are checked. Each maps to the matching
`test/oracle/test_*.jl` file — the harness builds the same model, so
fixes to the test path propagate here automatically.

| Problem                       | Source                                                                      | Likelihood       | Hyperpars |
|---                            |---                                                                          |---               |---        |
| `scotland_bym2`               | Scotland lip cancer                                                         | Poisson          | 2         |
| `scotland_bym`                | Scotland lip cancer                                                         | Poisson          | 2         |
| `pennsylvania_bym2`           | Pennsylvania lung cancer                                                    | Poisson          | 2         |
| `synthetic_gamma`             | Synthetic                                                                   | Gamma            | 2         |
| `synthetic_seasonal`          | Synthetic                                                                   | Gaussian         | 2         |
| `synthetic_generic0`          | Synthetic                                                                   | Gaussian         | 2         |
| `synthetic_generic1`          | Synthetic                                                                   | Gaussian         | 3         |
| `synthetic_leroux`            | Synthetic                                                                   | Gaussian         | 3         |
| `synthetic_nbinomial`         | Synthetic                                                                   | NegativeBinomial | 2         |
| `synthetic_disconnected_besag`| Synthetic, 3 connected components                                           | Gaussian         | 2         |
| `meuse_spde`                  | Meuse zinc, INLASPDE.jl                                                     | Gaussian         | 3         |

## Output columns

The harness prints two markdown tables.

### Quality (relative error vs R-INLA fixture)

| Column             | Meaning                                                                       |
|---                 |---                                                                            |
| `n`                | Number of observations.                                                       |
| `fixed_max_rel`    | Max over fixed-effects of `|julia_mean − r_mean| / max(|r_mean|, 1)`.         |
| `hyperpar_max_rel` | Max over hyperparameters of `|julia_mean − r_mean| / max(|r_mean|, 1e-12)`.   |
| `mlik_rel`         | Relative error of the marginal log-likelihood: `|Δmlik| / |r_mlik|`.          |
| `mlik_abs`         | Absolute error of the marginal log-likelihood, in nats.                       |

R-INLA's own integration noise is ≈ 1-5 % on hyperparameter means and
≤ 1 % on `mlik`, so values inside that band agree with R-INLA to within
its own numerical reproducibility. Per-problem assertion thresholds
live in the corresponding `test_*.jl` file.

### Performance (wall-clock seconds, single thread)

| Column           | Meaning                                                                                  |
|---               |---                                                                                       |
| `julia_inla_s`   | `@elapsed inla(model, y; int_strategy = :grid)` — full INLA: Laplace + θ-grid mixture.   |
| `julia_eb_s`     | `@elapsed empirical_bayes(model, y)` — Laplace only at θ̂, no θ-integration.             |
| `r_inla_s`       | R-INLA's `fit$cpu.used[4]` (the "Total" column), stored in the JLD2 fixture.             |
| `speedup×_vs_r`  | `r_inla_s / julia_inla_s`. > 1 means Julia is faster; < 1 means R-INLA is faster.        |

`julia_eb_s` is the cheaper "just-Laplace-at-θ̂" mode that's typical
for production likelihood-evaluation use. The `julia_inla_s` column is
the apples-to-apples comparison against R-INLA's default behaviour.

R-INLA timings are stored once at fixture-generation time on the
machine listed in each fixture's `r_session_info`. Comparing against
local Julia times therefore mixes hardware — useful as an
order-of-magnitude check, not a microbenchmark.

## JSON schema

`bench/oracle_compare_julia.json` is a single object:

```jsonc
{
  "generated_at": "2026-04-28T12:34:56.789",         // UTC ISO-8601
  "problems": [
    {
      "problem": "scotland_bym2",
      "status":  "ok",                               // or "skipped" / "error"
      "n":       56,
      "timings": {
        "inla_warmup":     1.234,                    // first (cold) inla() call
        "inla":            0.412,                    // second timed inla() call
        "empirical_bayes": 0.118,
        "r_inla":          5.301,                    // from cpu.used[4]; NaN if missing
        "speedup_vs_r":   12.86                      // r_inla / inla; NaN if r_inla missing
      },
      "julia": {
        "fixed":    [{"name": "(Intercept)", "mean": …, "sd": …}, …],
        "hyperpar": [{"rowname": "Precision for region", "mean": …, "sd": …}, …],
        "mlik":     -123.45,
        "empirical_bayes_log_marginal": -123.50
      },
      "r": {
        "fixed":    [...],                           // same shape, R-INLA fixture values
        "hyperpar": [...],
        "mlik":     -123.40
      },
      "deltas": {
        "fixed":            [{"name": "(Intercept)", "rel": 0.0021}, …],
        "hyperpar":         [{"rowname": "Precision for region", "rel": 0.0084}, …],
        "mlik_abs":         0.05,
        "mlik_rel":         4.0e-4,
        "fixed_max_rel":    0.0021,
        "hyperpar_max_rel": 0.0084
      }
    }
    // … one entry per oracle problem
  ]
}
```

`status = "skipped"` is emitted when the JLD2 fixture is missing on
disk. `status = "error"` is emitted when the Julia run threw — the
error string is in an `error` field, the harness keeps going.

The JSON is hand-encoded (no `JSON3` dep) — it handles only the
`Real`, `String`, `Vector`, `NamedTuple`, and `Dict{String,Any}` shapes
used in the harness output.

## Where the parity story lives

- **Per-problem tolerances and the documented exceptions** (heavy-tail
  τ_b on Scotland classical-BYM, weakly-identified τ on
  `disconnected_besag`, BYM2 φ / Leroux ρ at small n) — in each
  matching `test/oracle/test_*.jl` file, plus the "Known limitations"
  section of [`../CHANGELOG.md`](../CHANGELOG.md).
- **R-INLA fixture provenance** — the `r_session_info` and
  `inla_version` fields of each JLD2 fixture, regenerated via the
  scripts under [`../scripts/generate-fixtures/`](../scripts/generate-fixtures/).

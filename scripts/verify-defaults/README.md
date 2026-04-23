# verify-defaults

Scripts that read the running R-INLA installation and report the default
values of hyper-priors and control arguments we rely on in
[`plans/defaults-parity.md`](../../plans/defaults-parity.md).

Any default that is not verified against a pinned R-INLA here is a
latent divergence risk. Every Phase-2+ component merge requires its
relevant defaults to be verified.

## Setup

Uses the same `renv.lock` as `scripts/generate-fixtures/` — no new R
environment.

```
cd scripts/verify-defaults
Rscript bym2_phi.R
Rscript pc_prec.R
Rscript spde_matern.R
```

Each script prints a table of `(name, value, source)` that is diffed
against the corresponding default in the Julia package. The diff is
machine-readable (CSV) so CI can gate on drift.

## Current scripts (to be written)

- `bym2_phi.R` — reads the default for `pc` prior on the BYM2 mixing
  parameter φ, i.e. `U` and `α`.
- `pc_prec.R` — reads the default for `pc.prec` on `log τ`.
- `spde_matern.R` — reads the default for Fuglstad et al. 2019 PC
  priors on range and σ.
- `int_strategy.R` — confirms the `int.strategy` auto-selection rule.
- `constr_defaults.R` — confirms per-component `constr` defaults.

## Output format

Each script writes `<name>.csv` with columns
`parameter, rinla_value, rinla_version, julia_const, julia_value, match`
into `scripts/verify-defaults/output/`. CI compares against the last
committed snapshot; any drift fails the build unless the snapshot is
updated in the same PR.

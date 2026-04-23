# Contributing to the Julia INLA Ecosystem

This is a **monorepo** containing seven Julia packages (three core, four
optional sub-packages). Each package has its own `Project.toml`, test
suite, and release cadence, but they share history, issues, and CI.

Design documents live in top-level [`plans/`](plans/) and per-package
`plans/plan.md`. Read these before opening a non-trivial PR — the
architecture is load-bearing.

## Legal / licensing

All seven packages are **MIT-licensed**. R-INLA itself is GPL-2, but this
ecosystem is a **clean-room reimplementation from published papers** (see
[`references/papers.md`](references/papers.md)), not a derivative work.
Do not copy code from R-INLA's source. Method citations in docstrings are
fine and encouraged.

If a PR introduces code from a copyleft source, flag it in the PR
description — we will either rewrite from primary references or reject.

## Branching model

- `main` — always green, always releasable. No direct pushes.
- Feature branches: `feat/<short-name>`, `fix/<short-name>`,
  `docs/<short-name>`, `refactor/<short-name>`, `test/<short-name>`.
- PRs merge to `main` via **squash**. The squash commit message is what
  lives in history, so write it carefully.
- Release branches: none during the pre-1.0 phase. Tags cut directly
  from `main`.

## Commit messages — Conventional Commits

We use [Conventional Commits](https://www.conventionalcommits.org/).
Since we squash-merge, the *PR title* is what counts.

Format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:** `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`,
`ci`, `chore`, `revert`.

**Scopes** are package short names (`gmrfs`, `lgm`, `spde`, `formula`,
`turing`, `pardiso`, `rasters`) or ecosystem areas (`docs`, `ci`,
`plans`, `scripts`). Multi-scope PRs should be split — if they can't
be, list scopes comma-separated: `feat(gmrfs,lgm): …`.

**Breaking changes** use `!` and a `BREAKING CHANGE:` footer:

```
feat(lgm)!: replace `fit` positional strategy with kwarg

BREAKING CHANGE: `fit(model, y, INLA())` is still supported; the
positional third arg is now typed more strictly. See ADR-011.
```

Examples:

- `feat(gmrfs): add Besag GMRF constructor and precision matrix`
- `fix(lgm): correct sign in log-hyperprior Jacobian for log(τ)`
- `perf(spde): reuse symbolic Cholesky factor across CCD points`
- `docs(plans): clarify BYM2 scaling formula in defaults-parity.md`
- `ci: add nightly upstream R-INLA oracle refresh job`

## PR checklist

- [ ] Title is a Conventional Commit line.
- [ ] Each package touched has green `Pkg.test()`.
- [ ] New or changed public API has a docstring with an example.
- [ ] If defaults diverge from R-INLA, documented in
      [`plans/defaults-parity.md`](plans/defaults-parity.md).
- [ ] If a new `[deps]` entry is added, an ADR is recorded in
      [`plans/decisions.md`](plans/decisions.md) and
      [`plans/dependencies.md`](plans/dependencies.md) updated.
- [ ] Formatter run: `julia -e 'using JuliaFormatter; format(".")'`
      (SciML style, configured via `.JuliaFormatter.toml`).

## Testing tiers

See [`plans/testing-strategy.md`](plans/testing-strategy.md). Every
non-trivial component lands with Tier 1 (regression, closed-form) and
Tier 2 (R-INLA oracle fixture) tests. Tier 3 (Stan/NIMBLE triangulation)
and Tier 4 (textbook end-to-end) are nice-to-haves per PR, required
before a minor release that touches the component.

## Releases

Per-package tags: `GMRFs-v0.1.0`, `LatentGaussianModels-v0.1.0`, etc.
Bumping a dependent package requires updating `[compat]` and bumping
the dependent's patch version at minimum.

Until the first `v0.1.0` of each core package ships, the repo stays
private. `main` can still break between tags during pre-release.

## Large binary fixtures

Never commit binaries larger than ~100 KB to the repo. R-INLA reference
fits (JLD2) and Meuse/Scotland raster extracts live behind
[Julia Artifacts](https://pkgdocs.julialang.org/v1/artifacts/). A
sentinel `Artifacts.toml` at the repo root points at a stable URL (TBD
— likely GitHub Releases or an OSF project). Fixture generation scripts
under [`scripts/generate-fixtures/`](scripts/generate-fixtures/) produce
the contents; uploads require repo-maintainer credentials.

## Issues

- Bug reports: include Julia version, package version(s), and a minimal
  reproducer. If the bug is a numerical disagreement with R-INLA,
  include the exact R script.
- Feature requests: link to the method paper. The bar for adding a new
  component is a clear reference and a test dataset.

## Code review norms

- Every PR gets at least one reviewer who is not the author.
- Reviewers check against [`CLAUDE.md`](CLAUDE.md) conventions, ADRs,
  and the package's `plans/plan.md`.
- "Nit" comments are optional to address. "Blocker" comments are not.
- If a review surfaces a design question bigger than the PR, it lands as
  an ADR entry before the PR merges.

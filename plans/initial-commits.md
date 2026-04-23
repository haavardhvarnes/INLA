# Initial commit sequence

The repo starts as a single `git init` on the current directory. To
keep history coherent and biset-able from day one, the initial state
lands as **five scoped commits** rather than one "initial commit"
dump. This matches the Conventional Commits scopes in
[`CONTRIBUTING.md`](../CONTRIBUTING.md).

## Commit 1 — repo scaffolding

```
chore: bootstrap repo (license, gitignore, formatter, CI)
```

Files:

- `LICENSE`
- `.gitignore`
- `.gitattributes`
- `.JuliaFormatter.toml`
- `CONTRIBUTING.md`
- `.github/workflows/test.yml`
- `.github/workflows/nightly-upstream.yml`
- `.github/workflows/docs.yml`
- `.github/workflows/benchmark.yml`

## Commit 2 — ecosystem-wide design

```
docs(plans): ecosystem design documents and ADR log
```

Files:

- `README.md`
- `ROADMAP.md`
- `CLAUDE.md`
- `plans/architecture.md`
- `plans/decisions.md`
- `plans/defaults-parity.md`
- `plans/dependencies.md`
- `plans/macro-policy.md`
- `plans/testing-strategy.md`
- `plans/initial-commits.md` (this file)
- `references/papers.md` and any other references
- `scripts/verify-defaults/README.md`
- `scripts/generate-fixtures/` placeholders
- `scripts/run-benchmarks/` placeholders

## Commit 3 — core package scaffolds

```
chore(gmrfs,lgm,spde): scaffold core package projects
```

Files:

- `packages/GMRFs.jl/{Project.toml,LICENSE,CLAUDE.md,plans/plan.md}`
- `packages/LatentGaussianModels.jl/{Project.toml,LICENSE,CLAUDE.md,plans/plan.md}`
- `packages/INLASPDE.jl/{Project.toml,LICENSE,CLAUDE.md,plans/plan.md}`

No `src/` yet. Source lands in Phase 1+ commits.

## Commit 4 — optional sub-package scaffolds

```
chore(formula,turing,pardiso,rasters): scaffold sub-package projects
```

Files:

- `packages/LGMFormula.jl/*`
- `packages/LGMTuring.jl/*`
- `packages/GMRFsPardiso.jl/*`
- `packages/INLASPDERasters.jl/*`

## Commit 5 — commit-time hooks / linting (optional)

If a `pre-commit` config lands on day zero, it goes here. Otherwise
skip.

## After the fifth commit

- Push to a **private** GitHub repo named `INLA.jl-ecosystem`.
- Protect `main`: require PRs, passing `test` workflow, one approving
  review.
- Register placeholder UUIDs for each package (`julia -e 'using UUIDs;
  println(uuid4())'`) and commit as `chore(pkg): assign UUIDs` once
  the scaffolds pass local `Pkg.test()`.
- Reserve package names on the General registry via a name-squat PR
  once UUIDs are stable — but **do not register for real until v0.1.0**
  of each package.

## Branch protection rules (to set in GitHub after first push)

- `main`: no direct push; require PR with 1 approval; require `test`
  workflow green; require linear history (enforced by squash merges);
  dismiss stale approvals on new pushes.
- Allow force-push only for repo admins and only on non-`main`
  branches.

# v0.1.0 release checklist

This is the **operator runbook** for promoting v0.1.0-rc1 to v0.1.0 and
registering the four `src/`-bearing packages in Julia's General
registry. Phase E4 of [`replan-2026-04.md`](replan-2026-04.md).

The actual `Pkg.register` / `git tag` / `git push --tags` calls are
gated on the user (the repo owner). This document describes what to
check, in what order, and what command to run when each gate clears.

## Release manifest

Per [ADR-015](decisions.md), v0.1.0 ships exactly four packages:

| Package | UUID | Version | Direct dep on |
|---|---|---|---|
| GMRFs | `a8720bf5-58d3-44b3-b5a9-acd31420dc17` | 0.1.0 | (none in-repo) |
| LatentGaussianModels | `50d8cc41-879e-4ac9-9bf2-e3c789b17716` | 0.1.0 | GMRFs |
| INLASPDE | `2835c710-3f40-4945-979f-d21c9e20d425` | 0.1.0 | GMRFs, LatentGaussianModels |
| INLASPDERasters | `28708ac2-a765-4cf9-a54c-708b26c5e9a3` | 0.1.0 | INLASPDE |

`LGMFormula.jl` and `GMRFsPardiso.jl` are deferred to v0.2 per
ADR-015. `LGMTuring.jl` is the optional Turing bridge — register only
if it lands cleanly with Turing's compat range; otherwise defer.

## Pre-flight checks

Run these before the first `Pkg.register` call. None of them require
network access until the actual registration step.

### 1. Project.toml hygiene

For each package:

- [ ] `version = "0.1.0"` (currently `0.1.0-rc1` — bump on tag day).
- [ ] `authors = ["..."]` is **not** `["TBD"]`. Set to the actual
      maintainer identity (name + email) before registering. Updates
      to authors after registration require a new release.
- [ ] Every entry in `[deps]` has a matching entry in `[compat]`.
      Aqua-checked in Phase E1; re-verify by running `Pkg.test()`.
- [ ] Stdlibs (`LinearAlgebra`, `Random`, `SparseArrays`, `Statistics`,
      `Printf`) carry `= "1"` compat entries — required by the
      registrator since Julia 1.9.
- [ ] `julia` compat is `"1.10"` (or higher; INLASPDERasters declares
      `"1.11"` to match Rasters' lower bound).

### 2. INLASPDERasters.jl — `[sources]` removal

`INLASPDERasters/Project.toml` currently has:

```toml
[sources]
INLASPDE = {path = "../INLASPDE.jl"}
GMRFs = {path = "../GMRFs.jl"}
LatentGaussianModels = {path = "../LatentGaussianModels.jl"}
```

This **must be removed** before registration — the General registry
rejects packages with `[sources]` entries. After GMRFs,
LatentGaussianModels, and INLASPDE are registered (in that order),
delete the `[sources]` block from INLASPDERasters' Project.toml and
verify `Pkg.test()` still passes resolving through the registry.

### 3. CI green

- [ ] CI green on the `v0.1.0` tag commit for all four packages.
- [ ] Documenter site (`docs/`) builds on CI without errors. Cross-
      reference warnings (`plans/plan.md` paths, `MeshProjector`
      autodocs cross-link) are non-fatal — already gated by
      `warnonly` in `docs/make.jl`.

### 4. Oracle tests

- [ ] All Tier-2 R-INLA oracle tests pass (`@test`, not just
      `@test_broken`) on the release commit. The current
      `@test_broken` items are listed in
      [`replan-2026-04.md`](replan-2026-04.md) "Open follow-ups";
      they document known divergences acceptable for v0.1 and are
      not release-blocking.

### 5. README badges

- [ ] Each package's `README.md` carries a CI badge, a Julia version
      badge, and (post-registration) a registry version badge. The
      registry badge is added in a follow-up PR after the package is
      live in General.

## Registration order

Strict dependency order; each package must be live in General before
its dependents can register.

```
GMRFs            → register first
LatentGaussianModels → register after GMRFs
INLASPDE          → register after LatentGaussianModels
INLASPDERasters   → register last (drop [sources] first)
```

For each package, the registration sequence is:

1. Bump `version` in `Project.toml` from `0.1.0-rc1` to `0.1.0`.
2. Commit (`chore(<pkg>): release v0.1.0`).
3. Tag the monorepo commit: `git tag <pkg>-v0.1.0`.
4. Push tag: `git push origin <pkg>-v0.1.0`.
5. Open a PR against
   [JuliaRegistries/General](https://github.com/JuliaRegistries/General)
   via the [Registrator GitHub app](https://github.com/JuliaRegistries/Registrator.jl)
   on the tagged commit. Comment `@JuliaRegistrator register
   subdir=packages/<pkg>` on the corresponding monorepo commit.
6. Wait for the `AutoMerge` checks to pass (~3 days for new packages,
   minutes for follow-ups).
7. Once merged, the next package in dependency order can begin step 1.

## Tag scheme

Tags are package-prefixed since the repo is a monorepo:

```
GMRFs-v0.1.0
LatentGaussianModels-v0.1.0
INLASPDE-v0.1.0
INLASPDERasters-v0.1.0
```

The Registrator app picks up the `subdir=packages/<pkg>` argument from
the registration comment to find the right `Project.toml`.

## Rollback

If a registered package needs to be withdrawn (only valid for the most
recent registration; General does not delete versions otherwise), open
an issue at
[JuliaRegistries/General](https://github.com/JuliaRegistries/General)
referencing the offending PR. Plan the fix as a `0.1.1` patch release
rather than a withdrawal.

## After release

- [ ] Tag `v0.1.0` on the monorepo itself (an umbrella tag pointing at
      the same commit as the four package tags).
- [ ] GitHub release notes summarising the four packages, the docs
      site link, and the v0.2 roadmap (ADR-015 deferrals + Phase D
      `@test_broken` follow-ups).
- [ ] Update `MEMORY.md` (claude memory) and any project README badges
      with the live registry version.

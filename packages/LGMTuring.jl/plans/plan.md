# LGMTuring.jl — package plan

## Goal

Turing / AdvancedHMC bridge for `LatentGaussianModels.jl`. Provides
HMC/NUTS on the same model fit by INLA (for tier-3 triangulation tests)
and the INLA-within-MCMC pattern from Gómez-Rubio's book.

## Module layout

```
src/
├── LGMTuring.jl                # main module, exports sample, compare
├── logdensity.jl               # build LogDensityModel from LatentGaussianModel
├── sample_nuts.jl              # Turing/AdvancedHMC NUTS wrapper
├── inla_within_mcmc.jl         # outer θ MCMC, inner INLA conditional
└── compare.jl                  # posterior-summary diff table

test/
├── runtests.jl
├── regression/                  # analytic-posterior MC checks
├── oracle/                      # pre-baked chains vs recomputed
└── triangulation/               # Stan / NIMBLE / Turing cross-check
```

## Milestones

### M1 — LogDensityProblems consumption — DONE (2026-04-26)

- [x] `INLALogDensity` already conforms to `LogDensityOrder{1}()` upstream
      in `LatentGaussianModels.jl/src/inference/log_density.jl`; this
      package re-exports it (`inla_log_density`) so HMC users only need
      one import.
- [x] Gradient path: central finite differences inside core LGM.
      `LogDensityProblemsAD` + ForwardDiff is *not* used in v0.1 because
      ForwardDiff would have to differentiate through Optim's inner
      Newton solver; FD is robust and cheap relative to the per-call
      Laplace cost.
- [x] Tests: dimension / capabilities / `logdensity` finite at
      reasonable θ / `logdensity_and_gradient` consistency
      (`test/regression/test_logdensity.jl`).

### M2 — NUTS sampling wrapper — DONE (2026-04-26)

- [x] `nuts_sample(model, y, n_samples; n_adapts, init_θ,
      init_from_inla, target_acceptance, rng, laplace, drop_warmup,
      progress)`. `n_samples` is post-warmup (Stan convention).
- [x] `init_from_inla::Union{Nothing, INLAResult}` accepts the result
      object directly; cold start uses `initial_hyperparameters(model)`.
- [x] Output as `MCMCChains.Chains` with `_hyperparameter_names(model)`
      column names.
- [x] Built on AdvancedHMC's lower-level kernel API
      (`HMCKernel{Trajectory{MultinomialTS}}` + `GeneralisedNoUTurn` +
      `StanHMCAdaptor`) — no Turing or AbstractMCMC dep in v0.1.
- [x] Tests: 1-D θ recovery, 2-D θ with `init_from_inla`, dim-mismatch
      and argument-validation errors
      (`test/regression/test_nuts_sample.jl`).

### M3 — `compare_posteriors(inla_fit, nuts_chain)` — DONE (2026-04-26)

- [x] Posterior-summary diff: per-hyperparameter
      `(name, inla_mean, nuts_mean, mean_abs_diff, inla_sd, nuts_sd,
      sd_rel_diff, flagged)` rows.
- [x] Mean threshold in units of `max(inla_sd, nuts_sd)`; SD threshold
      relative; both configurable (`tol_mean`, `tol_sd`).
- [x] Tests: passes with generous tolerance, flags with tight tolerance
      (`test/regression/test_compare.jl`).
- [x] Ready to wire into tier-3 triangulation gate (Phase D item 4).

### M4 — INLA-within-MCMC (2 weeks, later phase)

- [ ] Outer MCMC on θ with a user-specified proposal.
- [ ] Inner call to LGM's `sample_conditional(lgm, θ, y)`.
- [ ] Requires core LGM to expose `sample_conditional` publicly —
      tracked in `packages/LatentGaussianModels.jl/plans/plan.md` M3.

### M5 — Triangulation test suite (Phase 5)

- [ ] Scotland BYM2: INLA vs Stan-IAR vs NIMBLE vs our NUTS.
- [ ] Meuse SPDE: INLA vs Stan SPDE vs our NUTS.
- [ ] Results archived per release.

## Risk items

- **Turing breaking changes.** Turing has broken `MCMCChains` repr and
  `sample` signatures historically. Pin compat narrowly, nightly-CI
  against Turing master.
- **AD performance on large latent fields.** For dim(x) > 10⁴ the HMC
  gradient becomes expensive. Tier-3 tests keep n small; production
  use of this package is not the point.

## Out of scope for v0.1

- Non-HMC Turing samplers. They are fine to add later; not MVP.
- Variational inference. Separate sub-package if ever needed.

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

### M1 — LogDensityProblems consumption (1 week)

- [ ] Wrap `LatentGaussianModel` into an `AbstractMCMC.LogDensityModel`.
- [ ] Gradient path via `LogDensityProblemsAD` + ForwardDiff (default)
      or ReverseDiff (for dim(θ) ≥ 20).
- [ ] Test: `logdensity` and `logdensity_and_gradient` agree with
      finite differences.

### M2 — NUTS sampling wrapper (1 week)

- [ ] `sample(lgm, ::NUTS, n; init_from_inla, rng, adtype, ...)`.
- [ ] `init_from_inla::Union{Bool, INLAResult}` default `false`; when an
      `INLAResult` is passed, use its hyperparameter posterior mode as
      initial θ.
- [ ] Output as `MCMCChains.Chains`.

### M3 — `compare(inla_fit, nuts_chain)` (1 week)

- [ ] Posterior-summary diff table: mean / sd / quantiles side by side.
- [ ] Highlight entries outside a configurable tolerance.
- [ ] Used by tier-3 triangulation tests.

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

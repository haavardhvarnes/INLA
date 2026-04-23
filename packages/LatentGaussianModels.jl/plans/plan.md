# LatentGaussianModels.jl — package plan

## Goal

The LGM abstraction plus the INLA algorithm. Built on top of `GMRFs.jl`.
Phase 2–5 of the ecosystem roadmap happen here.

## Module layout

```
src/
├── LatentGaussianModels.jl          # main module, exports
├── components/
│   ├── abstract.jl                   # AbstractLatentComponent contract
│   ├── intercept.jl                  # Intercept (trivial, but explicit)
│   ├── fixed.jl                      # FixedEffect (linear predictor)
│   ├── iid.jl                        # IID
│   ├── rw.jl                         # RW1, RW2
│   ├── ar1.jl                        # AR1
│   ├── seasonal.jl                   # Seasonal
│   ├── besag.jl                      # Besag, ICAR
│   ├── bym.jl                        # BYM, BYM2
│   ├── leroux.jl                     # Leroux CAR
│   ├── generic.jl                    # Generic0, Generic1 (user-supplied R)
│   └── kronecker.jl                  # KroneckerComponent (space × time)
├── likelihoods/
│   ├── abstract.jl                   # AbstractLikelihood contract
│   ├── gaussian.jl
│   ├── poisson.jl
│   ├── binomial.jl
│   ├── negbinomial.jl
│   ├── gamma.jl
│   └── links.jl                      # AbstractLinkFunction + concrete
├── priors/
│   ├── abstract.jl                   # AbstractHyperPrior contract
│   ├── pc.jl                         # PC priors (prec, phi, range)
│   ├── gamma.jl                      # Gamma prior on precisions
│   └── lognormal.jl
├── model.jl                          # LatentGaussianModel struct, projector
├── inference/
│   ├── abstract.jl                   # AbstractInferenceStrategy
│   ├── empirical_bayes.jl
│   ├── laplace.jl                    # inner Newton + Laplace for x | θ, y
│   ├── inla.jl                       # outer θ integration
│   └── integration/
│       ├── ccd.jl                    # CCD design matrices (Rue & Martino 2007)
│       ├── grid.jl
│       ├── gauss_hermite.jl
│       └── grid_mcmc.jl
├── result.jl                         # INLAResult + accessor API
└── diagnostics.jl                    # DIC, WAIC, CPO, PIT, mlik

# Note: the @lgm formula macro lives in the separate LGMFormula.jl
# sub-package (ADR-008). The Turing/NUTS bridge lives in LGMTuring.jl
# (ADR-009). Neither is in core LGM.

ext/
├── LGMHCubatureExt.jl
├── LGMIntegralsExt.jl
└── LGMMakieExt.jl

test/
├── runtests.jl
├── regression/                       # closed-form
├── oracle/                           # vs R-INLA
└── triangulation/                    # vs Stan/NIMBLE
```

## Milestones

### M1 — Components + likelihoods scaffolding (Phase 2 start)

- [ ] `AbstractLatentComponent` contract.
- [ ] `Intercept`, `FixedEffect`, `IID`.
- [ ] `AbstractLikelihood` contract; `Gaussian`, `Poisson`, `Binomial`.
- [ ] `AbstractLinkFunction`: Identity, Log, Logit, Probit, Cloglog.
- [ ] `AbstractHyperPrior`: PC-prec, Gamma, LogNormal.
- [ ] `LatentGaussianModel` struct, Gaussian-likelihood closed-form fit.
- [ ] First oracle test: simple IID Gaussian model vs R-INLA.

### M2 — Areal models (Phase 2 main)

- [ ] `Besag` with correct disconnected-components handling.
- [ ] `BYM`, `BYM2` with Sørbye-Rue scaling.
- [ ] `Leroux`.
- [ ] `RW1`, `RW2` with cyclic option.
- [ ] PC-phi prior for BYM2.
- [ ] **Scotland lip cancer milestone:** fit matches R-INLA within tolerance.

### M3 — Laplace inference (Phase 3 start)

- [ ] Inner Newton for `argmax_x log p(x, y | θ)`.
- [ ] Full Laplace approximation for marginal `p(x_i | θ, y)`.
- [ ] `FactorCache` reuse across Newton steps.
- [ ] `EmpiricalBayes` strategy complete.

### M4 — INLA integration (Phase 3 main)

- [ ] θ-mode finding via Optimization.jl (LBFGS / trust region).
- [ ] Hessian at the mode, eigendecomposition.
- [ ] CCD design on the eigenbasis.
- [ ] Grid strategy for dim(θ) ≤ 2.
- [ ] Gauss-Hermite strategy.
- [ ] `INLAResult` with accessors matching R-INLA's `summary.inla` layout.
- [ ] **Pennsylvania lung cancer milestone:** Poisson BYM2 matches R-INLA.

### M5 — Diagnostics (Phase 3 tail)

- [ ] DIC, WAIC, CPO, PIT.
- [ ] Log marginal likelihood (integration and Gaussian).
- [ ] `summary(fit)` reproduces R-INLA's layout.

### M6 — Additional likelihoods (Phase 3 / 5)

- [ ] NegativeBinomial (disease mapping essential).
- [ ] Gamma.
- [ ] Beta-binomial (malaria-prevalence use case).
- [ ] Zero-inflated Poisson.

### M7 — `LogDensityProblems` conformance (Phase 3 end)

Core LGM commits only to the `LogDensityProblems.jl` seam; downstream
sampler integrations are sub-packages (ADR-008, ADR-009).

- [ ] `LogDensityProblems.capabilities(::LatentGaussianModel)`.
- [ ] `LogDensityProblems.logdensity(::LatentGaussianModel, θ)`.
- [ ] `LogDensityProblems.logdensity_and_gradient(...)`.
- [ ] `LogDensityProblems.dimension(::LatentGaussianModel)`.
- [ ] Public `sample_conditional(lgm, θ, y; rng)` method — needed by
      `LGMTuring.jl`'s INLA-within-MCMC loop.

Formula sugar (`@lgm`) ships separately in `packages/LGMFormula.jl/`;
NUTS/HMC and INLA-within-MCMC ship separately in `packages/LGMTuring.jl/`.

## Open design questions

- **Projector `A`:** field on model vs its own abstract type. See ADR-005.
- **Space-time Kronecker components:** how to express a BYM2 ⊗ AR1 cleanly
  while keeping the precision lazy. See Kronecker.jl vs SciMLOperators.jl
  vs hand-rolled.
- **Simplified Laplace:** Rue-Martino 2009 correction terms. Defer to v0.3.
- **Multiple likelihoods (joint models):** API design deferred to Phase 5.

## Validation checklist before v0.1 release

- [ ] All canonical datasets in `plans/testing-strategy.md` pass oracle
      tier.
- [ ] Scotland, Germany, Pennsylvania, Ohio all pass textbook-reproduction
      tier.
- [ ] Documenter.jl site published with at least 4 vignettes (areal,
      temporal, spatial, spatio-temporal).
- [ ] Zero `Any` in hot-path signatures (verified via JET).
- [ ] Aqua tests clean.

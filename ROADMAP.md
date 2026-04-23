# Roadmap

Phased delivery. Each phase ends with runnable, tested output. Phase 0 can
run concurrently with scaffolding in Phases 1 and 2, but substantive Phase 2
content (areal components) depends on Phase 1 M1–M3 (graph + concrete GMRFs
+ sampling). Phase 4 (SPDE end-to-end) depends on Phase 3 (INLA algorithm).
Phase 3 is the critical path.

## MVP — the go/no-go check

Before Phase 3 starts, we should have:

1. `AbstractGMRF` + sparse-graph + symmetric-Q implementation in `GMRFs.jl`.
2. Unconditional sampling from `N(0, Q⁻¹)` with sum-to-zero constraints via
   LinearSolve.jl (CHOLMOD backend).
3. `Besag` and `IID` latent components in `LatentGaussianModels.jl`.
4. A Gaussian-likelihood LGM closed-form solver (no Laplace yet) producing
   posterior mean and marginal variances of the latent field.
5. End-to-end vignette: Scottish lip cancer Gaussian-approximated BYM fit
   vs R-INLA, agreement on posterior means within 1%.

If this works, the design is load-bearing. If it doesn't, the disagreement
points to something structural (constraint correction, scaling, parameter
conventions) that must be fixed before the rest of the stack is built.

## Phase 0 — scaffolding (4–6 weeks)

- Package skeletons, CI (GitHub Actions: tests on Julia 1.10 LTS + current
  stable + nightly, on Linux/macOS/Windows).
- JuliaFormatter with SciML style, Aqua.jl, JET.jl in CI.
- Documenter.jl setup with a shared theme across the three packages.
- First ADRs written: package split, dependency policy, macro policy.
- R-INLA fixture generation harness: `scripts/generate-fixtures/` with
  pinned R version, renv lockfile, script to emit JLD2 files.

## Phase 1 — GMRF foundations (3–4 months, `GMRFs.jl`)

Deliverables listed in `packages/GMRFs.jl/plans/plan.md`. Milestone:
`qsample`, `qinv`, log-determinant, constraint correction all matching
R-INLA to 1e-10 on IID, RW1, RW2, AR1, seasonal, generic0/1.

## Phase 2 — Areal models (2 months, `LatentGaussianModels.jl`)

Besag, ICAR with sum-to-zero per connected component, BYM, BYM2, Leroux.
PC priors as first-class `AbstractHyperPrior`s. Scaling of intrinsic GMRFs.
Milestone: Scotland lip cancer BYM2 posterior matches R-INLA.

## Phase 3 — Laplace + INLA algorithm (4–6 months)

Inner Newton for mode of `x∣θ,y`. Simplified and full Laplace
approximations. Hyperparameter mode, Hessian, CCD/grid integration over θ.
Gaussian, Poisson, Binomial, NegativeBinomial, Gamma likelihoods. Selected
inverse (Takahashi recursion) — explicit sub-milestone, see
`plans/decisions.md`. Milestone: Poisson spatial GLMM marginal posteriors
within 5% of R-INLA hyperparameters, 1% of latent-field means.

## Phase 4 — SPDE (3–4 months, `INLASPDE.jl`)

FEM assembly of C, G₁, G₂ on `Meshes.jl` triangulations (α ∈ {1, 2} first).
PC priors on range and σ (Fuglstad et al. 2019). Projector matrices as
`SciMLOperators`. Milestone: Meuse zinc SPDE fit end-to-end vs R-INLA.

## Phase 5 — Extensions (ongoing)

Space-time separable via Kronecker components. Joint likelihoods. User-
defined components (the rgeneric/cgeneric equivalent, already free from
architecture). Model comparison: DIC, WAIC, CPO, PIT, log marginal
likelihood.

## Phase 6 — Performance polish (ongoing)

Threaded CHOLMOD / Pardiso paths via LinearSolve. Lazy Kronecker operators.
Fractional-α SPDE via rational approximations. GPU factorization via
CUDSS.jl for very large meshes.

## Realistic calendar

- MVP: 2–3 months full-time-equivalent work from a practitioner who knows
  the material.
- Competitive parity on small/medium problems (n ≲ 10⁴): 9–12 months FTE.
- Feature-sufficient for 80% of real spatial-epidemiology use cases:
  18–24 months FTE.
- Feature parity with R-INLA's long tail: not a realistic goal.

## Non-goals

- Byte-identical agreement with R-INLA. Different orderings and accumulation
  orders mean last-digit agreement isn't achievable.
- Covering every `inla.rgeneric` extension ever written. The rgeneric
  replacement is automatic from the architecture, but the long tail of
  existing esoteric R-INLA features is out of scope.
- Supporting R's formula syntax verbatim. A macro sugar layer may parse a
  close-but-not-identical variant.

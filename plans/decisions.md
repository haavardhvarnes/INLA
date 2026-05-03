# Architecture decision records

Each significant design decision gets a short entry here. Format:

```
## ADR-NNN: Title

Status: Accepted | Proposed | Superseded by ADR-MMM
Date: YYYY-MM-DD

### Context
What's the situation that forced a decision?

### Decision
What did we choose?

### Consequences
What does this buy us, what does it cost us, what's the escape hatch?
```

Numbering is sequential; never renumber.

---

## ADR-001: Package split into GMRFs / LatentGaussianModels / INLASPDE

Status: Accepted
Date: 2026-04

### Context
A single monolithic `INLA.jl` is tempting for simplicity. But the GMRF core
is independently useful to people doing sparse Gaussian modeling (Markov
random fields, disease mapping with Stan, image restoration, 4D-Var in
data assimilation). SPDE machinery brings in Meshes.jl and is irrelevant to
many users.

### Decision
Three packages:
- `GMRFs.jl` — numerical core, dependency-light.
- `LatentGaussianModels.jl` — LGM abstraction + INLA algorithm, depends on GMRFs.
- `INLASPDE.jl` — SPDE/FEM, depends on LatentGaussianModels + Meshes.

Kept in a monorepo during planning; each directory is a valid Julia
package and can move to its own repo once stable.

### Consequences
- Good: users install only what they need; GMRFs.jl attracts contributors
  outside the INLA community.
- Good: clear dependency layering; changes to SPDE can't break GMRFs.
- Cost: coordinated releases across three packages require discipline. A
  shared CompatHelper setup and `juliaup/pkg-release-manager` style tooling
  helps.

---

## ADR-002: SciML's LinearSolve.jl as the sparse factorization backend

Status: Accepted
Date: 2026-04

### Context
R-INLA's `GMRFLib_sparse_interface` abstracts over TAUCS, PARDISO, and
BAND solvers. We need the same abstraction. Options:
1. Wrap CHOLMOD directly via SuiteSparse.jl.
2. Use LinearSolve.jl's abstract interface.
3. Build our own dispatcher.

### Decision
LinearSolve.jl. Cholmod/KLU/Pardiso are all already exposed through it.
Symbolic-factorization reuse works via `init(prob)` + `solve!(cache)`.

### Consequences
- Good: zero wheel reinvention; get multi-backend for free.
- Good: standard SciML pattern, familiar to the ecosystem.
- Cost: LinearSolve has its own release cadence; we track master in CI to
  catch breakages.
- Escape hatch: if we hit a case where LinearSolve's overhead matters (inner
  loop allocation is suspected), drop to direct `cholmod_factorize!` for
  that specific hot path, with a benchmark justifying it.

---

## ADR-003: Multiple dispatch, not macros, for the primary API

Status: Accepted
Date: 2026-04

### Context
Turing's `@model` is idiomatic Julia for probabilistic programming. R-INLA
uses R formulas. Stan has its own DSL. For an LGM library, these trace-
based/formula-based APIs are a poor fit — the model is static structure,
not procedural code.

### Decision
Concrete types and multiple dispatch are the primary API. Optional formula
sugar macro (`@lgm`) expands to the explicit constructor. Full policy in
`plans/macro-policy.md`.

### Consequences
- Good: type-stable, AD-friendly, extensible by third parties via
  `struct + methods`, no DSL for users to learn.
- Good: the `rgeneric`/`cgeneric` equivalent is automatic — users define
  a subtype of `AbstractLatentComponent`, no C callbacks needed.
- Cost: R-INLA migrants face a syntax change. The macro sugar narrows the
  gap but doesn't eliminate it.

---

## ADR-004: Selected inversion (Takahashi recursion) as an explicit risk

Status: Proposed
Date: 2026-04

### Context
INLA requires marginal variances `diag(Q⁻¹)` for the Laplace approximation.
R-INLA uses a specialized C implementation of the Takahashi recursion,
which is not trivially reimplemented. Julia has `SelectedInversion.jl` (young),
CHOLMOD's `sparseinv` (not exposed conveniently), or we implement Takahashi
ourselves.

### Decision (proposed)
Evaluate `SelectedInversion.jl` early in Phase 3. If it meets correctness
and performance requirements on our benchmark suite, use it. If not,
implement Takahashi natively using sparse Cholesky factor access. Either
way, this is a named Phase 3 sub-milestone with a budget of 1–2 months.

### Consequences
- Good: if SelectedInversion.jl works, we save months.
- Cost: if it doesn't, we're writing a numerically delicate algorithm.
- Mitigation: both paths start with the same correctness tests against
  dense `inv(Q)` diagonal.

---

## ADR-005: Projector matrix A as a model field in v0.x, possibly promoted later

Status: Proposed
Date: 2026-04

### Context
R-INLA's `inla.stack` machinery conflates observation-to-latent mapping,
data stacking, and effect indexing. A clean Julia version needs to decide
whether the projector A is a field on `LatentGaussianModel` or its own
abstract type `AbstractObservationMapping`.

### Decision (proposed)
v0.1–0.3: `projector::A` as a model field, with a `IdentityProjector()` for
areal models and a concrete `MeshProjector(mesh, locations)` for SPDE.
Re-evaluate once misaligned and joint-likelihood models land in Phase 5.

### Consequences
- Good: simpler for the MVP; easier to explain.
- Cost: likely to be refactored to an abstract type by Phase 5, producing
  a minor breaking change.
- Mitigation: the constructor API `LatentGaussianModel(; projector = ...)`
  stays the same even if the field type changes.

---

## ADR-006: Full Laplace is the Phase-3 default; `simplified.laplace` deferred

Status: Accepted
Date: 2026-04

### Context
R-INLA ships `simplified.laplace` (Rue-Martino 2009) as the default
`strategy`. It is noticeably cheaper than full Laplace and gives more
accurate tails on skewed likelihoods. Implementing it correctly requires
the Rue-Martino correction terms, which are numerically delicate.

### Decision
v0.1 ships **full Laplace** as the default `strategy`. `Gaussian`
(fast preview) is also shipped. `SimplifiedLaplace` is deferred to v0.3
when we have tier-2 confidence in the full Laplace path.

### Consequences
- **Cost:** posterior tails on Poisson/Binomial with extreme counts will
  differ visibly from R-INLA at the first-releases. Users comparing to
  R-INLA at the 1% tolerance in tier-2 tests may see failures
  specifically on skewed likelihoods. Document the divergence
  prominently and widen tier-2 tolerance retroactively on the
  skewed-likelihood fixtures.
- **Good:** v0.1 lands sooner; the simpler algorithm is easier to
  validate in isolation.
- **Escape hatch:** `strategy = :gaussian` for a fast preview; explicit
  warning at fit time on likelihoods we know are tail-sensitive.

### References
- Rue, Martino, Chopin (2009), §3.2.
- `plans/defaults-parity.md` "Default Laplace strategy" section.

### Amendment 2026-04-24 — marginal-reconstruction scope split

The original ADR conflated two distinct things that R-INLA also
calls `strategy`:

1. **Top-level inference strategy** — the Laplace step used inside
   the INLA outer loop (Gaussian / simplified Laplace / full Laplace
   per marginal). This is what the original decision governed.
2. **Posterior marginal reconstruction** — how `p(x_i | y)` is
   reassembled from the θ-grid of per-θ Laplaces. R-INLA also
   exposes `:gaussian`, `:simplified_laplace`, `:laplace` on this
   step (post-processing, not inference-time).

Commit `85db314` added a `strategy` **kwarg on
`posterior_marginal_x`** covering case (2) only — opt-in, default
remains `:gaussian`. Likelihood contract extended with closed-form
`∇³_η_log_density` for Gaussian / Poisson / Binomial plus a
central-difference fallback. Collapses to Gaussian on
quadratic-in-η likelihoods; verified in
[test_simplified_laplace.jl](packages/LatentGaussianModels.jl/test/regression/test_simplified_laplace.jl).

This does **not** reverse the v0.3 deferral. The following items
remain Phase-3-late / v0.3 scope:

- Flipping the *default* on `posterior_marginal_x` from
  `:gaussian` to `:simplified_laplace` — blocked on the
  Pennsylvania Poisson oracle in the replan's Phase C.
- Adding an analogous kwarg to the inference-time Laplace strategy
  (case 1 above). The top-level default is and remains full
  Laplace per this ADR.
- The full Rue-Martino per-marginal `:laplace` strategy (the
  expensive per-marginal re-Laplace).

Rationale for keeping the landed work rather than reverting: the
cubic-derivative closed forms are real correctness infrastructure
needed in Phase C regardless of when the default flips; losing them
incurs the same re-implementation cost later with no payoff now.

Status of this ADR: Accepted, amended 2026-04-24.

---

## ADR-007: DelaunayTriangulation.jl for mesh generation; fmesher wrap as fallback sub-package

Status: Accepted
Date: 2026-04

### Context
R-INLA relies on `fmesher` (a Lindgren C++ tool, now externalized as
`inlabru-org/fmesher`) for constrained Delaunay triangulation with
boundary refinement and extension buffers. Julia's native option is
`DelaunayTriangulation.jl`, which is actively maintained and
feature-rich but has not been validated against `fmesher` output on
SPDE-grade meshes.

### Decision
Default path: `DelaunayTriangulation.jl`. `INLASPDE.jl` M2 includes a
mesh-quality comparison step against `fmesher` output on a fixed
boundary. If that comparison reveals quality gaps that affect SPDE
accuracy (minimum angle, maximum edge length, boundary-refinement
behavior), the fallback is a **new sub-package `INLASPDEFmesher.jl`**
that wraps `fmesher` via BinaryBuilder and exposes an
`fmesher_mesh_2d(...)` alternative constructor. The fallback is a
sub-package, not a weakdep, because binary-artifact dependencies are
heavy and the user should opt in explicitly.

### Consequences
- **Good:** native Julia is the default and works end-to-end without
  binary artifacts.
- **Cost:** if mesh quality is inadequate, Phase 4 M2 slips by 2–4 weeks
  while the fmesher fallback is packaged.
- **Escape hatch:** the fmesher sub-package is planned-but-deferred;
  creating it is a one-time cost not on the critical path for v0.1
  SPDE.

### References
- `packages/INLASPDE.jl/plans/plan.md` M2, Risk items.
- `plans/dependencies.md` — fmesher note.

### Resolution 2026-04-26 — native path sufficient, gate relaxed

The M3 parity fixtures (`6d0784d`) measured native mesh quality
against fmesher on the three reference boundaries. The strict
parity gate (5% vertex count, 0.95× angle, 1.05× edge) failed on
all three; the failure is structural (DT.jl uses an
equilateral-area Ruppert bound; fmesher uses per-edge bisection),
not a bug. The Meuse SPDE oracle (INLASPDE M5) nevertheless
**passes within tolerance using the native mesh**, so mesh quality
is sufficient for SPDE work in v0.1.

**Decision (resolution):** declare native path sufficient. Relax the
parity gate from strict fmesher equivalence to a regression floor
on DT.jl's measured behaviour:

|                | original gate         | resolved gate        |
|----------------|-----------------------|----------------------|
| `rel_vcount`   | ≤ 0.05                | ≤ 0.50               |
| `min_angle_J`  | ≥ max(20°, 0.95·R)    | ≥ 25.0°              |
| `max_edge_J/R` | ≤ 1.05                | ≤ 2.5                |

The resolved gate is locked in as plain `@test` (no
`@test_broken`); a DT.jl regression that materially degrades mesh
quality now fails CI immediately.

`INLASPDEFmesher.jl` remains a planned-but-deferred fallback. The
trigger to actually build it is a downstream user reporting that
mesh quality is biting them on a real fit, not the strict gate
failing in isolation.

ADR-007 is hereby **closed**; further `INLASPDEFmesher.jl` work is
tracked as a v0.2 candidate via `plans/replan-2026-04.md` Phase D.

### References (resolution)
- `packages/INLASPDE.jl/plans/plan.md` M3 parity table.
- `packages/INLASPDE.jl/test/oracle/test_fmesher_parity.jl`.
- `plans/replan-2026-04.md` Phase D, item 1.

---

## ADR-008: `@lgm` macro lives in a separate `LGMFormula.jl` package

Status: Accepted
Date: 2026-04

### Context
The Tier-2 formula sugar `@lgm y ~ 1 + f(region, Besag(W))` is desirable
for R-INLA migrants. Parsing it cleanly benefits from StatsModels.jl's
`@formula` infrastructure for the fixed-effects side. But StatsModels
pulls StatsBase / StatsFuns / DataAPI / Tables into the dep tree, and
Julia 1.9+ extensions cannot export new symbols — so `@lgm` cannot
live in a weakdep extension of core LGM.

### Decision
The macro and its StatsModels dependency live together in a separate
sub-package `packages/LGMFormula.jl/`. Core `LatentGaussianModels.jl`
ships only the Tier-1 explicit constructor. Users who want formula
sugar install one extra package:
```julia
using Pkg; Pkg.add("LGMFormula")
using LatentGaussianModels, LGMFormula
```

### Consequences
- **Good:** core LGM's dep tree stays narrow; StatsModels is not forced
  on users who don't want it; the macro can be a proper exported
  symbol with its own namespace.
- **Good:** macro-policy stays honest — Tier 1 is complete without the
  macro *and* without the macro's host package.
- **Cost:** one extra `Pkg.add` for users migrating from R-INLA. A line
  in the getting-started guide.
- **Scheduling:** LGMFormula.jl is not on the MVP / Phase-3 critical
  path. Build it once the constructor API is stable (end of Phase 3).

### References
- `plans/macro-policy.md` (rewritten alongside this ADR).
- `plans/dependencies.md` sub-packages table.

---

## ADR-009: Turing / HMC bridge lives in a separate `LGMTuring.jl` package

Status: Accepted
Date: 2026-04

### Context
A Turing / AdvancedHMC bridge has two use cases: (1) tier-3
triangulation against HMC posteriors; (2) INLA-within-MCMC flows
(Gómez-Rubio Ch. 5–7) that wrap INLA's conditional `p(x | θ, y)` inside
an outer MCMC loop on θ. Turing's transitive closure is 20–40 s of
TTFX; putting Turing even in `[weakdeps]` of core LGM inflates the
install and release surface.

### Decision
Core LGM commits **only** to `LogDensityProblems.jl` conformance —
implementing `LogDensityProblems.capabilities`, `logdensity`,
`logdensity_and_gradient`, `dimension`. Nothing Turing-specific in
core. A separate sub-package `packages/LGMTuring.jl/` depends on
LatentGaussianModels + Turing + AdvancedHMC and provides:

- `sample(lgm, ::NUTS, n; init_from_inla = true, kwargs...)` wrapper.
- INLA-within-MCMC loop using LGM's exposed
  `sample_conditional(lgm, θ, y)` — see P7 in the initial review.
- `compare(inla_fit, nuts_chain)` diagnostic for triangulation tier.

### Consequences
- **Good:** core LGM's `LogDensityProblems` conformance is already
  useful to anyone who wants to use AdvancedHMC or Pathfinder directly,
  without our bridge.
- **Good:** Turing's release cadence is isolated to one sub-package.
- **Cost:** triangulation-tier tests (Stan/NIMBLE/Turing cross-checks)
  move to `packages/LGMTuring.jl/test/triangulation/`, not core LGM's
  `test/triangulation/`. This means core LGM's tier-3 test list is
  shorter — acceptable, tier-3 is slow and not PR-gating anyway.
- **Requirement:** LGM must expose `sample_conditional(lgm, θ, y)` as
  public API, not internal. Add to the component/inference contract
  in M3.

### References
- `plans/macro-policy.md` Tier 3 description.
- `packages/LatentGaussianModels.jl/plans/plan.md` M3 (conditional
  sampling as public API).

---

## ADR-010: Public kwargs mirror R-INLA names with snake_case + symbol-or-type dual input

Status: Accepted
Date: 2026-04

### Context
R-INLA users coming to the Julia port will look for familiar kwargs:
`int.strategy`, `control.compute$dic`, `scale.model`. A pure-Julia API
using only `AbstractIntegrationScheme` type instances would be
correct-but-alien. A pure R-style API would clash with Julia
conventions. The design should accept both.

### Decision
Public fit-time and component kwargs mirror R-INLA's names with
snake_case translation (`int.strategy` → `int_strategy`,
`scale.model` → `scale_model`). Each such kwarg accepts either:

- a **symbol** (`:ccd`, `:grid`, `:laplace`) — resolves to the canonical
  type via an internal table; or
- a **type instance** (`CCD()`, `Grid(n = 15)`, `Laplace()`) — used
  directly for advanced configuration.

Nested R-INLA `control.*` groups are flattened into prefixed flat
kwargs (`compute_dic`, `compute_waic`) rather than nested NamedTuples.

The complete kwarg table lives in `plans/defaults-parity.md`. Defaults
match R-INLA's except where explicitly documented in the divergences
section.

### Consequences
- **Good:** R-INLA users keep muscle memory; Julia users have
  type-directed autocompletion via the type-instance form.
- **Good:** `help?>` on a specific kwarg type (`?CCD`) still yields
  useful documentation independently of the fit function.
- **Cost:** every kwarg-accepting function needs the
  symbol-dispatch boilerplate. Centralize it in
  `LatentGaussianModels.Inference._resolve(::Val{:ccd})`, etc.

### References
- `plans/defaults-parity.md` kwargs table.

---

## ADR-011: Top-level API is `fit(model, y, strategy; kwargs...)`; `inla(model, y; kwargs...)` is a convenience alias

Status: Accepted
Date: 2026-04

### Context
Two candidate public entry points appeared in earlier drafts:

- `fit(model, y, strategy; kwargs...)` — dispatched on
  `AbstractInferenceStrategy`, following the StatsBase /
  MLJ / Distributions convention for model-fitting in Julia.
- `inla(model, y; kwargs...)` — a short, R-INLA-familiar name with
  strategy selected via kwarg.

READMEs used `inla(...)`; the LGM CLAUDE.md used `fit(...)`. Pick one
canonical entry point; define the other as a thin wrapper.

### Decision
The **canonical** entry point is

```julia
fit(model, y, strategy = INLA(); kwargs...) -> AbstractInferenceResult
```

with `strategy::AbstractInferenceStrategy`. Dispatch on the strategy
type selects the algorithm; kwargs configure it per ADR-010.

`inla(model, y; kwargs...)` is a thin convenience alias defined as

```julia
inla(model, y; kwargs...) = fit(model, y, INLA(); kwargs...)
```

It exists because R-INLA users expect to type it and because it scans
as a domain verb in examples. It is not the canonical method, only
sugar.

Likewise:
- `empirical_bayes(model, y; kwargs...) = fit(model, y, EmpiricalBayes(); kwargs...)`
- `laplace(model, y; kwargs...) = fit(model, y, Laplace(); kwargs...)`

### Consequences
- **Good:** the underlying dispatch is via multiple-dispatch on the
  strategy type (StatsBase-idiomatic), so third-party strategies slot
  in by subtyping `AbstractInferenceStrategy` and defining a `fit`
  method — no change to the LGM package.
- **Good:** the aliases `inla`, `laplace`, `empirical_bayes` keep
  READMEs readable and preserve muscle memory for R-INLA migrants.
- **Cost:** two ways to do the same thing. Documentation must pick one
  style per page and not mix. Convention: the *quickstart* vignettes
  use `inla(...)`; the *reference* docs use `fit(model, y, INLA())`.
- **Cost:** third-party strategy authors must know to define `fit`,
  not a custom verb. Document this clearly in
  `AbstractInferenceStrategy`'s docstring.

### References
- `plans/macro-policy.md` Tier 3 bridge convention.
- `plans/defaults-parity.md` kwargs table — all kwargs listed apply
  to both `fit(..., INLA(); kwargs...)` and `inla(...; kwargs...)`.

---

## ADR-012: Adopt `SelectedInversion.jl` for sparse selected inverse

Status: Accepted
Date: 2026-04-23

### Context

ADR-004 named selected inversion (Takahashi recursion for
`diag(Q⁻¹)` on the sparsity pattern of the Cholesky factor) as the
single biggest Phase-3 numerical risk: R-INLA uses a specialized C
implementation, Julia options were (a) `SelectedInversion.jl` (young),
(b) CHOLMOD's `sparseinv` (not exposed cleanly), or (c) native
Takahashi on the sparse Cholesky factor.

At Phase 3 entry the GMRFs.jl `marginal_variances` reference impl
densifies `Q` and is gated on `n < 1000`. The INLA outer loop's
posterior-variance computation uses the same densification per design
point — a hard wall for any realistic problem.

Empirical evaluation of `SelectedInversion.jl` v0.2:
- API: `selinv(Q::SparseMatrixCSC; depermute = true)` returns a
  NamedTuple `(Z::SparseMatrixCSC, p::Vector{Int})`. `diag(Z)` gives
  the marginal variances directly.
- Correctness: matches `diag(inv(Matrix(Q)))` to ~1e-16 on band,
  random-SPD, and RW-style Laplacian matrices.
- Dependencies: SparseArrays + LinearAlgebra + SuiteSparse (all
  already transitive via LinearSolve) — no new heavy deps.
- Failure mode: throws `PosDefException` on non-PD input; standard
  LinearAlgebra behavior.

### Decision

Adopt `SelectedInversion.jl` as a **core `[deps]` of `GMRFs.jl`** (not
LGM — per ADR-001 layering, selected inversion is numerical core).
Route LGM's posterior marginal-variance path through
`GMRFs.marginal_variances`. The reference dense impl is kept as a
correctness oracle for small `n`, reachable via `method = :dense`.

### Consequences

- **Good:** unblocks any `n > ~1000` problem; no native Takahashi
  implementation needed, saving the 1–2 months of schedule budget
  named in ADR-004.
- **Good:** fixture tests can now compare directly against R-INLA's
  `inla.qinv` output at scales matching real spatial-epidemiology
  problems.
- **Cost:** one more direct dep; maintenance risk if
  `SelectedInversion.jl` goes stale. Mitigation: pin compat bounds,
  keep the dense reference path for fallback.
- **Cost:** does not solve `logdet(Q)` for intrinsic GMRFs —
  generalised log-determinant on the non-null subspace is a separate
  problem, tracked in `plans/defaults-parity.md`.

### References
- ADR-004 — Selected inversion named risk.
- `plans/dependencies.md` — updated core deps table for GMRFs.jl.

---

## ADR-013: SPDE2 v0.1 supports α ∈ {1, 2}; internal hyperparameters are `(log τ, log κ)`

Status: Accepted
Date: 2026-04-23

### Context

The Lindgren-Rue-Lindström 2011 SPDE-Matérn link yields a closed-form
sparse precision `Q(τ, κ)` only for integer `α`. R-INLA's
`inla.spde2.matern` defaults to `α = 2` (which corresponds to Matérn
smoothness `ν = α - d/2 = 1` in 2D). Fractional `α` via Bolin-Kirchner
rational approximations is deferred to v0.3 per
`packages/INLASPDE.jl/plans/plan.md`.

Two open parameterisation questions:

1. **User-scale vs internal scale.** PC priors are most naturally stated
   on the user-scale pair `(range ρ, marginal σ)` per Fuglstad et al.
   2019. The Laplace inner loop works on unconstrained real coordinates.
2. **α surface.** Expose `α` as a compile-time type parameter
   (`SPDE2{1}`, `SPDE2{2}`) or a runtime field.

### Decision

- SPDE2 v0.1 supports `α ∈ {1, 2}`. `α = 2` is the default, matching
  R-INLA. `α` is a **type parameter** (`SPDE2{α}`) so the assembly
  path branches at compile time and the precision-matrix method is
  fully type-stable.
- Internal hyperparameters on the Laplace scale are `θ = [log τ, log κ]`.
  PC-Matérn priors are authored on `(ρ, σ)` and transformed to
  `(log τ, log κ)` via the closed-form Jacobian (cf. Fuglstad et al.
  2019 eqs. 7-8). Users never see `(log τ, log κ)` directly; posterior
  summaries are reported on `(ρ, σ)` via `user_scale`.
- `log_hyperprior(spde, θ)` evaluates the PC density on user scale and
  adds the Jacobian term so that the Laplace-scale posterior is
  normalised consistently with R-INLA's.

### Consequences

- **Good:** type-stable `precision_matrix(::SPDE2{α}, θ)` for both
  α values; no run-time dispatch in the inner Newton loop.
- **Good:** user-facing priors are on the domain statisticians think
  in (range/sd), matching R-INLA's defaults — lowers the parity risk
  in tier-2 oracle tests.
- **Cost:** the Jacobian from `(ρ, σ)` to `(log τ, log κ)` must be
  implemented carefully and regression-tested against R-INLA's
  `inla.pc.prior.matern` — a known source of sign/factor bugs.
- **Escape hatch for fractional α:** v0.3 will add a separate
  `SPDEFractional` concrete type; the type parameter on `SPDE2` does
  not preclude this.

### References
- `packages/INLASPDE.jl/plans/plan.md` M2 and M4.
- Lindgren, Rue, Lindström 2011. SPDE.
- Fuglstad, Simpson, Lindgren, Rue 2019. PC priors for Gaussian random
  fields.

---

## ADR-014: `main` is fast-forwarded to `claude/hungry-pascal`; main becomes the integration branch going forward

Status: Accepted
Date: 2026-04-24

### Context

Between 2026-04-23 13:46 and 2026-04-24 11:15, 17 feature commits
landed on branch `claude/hungry-pascal` while `main` remained at
`9c2f69d`. A status audit on 2026-04-24 discovered the gap:

- `git merge-base main claude/hungry-pascal` resolves to `main`'s
  HEAD — the branches share a linear history, no divergence.
- Every commit on `hungry-pascal` is by the same author as on `main`.
- No pull request, review, or CI gate was configured; the branch was
  simply the workspace where work continued after `main` stopped
  being advanced.
- Roadmap progress actually landed on `hungry-pascal`: MVP go/no-go
  (Scotland BYM2, ADR-006 scope, ADR-012, ADR-013), much of Phase 3
  (simplified-Laplace correction, DIC/WAIC/CPO/PIT, posterior
  marginals, linear constraints in Laplace), and Phase 4 M1–M6-A
  (FEM assembly, SPDE2, mesh generation, MeshProjector, Meuse
  oracle, GeoInterface ext, rasters predict/quantile, MakieExt).

Reading the roadmap against `main` gave a misleadingly pessimistic
picture (late Phase 1). The true state is through Phase 4 M6-A.

### Decision

1. **Fast-forward `main` to `claude/hungry-pascal`.** The update is
   `git checkout main && git merge --ff-only claude/hungry-pascal`.
   No cherry-picking, no squash — the 17 commits are individually
   scoped (Conventional Commits, per-feature) and bisect-able as-is.
2. **Delete `claude/hungry-pascal`** after the fast-forward. It has
   no independent meaning.
3. **Adopt a no-stale-main rule going forward.** Either (a) work
   directly on `main` until branch protection lands, or (b) once
   branch protection and CI are enabled per
   `plans/initial-commits.md`, open a PR per feature branch and
   require merge before starting the next unrelated piece of work.
   Leaving a work branch more than one working day ahead of `main`
   without a PR is the concrete anti-pattern this ADR names.
4. **Record roadmap drift explicitly.** After the merge,
   `ROADMAP.md` is updated to reflect the new baseline, and a
   replan document lands in `plans/replan-2026-04.md` covering
   Phase B–E per the status review.

### Consequences

- **Good:** a single canonical branch; external readers (and Claude
  Code sessions) evaluating project status will not be misled by a
  stale `main`.
- **Good:** all prior review benefits are still available — the
  commit granularity on `hungry-pascal` is already per-feature, so
  `git log` on the merged `main` reads the same way a review would
  have produced.
- **Cost:** the ADR numbering on this branch (which only sees through
  ADR-011) must be reconciled at merge time. `hungry-pascal` adds
  ADR-012 and ADR-013; this branch adds ADR-014. The textual merge
  is trivial — no ADR is renumbered, only concatenated in order —
  but the merge commit must verify the sequence 012, 013, 014 is
  monotonic with no gaps before the fast-forward is accepted.
- **Cost:** one-off effort to backfill branch protection rules on
  GitHub (per `plans/initial-commits.md` §"Branch protection
  rules"). This is independent of the merge itself but should land
  in the same working session to prevent recurrence.

### Follow-up items flagged by this ADR (tracked but not resolved here)

- **ADR-006 divergence.** `85db314` landed a simplified-Laplace
  skew correction, which ADR-006 explicitly deferred to v0.3.
  Either amend ADR-006 with a superseding note or revert the
  feature. Decide before v0.1 tag.
- **Fixture generation pipeline.** Phase 4 M5 introduced JLD2
  fixtures under `packages/INLASPDE.jl/test/oracle/fixtures/` but
  the R generation scripts in `scripts/generate-fixtures/spde/`
  are not exercised by CI. This is the single largest untracked
  correctness risk identified by the status review.
- **Missing Phase 2 components.** `Leroux`, `BYM` (non-reparameterised
  form), `Seasonal`, and `Generic0/1` at the LGM component layer
  have plan entries but no source — the merge does not close these
  items; they are Phase B in the replan.

### References

- `ROADMAP.md` — phase numbering to be revised after merge.
- `plans/initial-commits.md` — branch protection rules §.
- ADR-006 — simplified Laplace deferral, now in tension with
  `85db314`.

---

## ADR-015: `LGMFormula.jl` and `GMRFsPardiso.jl` both cut from v0.1; deferred to v0.2

Status: Accepted
Date: 2026-04-26

### Context

Phase E3 of the v0.1 replan ([`replan-2026-04.md`](replan-2026-04.md))
flagged both sub-packages as cuttable, with explicit cut criteria:

- `LGMFormula.jl`: cut if a usable migration doc exists for users
  coming from R-INLA's `f(...)` syntax.
- `GMRFsPardiso.jl`: cut if comparative benchmark on a 10⁵-vertex
  SPDE Q does not beat CHOLMOD by ≥ 30%.

By 2026-04-26 the four `src/`-bearing packages are GA-ready (Phase E1
Aqua + JET pass, Phase E2 docs site live), and the question is whether
to delay tagging v0.1.0 to ship two more sub-packages.

### Decision

**Both cut from v0.1; defer to v0.2.**

`LGMFormula.jl`: the explicit `LatentGaussianModel(ℓ, components, A)`
constructor is a small surface and the
[Getting started](../docs/src/getting-started.md) and three vignettes
already cover the migration path from R-INLA. The macro adds zero
numerical capability — purely ergonomics — and macro design that's
robust against `StatsModels` schema-application edge cases is at
least 3 weeks of work that would block the v0.1 tag. Re-add as a
sub-package in v0.2 with the M1+M2 milestones from
[packages/LGMFormula.jl/plans/plan.md](../packages/LGMFormula.jl/plans/plan.md).

`GMRFsPardiso.jl`: Pardiso.jl upstream has had periods of disrepair
([packages/GMRFsPardiso.jl/plans/plan.md](../packages/GMRFsPardiso.jl/plans/plan.md)
"Risk items"), license-detection plumbing is non-trivial, and the
benchmark gate requires a real 10⁵-vertex run we have not done. CHOLMOD
remains the only backend in v0.1; the `FactorCache` interface in
`GMRFs.jl` is already designed so a Pardiso specialization can land
without an API break. Re-evaluate in v0.2 after a benchmark study.

The two scaffold directories (`packages/LGMFormula.jl/` and
`packages/GMRFsPardiso.jl/`) stay in the repo with their `Project.toml`
and `plans/` intact; they simply do not have `src/` yet and are not in
the v0.1 release manifest.

### Consequences

- v0.1.0 release manifest is exactly four packages: GMRFs.jl,
  LatentGaussianModels.jl, INLASPDE.jl, INLASPDERasters.jl.
- ADR-008 (`@lgm` lives in LGMFormula.jl) is unchanged; the package
  just doesn't ship in v0.1.
- No new public API surface to maintain in v0.1, which keeps the
  registry submission minimal and the deprecation surface for v0.2
  empty.
- Users wanting Pardiso must build a custom `FactorCache` against the
  GMRFs.jl `AbstractFactorCache` interface; this is a power-user path
  and is acceptable for v0.1.

### References

- [packages/LGMFormula.jl/plans/plan.md](../packages/LGMFormula.jl/plans/plan.md)
- [packages/GMRFsPardiso.jl/plans/plan.md](../packages/GMRFsPardiso.jl/plans/plan.md)
- ADR-008, ADR-009 — the "split into a sub-package" pattern.

---

## ADR-016: Simplified-Laplace mean-shift correction (Rue-Martino) added as opt-in `latent_strategy`

Status: Accepted
Date: 2026-04-28

### Context

ADR-006 (amended 2026-04-24) split simplified-Laplace into two pieces:
a density-shape correction on `posterior_marginal_x` (landed as the
`strategy = :simplified_laplace` kwarg in commit `85db314`) and a
mean-shift correction on the latent posterior summary (deferred). The
density correction multiplies each per-θ Gaussian by a Hermite-3 skew
factor `(1 + γ/6 · H₃(s))`, expanded around the **unshifted** Newton
mode `x̂(θ)`. R-INLA's `simplified.laplace` additionally shifts that
mode by `Δx = ½ H⁻¹ Aᵀ (h³ ⊙ σ²_η)`, so for a true match against
R-INLA's posterior mean on skewed likelihoods (Poisson, Binomial,
Gamma, NegBin) we need the mean shift as well.

Investigation prompted by `haavardhvarnes/IntegratedNestedLaplace.jl`
(`laplace_eval`), which ships the same formula. The mean shift is one
multi-RHS sparse triangular solve per integration point; with our
existing `FactorCache` `\` (`packages/GMRFs.jl/src/factorization.jl:74`),
existing closed-form `∇³_η_log_density` for all v0.1 likelihoods, and
the kriging projection idiom from `_latent_skewness`
(`marginals.jl:176`), the implementation is ~50 LoC plus a small wiring
delta in `inla.jl`.

We also evaluated and rejected porting the same reference's Edgeworth
log-marginal-likelihood correction and N=100 importance-sampling
log-marginal correction:

- mlik parity is already established (project memory
  `project_inla_mlik_gap.md`, resolved 2026-04-27 — the residual gap
  was a normalising constant in `Intercept`/`BYM`, not a missing
  higher-order term).
- The Edgeworth term materialises a dense `Σ_η = A H⁻¹ Aᵀ` and is
  `O(n_obs²)` to `O(n_obs³)` — blows the Phase-D 30 s Scotland budget.
  The reference repo's own code computes it but does not use it
  ("Phase 6g uses 3rd-order Taylor instead").
- The IS estimator at fixed N=100 ships no ESS diagnostic, and at this
  scale its Monte-Carlo error (~ 0.1 nat) is comparable to or larger
  than the corrections it claims to make.

### Decision

Add a new `latent_strategy::Symbol` kwarg to `INLA(...)`, accepting
`:gaussian` (default) and `:simplified_laplace`. Setting
`:simplified_laplace` applies the Rue-Martino mean shift to
`INLAResult.x_mean` and `x_var` while leaving `LaplaceResult.mode`
(the Newton fixed point) unchanged — downstream code that operates on
the Newton mode (`_latent_skewness`, sampling, log-marginal) is
deliberately unaffected. The density-shape correction in
`posterior_marginal_x(strategy = :simplified_laplace)` remains
orthogonal: a user can pick either, both, or neither.

The new field is on `INLA(...)`, not a new strategy type, so
third-party `AbstractInferenceStrategy` subtypes (per ADR-011) do not
have to enumerate latent variants.

Default stays `:gaussian` — flipping it is a separate v0.3 decision
gated on the Pennsylvania oracle re-run with the new path active.

The variance correction (R-INLA's third simplified-Laplace term,
`H⁻¹ Aᵀ diag(h⁴) A H⁻¹` per coordinate) and the Edgeworth / IS log-mlik
corrections remain v0.3 / out of scope.

### Consequences

- **Good:** closes the v0.1 gap to R-INLA's posterior *mean* on skewed
  likelihoods. The `:gaussian` and `:simplified_laplace` paths are
  orthogonal at the API level, so opt-in users see the new behaviour
  with zero risk to existing callers.
- **Good:** complements the already-shipped Hermite skew correction
  in `posterior_marginal_x` — together they cover R-INLA's
  `simplified.laplace` mean and shape; only the variance term is
  still owed.
- **Cost:** one multi-RHS sparse triangular solve per integration
  point. Measured at ~0% overhead on the small Pennsylvania-style
  models in the existing oracle suite; the projected upper bound for
  larger SPDE fits (Meuse) is ~5%.
- **Cost:** maintenance of two latent-strategy paths through the
  integration loop. Localised: ~3 lines in `fit(::INLA)` accumulator
  plus the 50-line helper in `simplified_laplace_correction.jl`.
- **Cost:** Pennsylvania oracle fixture needs regeneration to add a
  `bym_mean_sla` field for the matched R-INLA `strategy = "simplified.laplace"`
  output. The existing Gaussian-strategy assertions are untouched.
- **Escape hatch:** `latent_strategy = :gaussian` is exactly the prior
  behaviour, bit-for-bit. Verified by a regression test that asserts
  `‖x_mean_gaussian − x_mean_simplified_laplace‖∞ < 1e-12` on a
  Gaussian-likelihood model (where `h³ ≡ 0`).
- **Defer v0.3:** flipping the default to `:simplified_laplace`,
  adding the variance correction, and any per-marginal full
  `:laplace` strategy.

### References

- ADR-006 (and 2026-04-24 amendment) — original deferral.
- `replan-2026-04.md` "What this replan does not schedule" — entry
  rewritten to keep Edgeworth / variance correction deferred while
  removing the mean shift from the deferred list.
- Rue, Martino, Chopin (2009) §4.2.
- `haavardhvarnes/IntegratedNestedLaplace.jl` `laplace_eval` —
  reference implementation of the mean shift.
- `packages/LatentGaussianModels.jl/src/inference/simplified_laplace_correction.jl`
  — implementation.
- `packages/LatentGaussianModels.jl/test/regression/test_sla_mean_shift.jl`
  — regression coverage (Gaussian collapse, dense formula
  cross-check, BYM2 sum-to-zero preservation).

---

## ADR-017: Projector seam — `AbstractObservationMapping` for joint-likelihood / multi-response models

Status: Proposed
Date: 2026-04-29

Supersedes ADR-005 (which deferred the call to "Phase 5"; that phase
is now Phase G of `replan-2026-04-28.md`, and the call is due).

### Context

The v0.1 `LatentGaussianModel` carries a single `A::AbstractMatrix`
projector mapping the stacked latent vector `x` to the linear
predictor `η = A x`, alongside a single `likelihood::AbstractLikelihood`
applied row-wise to `η`. This is sufficient for every fixture
currently in `test/oracle/`: areal Poisson (Scotland, Pennsylvania),
geostatistical Gaussian (Meuse), all the synthetic singletons.

Phase G unlocks the rest of R-INLA's mainstream coverage —
joint longitudinal-survival, multi-response geostatistics, disease
mapping with multiple sentinels — and every one of these models
violates **both** assumptions simultaneously:

1. There are *several* likelihood blocks, e.g. Gaussian on the
   longitudinal arm and Weibull on the survival arm.
2. Each block has its own row-to-latent mapping `A_k`, and the
   `Copy` component (Phase G scope) shares one block's latent into
   another block's predictor with an estimated scaling β.

R-INLA papers over this with `inla.stack`, which conflates *three*
concerns: observation mapping, data stacking, and effect indexing.
Replicating `inla.stack` verbatim would import its sharpest
ergonomic edge — most R-INLA support questions are stack-related —
while losing the type-stable structural API that ADR-003 committed
us to. We need an alternative that (a) handles the multi-block case
cleanly, (b) keeps the v0.1 single-block users on the existing
constructor with no source change, and (c) gives third-party
components a typed extension point rather than a magic
`hcat`-shaped contract.

The architecture document (`plans/architecture.md` §"The projector
question — open issue") flags this and explicitly defers the
abstract-type promotion to an ADR. The replan's Phase G "Risks"
section names a wrong abstraction here as "the first phase where
[it] will compound across the rest of the v0.2 horizon" and asks
for an outside reviewer before the seam merges. This ADR is that
review prerequisite — the code change does not start before this
record is accepted.

### Decision (proposed)

Promote the projector slot to a dispatchable abstract type
`AbstractObservationMapping <: Any`, owned by
`LatentGaussianModels.jl`. The `LatentGaussianModel` struct's `A`
field becomes `mapping::M <: AbstractObservationMapping`, and the
`likelihood::L <: AbstractLikelihood` field becomes
`likelihoods::Tuple{Vararg{<:AbstractLikelihood}}` keyed by
observation block via the mapping.

Concrete subtypes shipped with the seam:

```julia
abstract type AbstractObservationMapping end

# v0.1 default — wraps the current m.A. Single-block; one likelihood.
struct LinearProjector{A <: AbstractMatrix} <: AbstractObservationMapping
    A::A
end

# Areal / row-aligned shortcut — algebraically a no-op A = I.
struct IdentityMapping <: AbstractObservationMapping
    n::Int
end

# Multi-block joint models — one (A_k, likelihood_k) per block.
# Block boundaries are encoded as a Vector{UnitRange{Int}} keyed
# parallel to `likelihoods`.
struct StackedMapping{T <: Tuple} <: AbstractObservationMapping
    blocks::T                    # Tuple of LinearProjector / IdentityMapping
    rows::Vector{UnitRange{Int}} # observation-index ranges per block
end

# Separable space-time — also serves Phase M.
struct KroneckerMapping{S, T} <: AbstractObservationMapping
    A_space::S
    A_time::T
end
```

Required interface (every `AbstractObservationMapping` implements):

```julia
apply!(η, mapping, x)              # η .= mapping * x, in place
apply_adjoint!(g, mapping, r)      # g .+= mappingᵀ * r, accumulating
nrows(mapping) -> Int              # row count of the implicit matrix
ncols(mapping) -> Int              # column count = length(x)
likelihood_for(mapping, i) -> Int  # which block-index owns row i
                                   # (default returns 1 for single-block)
```

`apply!` and `apply_adjoint!` are the load-bearing inner-Newton
operations — they replace every `m.A * x` and `m.A' * r` site in
`inference/laplace.jl`. Performance contract: an `IdentityMapping`
falls through to a `copyto!`; a `LinearProjector{<:SparseMatrixCSC}`
calls `mul!` on the underlying matrix with no wrapper allocation.
Benchmarked against the v0.1 baseline on `test/oracle/`: target ≤2 %
regression on Scotland BYM2 wall-clock, ≤5 % on Meuse SPDE.

Constructor compatibility:

```julia
# v0.1 form — unchanged; wraps in LinearProjector internally.
LatentGaussianModel(ℓ::AbstractLikelihood, components, A::AbstractMatrix)

# v0.2 multi-likelihood form.
LatentGaussianModel(ℓs::Tuple{Vararg{AbstractLikelihood}}, components,
                    mapping::AbstractObservationMapping)
```

The single-likelihood signature continues to work for **two minor
versions** with a one-line deprecation warning advising the new tuple
form; v0.4 drops it. Existing `test/oracle/` fixtures keep their
constructors unchanged across v0.1 → v0.2.

### Alternatives considered

**A. Carry a tuple of `(likelihood, A)` pairs directly on the
model.** Simpler — no new abstract type, no dispatch. Rejected
because it inlines the abstraction at every call site
(`for (ℓ, A) in pairs ...`) and forecloses on `KroneckerMapping`,
which Phase M needs anyway. The dispatch seam is the cheap insurance
against re-litigating this in 18 months.

**B. Keep `A` and `likelihood` as v0.1, expose multi-block via a new
`MultiLikelihoodModel` type.** Two parallel model types means every
inference strategy, diagnostic, and accessor in
`LatentGaussianModels.jl` and `INLASPDE.jl` ships in two flavours.
Rejected on cost.

**C. Adopt R-INLA's `inla.stack` 1:1.** Already rejected by ADR-003
(structural over procedural API).

### Consequences

**Good.**

- Joint-likelihood models become a struct + a tuple, not a DSL —
  preserves the ADR-003 commitment.
- `Copy` (also Phase G) lands as a `LinearPredictorTerm` that the
  mapping holds, with no further surgery on the model struct.
- Phase M's separable space-time gets `KroneckerMapping` for free.
- The single-likelihood hot path is unchanged: `LinearProjector{<:
  SparseMatrixCSC}` is one indirection that the compiler inlines.
- Third-party components define their own
  `AbstractObservationMapping` (e.g. `LowRankProjector` for
  factor-analytic models) without touching LGM's source.

**Cost.**

- One breaking change in the `LatentGaussianModel` field layout
  (`A` → `mapping`). Mitigated by the constructor compatibility
  shim and the deprecation window.
- Every `m.A` access inside `LatentGaussianModels.jl`,
  `INLASPDE.jl`, and the LGM tests is rewritten to go through
  `apply!` / `apply_adjoint!`. ~30 sites by `grep`. One PR.
- The `nhyperparameters` accounting changes — likelihood θ is now
  a vector-of-tuples rather than a scalar offset. Touches
  `θ_ranges` initialisation in `model.jl:52-58`.
- Inference strategies that materialise `A` densely (none today, but
  `simplified_laplace_correction.jl` does row-slicing) need a path
  through the mapping interface. Tracked as a follow-up sub-task,
  not a blocker for the seam itself.

**Escape hatch.** If the mapping abstraction underperforms in a
benchmarked hot path, drop to the underlying matrix via
`mapping.A` (for `LinearProjector`) and call `mul!` directly —
the wrapper is intentionally thin enough that this is a single-line
opt-out for that call site.

### Phasing — how this lands

The replan asks for multi-likelihood as one PR series and `Copy` as
a follow-up; this ADR keeps that split:

1. **PR series 1 — seam only.** Introduce
   `AbstractObservationMapping`, the four concrete subtypes, the
   compatibility constructor, and the `m.A → m.mapping` rewrite.
   Single-likelihood semantics throughout. Goal: every existing
   oracle test passes unchanged.
2. **PR series 2 — multi-likelihood.** Promote `likelihood` →
   `likelihoods` and route per-block density evaluation through
   `likelihood_for(mapping, i)`. New oracle fixture: a two-block
   Gaussian + Poisson synthetic.
3. **PR series 3 — `Copy` component.** Lands on top of (2) using
   `LinearPredictorTerm` to express shared latents. Fixture:
   Baghfalaki et al. (gated on Phase H survival likelihoods —
   merges as a later vignette PR).

### Open questions for the reviewer

1. Should `IdentityMapping` carry `n::Int` or be dimension-free?
   Carrying `n` keeps `nrows`/`ncols` cheap and lets us assert
   shape at construction; the cost is one `Int` per model.
2. Block ordering invariant: are the rows of `StackedMapping`
   *required* to be contiguous and sorted, or should we allow
   interleaved row indices? Contiguous is simpler; interleaved
   matches R-INLA's `inla.stack` more closely. Recommendation:
   contiguous in v0.2, defer interleaved to a follow-up if a real
   user needs it.
3. Do we expose `apply!`/`apply_adjoint!` as public API, or hide
   them behind `*` / `mul!` overloads only? Public is friendlier
   to third-party components; private keeps us free to tweak the
   signature. Recommendation: public, documented in the package
   `CLAUDE.md`.

### References

- ADR-003 (multi-dispatch primary API) — the commitment this ADR
  preserves under multi-likelihood pressure.
- ADR-005 (projector as field, deferred decision) — superseded.
- `plans/architecture.md` §"The projector question — open issue" —
  the open issue this ADR closes.
- `plans/replan-2026-04-28.md` Phase G — "Multi-likelihood support",
  "AbstractObservationMapping seam", "`Copy` component", and the
  "Risks" subsection asking for outside ADR review.
- R-INLA source: `inla.stack` in [`r-inla/rinla/R/stack.R`](https://github.com/hrue/r-inla/blob/devel/rinla/R/stack.R)
  — the abstraction we are explicitly *not* copying.
- Baghfalaki et al. tutorial — the joint-model fixture target.

---

## ADR-018: Censoring as a likelihood-level feature — `Censoring` enum + per-row vector on the survival likelihood struct

Status: Proposed
Date: 2026-04-29

### Context

Phase H of `plans/replan-2026-04-28.md` ships the survival / time-to-event
likelihood pack — Exponential, Weibull (PH and AFT), LognormalSurv,
GammaSurv, Coxph (piecewise-baseline), WeibullCure — plus the zero-inflated
count families. Every survival family in this pack must distinguish four
observation modes: uncensored event time, right-censored, left-censored,
and interval-censored. R-INLA encodes this via `inla.surv(time, event,
time2)` where `event ∈ {0, 1, 2, 3}` is the censoring code and `time2` is
the upper interval bound.

The replan's Phase H scope (line 208) calls for a "`Censoring` enum
(`:none`, `:right`, `:left`, `:interval`) + per-row vector on the
likelihood struct". This ADR pins down (a) the type and where it lives,
(b) how the boundary times for interval-censoring are carried, (c) the
hot-path contract for `log_density` / `∇_η` / `∇²_η` / `∇³_η` under
censoring, and (d) the upgrade path for non-survival likelihoods that
might want censoring later (censored Gaussian → tobit). The ADR is a
prerequisite for PR1 of the survival pack (Exponential + the censoring
contract); Weibull / Lognormal / Gamma / Coxph land as follow-up PRs that
inherit the contract, not re-litigate it.

### Decision (proposed)

**1. The enum.** Define a value-type enum in `LatentGaussianModels`:

```julia
@enum Censoring NONE RIGHT LEFT INTERVAL
```

Exported. Symbol coercion (`Censoring(:none)`, etc.) is provided as a
convenience for keyword-call sites; storage is always the enum, not a
`Symbol`, so the inner Newton loop sees a single byte per row and the
compiler can unbox the branch.

**2. Per-row vector on the survival likelihood struct.** Each survival
family carries the censoring data as struct fields, *not* on `y`:

```julia
struct ExponentialLikelihood{L, C, V} <: AbstractLikelihood
    link::L              # LogLink default — rate parameterisation
    censoring::C         # Nothing | AbstractVector{Censoring}
    time_hi::V           # Nothing | AbstractVector{<:Real} — INTERVAL upper bounds
end
```

Construction:

```julia
ExponentialLikelihood()                                   # all NONE, fastest path
ExponentialLikelihood(censoring = [NONE, RIGHT, NONE])    # mixed; time_hi unused
ExponentialLikelihood(censoring = [NONE, INTERVAL],
                      time_hi   = [0.0, 5.0])             # interval row uses time_hi[2]
```

The `Nothing` sentinel is load-bearing — it gives a separate dispatch for
the all-uncensored case so the closed-form simple-density path is not
dragged through a per-row switch in the hot path.

**3. The `log_density` contract under censoring.** For a survival family
with hazard `h(t; η, θ)` and survival `S(t; η, θ) = exp(-Λ(t; η, θ))`:

| Censoring        | log p(y_i \| η_i, θ)                                        |
|------------------|------------------------------------------------------------|
| `NONE`           | `log f(t_i) = log h(t_i) − Λ(t_i)`                          |
| `RIGHT`          | `log S(t_i) = −Λ(t_i)`                                      |
| `LEFT`           | `log F(t_i) = log(1 − S(t_i))`                              |
| `INTERVAL`       | `log[S(t_lo) − S(t_hi)] = log[exp(−Λ_lo) − exp(−Λ_hi)]`     |

`y[i]` carries `t_lo` for `INTERVAL` rows; `time_hi[i]` carries `t_hi`. For
`NONE`/`RIGHT`/`LEFT`, `y[i]` is the relevant single time. The `INTERVAL`
log-difference uses `logsubexp(a, b) = a + log1p(-exp(b - a))` for `a > b`
to retain digits; this is a small helper, not a new dependency.

`∇_η`, `∇²_η`, `∇³_η` follow the same per-row dispatch — for each
censoring mode the chain rule against the link function `g(η) = ` rate
(or scale, depending on family) yields a closed form that is no more
expensive than the uncensored case modulo a single `expm1` per
non-`NONE` row. The `Nothing`-censoring dispatch reuses the simple
broadcast path.

**4. Boundary-time storage decision.** Three alternatives considered:

- **A. `time_hi::Union{Nothing, Vector{<:Real}}` on the struct.**
  Adopted. Cheap for the common single-bound case; one extra Vector for
  the interval case; the type-parameterised `Nothing` keeps dispatch
  type-stable. Cost: a tiny amount of dead storage when only some rows
  are `INTERVAL` (`time_hi[i]` for non-`INTERVAL` rows is unread).
- **B. Heterogeneous `y::Vector{Union{T,Tuple{T,T}}}`.** Rejected —
  breaks every existing `y::AbstractVector{<:Real}` contract throughout
  diagnostics, plotting, and the predict/predict_quantile API.
- **C. A `SurvivalOutcome{T}` data type carrying `(t_lo, t_hi, c)` per
  row.** Rejected — forces a dependent type on every survival likelihood
  and on the `y` argument across the public API; the per-row vector
  approach achieves the same density of information with no surface
  change to `y`.

**5. Default for non-censored families.** Gaussian, Poisson, Binomial,
NegBinomial do **not** acquire censoring fields in PR1. The contract
is opt-in per likelihood. A future censored Gaussian (tobit) will follow
the same pattern: add `censoring::Union{Nothing, Vector{Censoring}}`
plus `time_hi` to its struct, dispatch on the `Nothing` field for the
existing fast path. No cross-cutting change to `AbstractLikelihood` is
required; the seam is the per-likelihood struct field, not the abstract
type.

**6. Validation.** At `fit` time, assert
`length(censoring) == length(y)` and `time_hi[i] > y[i]` for every
`INTERVAL` row. These are public-boundary `@assert`s, removed from the
inner Newton loop per the existing hot-path policy in
`packages/LatentGaussianModels.jl/CLAUDE.md`.

**7. Diagnostics under censoring.** `pointwise_log_density` honours the
censoring code per row (i.e., its sum equals `log_density`, regardless
of the censoring mix). `pointwise_cdf` for censored rows is undefined in
v0.1 — PIT for censored observations needs a non-trivial reweighting
(Henderson-Crowther) that is deferred to a v0.2 follow-up; the default
`pointwise_cdf` fallback `throw`s on first call so a user who tries to
compute PIT with censored data gets a clean error.

### Alternatives considered

**A. Censoring as an outer wrapper likelihood —
`CensoredLikelihood{L<:AbstractLikelihood}`.** This keeps every base
likelihood (Gaussian, Exponential, Weibull) censoring-naïve and adds
censoring through composition. Rejected because the survival families
*already* use the survival function `S` directly in their derivative
formulas — a wrapper would have to re-derive `∇²` and `∇³` of
`log[S − S]` numerically, blowing up cost for `INTERVAL`. Composition is
attractive in the abstract but pays a real performance cost in the inner
Newton loop, and does not naturally extend to censoring upper-bound
storage.

**B. Censoring as part of the `LikelihoodMapping` (ADR-017) instead of
on the likelihood struct.** Rejected — `LikelihoodMapping` is the
*routing* layer (which row goes to which likelihood block), not the
*outcome model* layer. Censoring is intrinsic to the outcome model
(survival vs. event) and naturally co-locates with the family-specific
`log_density` formula. Putting it on the mapping would force every
joint-likelihood model to carry a redundant censoring vector even for
its non-survival blocks.

**C. `Vector{Symbol}` instead of `Vector{Censoring}`.** Rejected on
performance — `Symbol` is a pointer, the per-row branch through `===`
is heavier than an enum compare; on a 5e4-observation Tonsil fit the
branch dominates the inner-loop cost relative to the closed-form
hazard arithmetic. The `Censoring(:none)` symbol coercion at
construction is the convenience users wanted; the storage is the
enum.

### Consequences

**Good.**

- Survival families share one censoring contract; PR1 (Exponential)
  shakes it down end-to-end before Weibull adds a shape hyperparameter
  and Coxph adds a piecewise baseline.
- The `Nothing`-censoring fast path means uncensored exponential fits
  pay zero overhead from the censoring infrastructure — same throughput
  as a hypothetical "no-censoring-ever" implementation.
- Future censored-Gaussian (tobit) lands as a struct-field addition, not
  an `AbstractLikelihood` contract change.
- Aligns with R-INLA's `inla.surv` on the data side (lo/hi/code triple)
  while keeping the Julia API typed and dispatch-friendly.

**Cost.**

- Each survival family must implement the four-way per-row dispatch in
  `log_density`, `∇_η`, `∇²_η`, `∇³_η`. ~40 LoC per family. Mitigated by
  a shared internal helper module
  (`packages/LatentGaussianModels.jl/src/likelihoods/survival/_censoring.jl`)
  exposing the censoring branches as inlineable building blocks.
- `time_hi` storage is wasted for non-`INTERVAL` rows in mixed datasets.
  In practice interval censoring is rare (Tonsil: 0%, Leuk: 0%, Gambia:
  0%); the typical fit pays one extra `Vector` of length n_obs that goes
  unread. Acceptable.
- Validation assertions at the public boundary need clear, R-INLA-style
  error messages — a row with `INTERVAL` and `time_hi[i] ≤ y[i]` is the
  most common user mistake and the message must call this out.

**Escape hatch.** If a user needs a censoring mode the enum doesn't cover
(e.g. Type-II progressive censoring), they can implement their own
`AbstractLikelihood` subtype that handles its own outcome encoding
without using `Censoring` — the enum is convention, not a load-bearing
hook in the abstract type's contract. The shared helper module is
internal API.

### Phasing — how this lands

1. **PR1 — `Censoring` + `ExponentialLikelihood` + oracle.** Lands the
   enum in `LatentGaussianModels`, the `_censoring.jl` helper, the
   Exponential family with full censoring support, an R-INLA oracle
   fixture (synthetic exponential survival, Phase H scope) under
   `scripts/generate-fixtures/lgm/exp_survival/`, and tier-1 / tier-2
   tests. **This ADR's deliverable.**
2. **PR2 — `WeibullLikelihood` (PH parameterisation).** Reuses the
   censoring helper. Adds the shape hyperparameter and PC prior on it.
3. **PR3 — `LognormalSurvLikelihood`, `GammaSurvLikelihood`,
   `WeibullCureLikelihood`.** Three families, one PR; the censoring
   pattern is now mechanical.
4. **PR4 — `CoxphLikelihood`** with piecewise-constant baseline and
   stratification. Most complex; censoring contract carried over.
5. **PR5 — Zero-inflated families.** Independent of censoring; gated on
   the multi-likelihood seam (ADR-017) only. Lands separately.
6. **PR6 — `PCAlphaW` hyperprior.** Sørbye-Rue 2017. Pure prior
   addition; independent of the censoring contract.

### Open questions

1. **Left-truncation.** R-INLA's `inla.surv` carries a `truncation`
   column for left-truncated data (e.g., delayed entry into a cohort).
   This is **not** in the v0.1 censoring enum. Recommendation: add
   `truncation::Union{Nothing, Vector{<:Real}}` on the survival likelihood
   in a v0.2 PR if a user reports needing it; the contract here doesn't
   foreclose on it.
2. **`pointwise_cdf` under censoring.** The Henderson-Crowther PIT for
   censored observations is non-trivial. Recommendation: deferred to
   v0.2; the v0.1 default `throw` is the right ergonomic for now.
3. **`@enum` vs. `BitFlags`.** A `@enum` cannot be combined (e.g.,
   right-censored *and* left-truncated). If we need the cross product
   later we'll move to `@bitflag`. Recommendation: stick with `@enum`
   in v0.1; revisit in v0.2 alongside left-truncation.

### References

- `plans/replan-2026-04-28.md` Phase H, line 208 — censoring enum scope.
- ADR-017 — multi-likelihood seam; Phase H survival families plug into
  the same `LikelihoodMapping` story but the censoring contract is
  orthogonal.
- ADR-006 (and 2026-04-24 amendment) — sets the precedent of opt-in
  per-likelihood extensions to the inner-Newton path
  (`∇³_η_log_density` was added the same way).
- Klein & Moeschberger (2003), *Survival Analysis*, §3.5 — the four
  censoring modes and their likelihoods.
- R-INLA `inla.surv` source —
  [`r-inla/rinla/R/inla.surv.R`](https://github.com/hrue/r-inla/blob/devel/rinla/R/inla.surv.R).
- `packages/LatentGaussianModels.jl/CLAUDE.md` — likelihood contract +
  hot-path policy this ADR extends.

---

## ADR-019: Zero-inflated count families — three R-INLA parameterisations × three base distributions

Status: Proposed
Date: 2026-05-01

### Context

ADR-018 PR5 ships the zero-inflated count families promised in Phase H of
`plans/replan-2026-04-28.md`. R-INLA exposes three parameterisations
(`zeroinflatedX0`, `zeroinflatedX1`, `zeroinflatedX2`) over three count
families (Poisson, Binomial, NegativeBinomial), giving nine concrete
likelihoods. They differ in (a) how the zero-inflation probability `π`
relates to the count distribution's mean and (b) the hyperparameter scale.
Three ADR questions need pinning before nine likelihoods land:

1. **Which parameterisation is which?** R-INLA's documentation is sparse;
   the source-of-truth is `r-inla/inlaprog/src/likelihood.c`. We need to
   commit to a single canonical mapping so future debugging against
   R-INLA fixtures isn't a guessing game.
2. **One file per family, or one per parameterisation?** Nine likelihoods
   is enough that the structural choice matters for both readability and
   the AbstractLikelihood contract.
3. **Which gradients are closed-form?** `∇³_η_log_density` matters only
   for the simplified-Laplace correction (ADR-006 amendment). Closed-form
   gradients across all 9 × 3 = 27 method positions is unnecessary.

### Decision (proposed)

**1. Parameterisations — canonical mapping.** Three families × three
suffixes; the suffix names the parameterisation, *not* the base
distribution. Across all nine, `θ` carries `[log(size)?, zi_scalar]`:

| Suffix | Name                    | π_i formula                | Count component on `y > 0`           |
|--------|-------------------------|----------------------------|--------------------------------------|
| `0`    | hurdle                  | `logit(π) = θ` (constant)  | base distribution truncated at 0     |
| `1`    | standard mixture        | `logit(π) = θ` (constant)  | base distribution (zero allowed)     |
| `2`    | intensity-modulated     | `π_i = 1 - q_i^α`, `θ = log α` | base distribution (zero allowed) |

`q_i` is family-specific: `μ_i / (1 + μ_i)` for Poisson and
NegativeBinomial (with `μ_i = E_i · exp(η_i)`); `sigmoid(η_i)` for
Binomial. ZINB carries an extra `θ[1] = log(size)` for overdispersion;
the zi scalar is `θ[2]`. ZIP and ZIB carry only the zi scalar (1
hyperparameter total).

This mapping matches R-INLA's `family = "zeroinflated{poisson,binomial,
nbinomial}{0,1,2}"` byte-for-byte.

**2. File layout — one file per family, three structs each.** All three
parameterisations of a family share the same `y > 0` count component
arithmetic; bundling them in a single file lets the shared expressions
appear once in comments and the differences live in adjacent functions.

```
src/likelihoods/zero_inflated/
    _helpers.jl        # logsumexp2, only
    poisson.jl         # ZIP0/1/2 — three structs, three log_density,
                       #   three ∇_η, three ∇²_η, plus ZIP1 ∇³_η
    binomial.jl        # ZIB0/1/2 — same shape; ZIB1 also closed-form ∇³
    negbinomial.jl     # ZINB0/1/2 — same shape; ∇³ falls back to FD
```

Each struct is a separate `<: AbstractLikelihood` so dispatch picks the
right closed form without runtime branching on `family`. The
constructor signature mirrors the plain count family:

- ZIP / ZINB: `T(; link = LogLink(), E = nothing,
  hyperprior_size = GammaPrecision(1.0, 0.1),  # ZINB only
  hyperprior_zi = GaussianPrior(0.0, 1.0))`
- ZIB: `T(n_trials; link = LogitLink(),
  hyperprior = GaussianPrior(0.0, 1.0))`

ZIP/ZINB enforce `LogLink`; ZIB enforces `LogitLink`. Other links throw
in the constructor (matches the plain-NB / plain-Bin policy already in
the package).

**3. Default hyperpriors — match R-INLA verbatim.**

- `θ = logit(π)` (types 0/1): `gaussian(mean = 0, prec = 1)` on the
  internal scale, encoded as `GaussianPrior(0.0, 1.0)` (the new
  R-INLA-equivalent prior added in PR5 step A).
- `θ = log(α)` (type 2): same `gaussian(0, 1)`.
- `θ[1] = log(size)` (ZINB): `loggamma(1, 0.1)`, encoded as the
  existing `GammaPrecision(1.0, 0.1)`. Identical to plain NB.

**4. Gradient closure — closed-form everywhere except `∇³_η` on types
0/2.** `∇_η_log_density` and `∇²_η_log_density` are closed-form for all
nine likelihoods. `∇³_η_log_density` is closed-form for **type 1 only**
(both ZIP1 and ZIB1; ZINB1 falls back via the abstract default). The
type-1 simplified-Laplace correction is the most common
zero-inflated use case in disease-mapping; types 0 and 2 fall back to
the AbstractLikelihood FD default (acceptable since ADR-006's amendment
only requires closed-form `∇³` where it materially affects the inner
hot path).

Gradient derivations (recorded in source comments, not the ADR):

- **Type 0 (hurdle).** `y = 0` branch's η-derivative vanishes because
  `log p = log π` is `η`-independent. `y > 0` derivatives need a
  truncation correction `-K = -∂η log(1 - P_count(0))`.
- **Type 1 (standard mixture).** `y > 0` reduces to plain count family.
  `y = 0` uses a posterior weight `w = (1-π)·P_count(0) /
  (π + (1-π)·P_count(0))`, computed via `logsumexp2` (the only shared
  helper). All three derivatives close cleanly:
  `∇_η = -μ·w` (ZIP1), `-n·p·w` (ZIB1), `-s·μ·w/M` (ZINB1) and similar
  for higher orders.
- **Type 2 (intensity-modulated).** `y > 0` adds `α/(1+μ)` to the plain
  count gradient (ZIP/ZINB) or `α(1-p) - n·p` (ZIB1). `y = 0` is
  `log f` with `f = 1 - q^α·D`, `D = 1 - P_count(0)`; ∂²η computed via
  the quotient rule `(∂²f·f - (∂f)²)/f²`.

**5. Hyperprior split for ZINB.** Two hyperparameters → two prior fields.
The struct carries `hyperprior_size::P1` and `hyperprior_zi::P2`
separately so each can be tuned independently via R-INLA-style kwargs:
`ZeroInflatedNegativeBinomialLikelihood1(; hyperprior_size =
PCPrecision(1.0, 0.01), hyperprior_zi = GaussianPrior(0.0, 0.5))`.

### Consequences

- 27 closed-form methods across 9 likelihoods, ≈800 LoC. Validated to
  ~1e-9 against FiniteDiff on a mix-of-zeros, mix-of-positives, and
  large-count test grid (`test/regression/test_zero_inflated.jl`).
- Type-2 ZINB `y = 0` ∂²η has the most algebraically dense closed form
  (the `∂A` term mixes both `pn0` and `(1-μ)`). FD validation is the
  primary correctness check; the comment block in `negbinomial.jl`
  records the derivation step-by-step so future audits can reproduce it.
- Adding a fourth parameterisation (R-INLA does not currently support
  one, but ZIB has been discussed) is mechanical: drop in a new
  `ZeroInflatedXLikelihood3` with its own `log_density` and gradients.
  No abstract-type change.
- The simplified-Laplace correction (ADR-006 amendment) gains the
  type-1 families immediately; types 0 and 2 will fall back to the
  classical Gaussian approximation at the inner Laplace step. This is
  acceptable for v0.1 — disease-mapping fits use type 1 in the vast
  majority of published applications.

### Phasing — how this lands

PR5 of ADR-018 is the single PR landing all nine likelihoods plus
regression tests. Oracle fixture (synthetic ZIP1 vs R-INLA) lands in a
follow-up PR per the replan; tier-1 regression tests are sufficient to
unblock further survival/zero-inflated documentation work.

### Open questions

1. **R-INLA's `quantile` parameter on type 2.** R-INLA exposes a
   `quantile` kwarg that re-parameterises α via the prior expected
   probability `P(y = 0)` at a chosen quantile of the linear predictor.
   This is a *user-facing convenience* that gets translated to a prior
   on `θ = log α`. Recommendation: skip in v0.1; users can set the
   prior on `log α` directly via `hyperprior_zi`. Add in v0.2 if there
   is demand.
2. **Closed-form ∇³ for ZINB1.** The ZINB1 `y = 0` posterior weight
   structure is the same as ZIP1 / ZIB1. The ∇³ derivation is
   mechanically the same shape but algebraically dense (extra `s` and
   `M` factors). Recommendation: deferred until simplified-Laplace
   correction performance on ZINB1 becomes a documented bottleneck.
3. **Predictive PIT under zero-inflation.** `pointwise_cdf` for ZI
   families needs the mixture CDF, which has a discrete jump at zero.
   Recommendation: deferred to v0.2; the `pointwise_log_density`
   methods are sufficient for DIC / WAIC / log marginal likelihood
   diagnostics shipping in v0.1.

### References

- `plans/replan-2026-04-28.md` Phase H — zero-inflated families scope.
- ADR-006 (and 2026-04-24 amendment) — `∇³_η_log_density` opt-in.
- ADR-017 — multi-likelihood seam; ZI families plug into the same
  `LikelihoodMapping` story.
- ADR-018 — PR5 of the survival pack is this work.
- R-INLA likelihood source —
  [`r-inla/inlaprog/src/likelihood.c`](https://github.com/hrue/r-inla/blob/devel/inlaprog/src/likelihood.c).
- Lambert (1992), *Technometrics* 34(1) — original ZIP1 standard
  mixture parameterisation.
- Heilbron (1994), *Biometrical Journal* 36(5) — hurdle (type 0)
  parameterisation.

---

## ADR-020: Drop Julia 1.10 LTS support — Julia 1.12 is the minimum supported version

Status: Accepted
Date: 2026-05-01

### Context

The replan-2026-04-28 Phase F plan committed the v0.1.0 / v0.1.1 release
to Julia 1.10 LTS + current stable, on the assumption that LTS coverage
broadens the user base. In practice this project has zero LTS-pinned
users (no downstream issues filed against 1.10, no LTS-only deps), and
maintaining the LTS lane has carried real cost:

- The CI matrix doubles for the four core packages (LTS × current,
  Linux × macOS × Windows), and the LTS includes are the slow tail.
- Several recent commits used 1.11+ syntax (`@kwdef` improvements,
  `Returns`, `Splat`) that needed manual back-porting for the LTS lane.
- Julia 1.12 is the current stable — the pragmatic floor — and is what
  the local development environment, the benchmark machine, and the
  authors' editors all run on.

### Decision

**Julia 1.12 is the minimum supported version across the entire monorepo.**
This applies to every package's `[compat] julia` field and every CI
matrix lane.

- All `Project.toml` `[compat] julia` entries are bumped to `"1.12"`.
- The `.github/workflows/test.yml` matrix drops `'1.10'` and keeps `'1'`
  (which resolves to 1.12.x today and to whatever stable is when CI
  runs in the future).
- Cross-platform coverage (macOS, Windows) tracks `'1'` rather than the
  former 1.10 pin.
- The replan-2026-04-28 acceptance criterion ("`Pkg.add(\"INLA\")` from
  a fresh depot resolves on Julia 1.10 LTS and current stable") is
  superseded — only current stable.

### Consequences

- New language features ≥ 1.11 (e.g. `Returns`, `Splat`, public marker
  in `module`) are now usable without conditional shims.
- Smaller CI matrix → faster PR feedback, lower spend.
- Users on 1.10 LTS who try `Pkg.add("INLA")` will get a clean compat
  error from Pkg.resolve, not a broken install.
- AutoMerge on the General registry should be untroubled — `julia =
  "1.12"` is a valid lower bound for Pkg.

### References

- `plans/replan-2026-04-28.md` Phase F — superseded acceptance criterion.
- ADR-001 — package split context (which versioning policy applies to all
  four core packages uniformly).

---

## ADR-021: `Copy` component — scaling β lives on the receiving likelihood, not on the projection mapping

Status: Accepted
Date: 2026-05-02

### Context

ADR-017 closed the projector seam question by promoting the model's
`A` slot to a dispatchable `AbstractObservationMapping`. Phase G PR
series 1 and 2 (multi-likelihood + the seam) have since merged. PR
series 3 — the `Copy` component, the single largest blocker on the
joint longitudinal-survival vignette — needs to express
`η_target[i] += β · x_source[k(i)]`, where `x_source` is some other
component's latent slice and `β` is an estimated hyperparameter.

The plumbing question β raises is *where in the architecture does β
live*. Three places it could live:

A. **θ-aware `apply!`.** Extend the seam contract from
   `apply!(η, mapping, x)` to `apply!(η, mapping, x, θ)`. The Copy
   mapping reads β from θ via a stored hyperparameter index and
   applies the scaling in its `apply!` method.

B. **Mutable scaling on the mapping struct.** Add a
   `set_scaling!(mapping, θ)` method called once per θ-iteration before
   `apply!` runs; the Copy mapping caches the current β internally.

C. **β on the receiving likelihood.** β is a hyperparameter of the
   likelihood that receives the copied effect (not of the source
   component or of the mapping). The mapping stays β-free; after the
   main `apply!` runs, the receiving likelihood adds its own
   `β · x_source[k(i)]` contribution to η via a new hook.

ADR-017's `LinearPredictorTerm` paragraph (line 1036) is silent on
which of these wins; the ADR explicitly leaves Copy's plumbing for
PR series 3 to settle, with the "outside reviewer" prerequisite from
the replan-2026-04-28 Phase G "Risks" subsection.

### Decision

**Option C: β is a hyperparameter of the receiving likelihood.**

The seam contract from ADR-017 stays exactly as written —
`apply!(η, mapping, x)` is θ-free. After the mapping's `apply!`
populates η from x, each likelihood gets a chance to apply its own
post-projection contributions via a new hook on `AbstractLikelihood`:

```julia
# Default no-op — most likelihoods don't have copies.
add_copy_contributions!(η_block, ℓ::AbstractLikelihood,
                        x::AbstractVector, θ_ℓ) = η_block
```

Likelihoods that participate in joint models (the survival likelihoods
from Phase H, primarily) override this to read their `β` slot from
`θ_ℓ` and apply `β * x[component_range]` to their η block. The
`Copy` "component" that user code interacts with is therefore an
ergonomic constructor that:

1. registers no new entry in the latent vector (no extra columns in
   x; the source component's latent is the only copy);
2. registers no new precision-matrix block;
3. records `(target_block, source_component_index, β_prior)` on the
   receiving likelihood when the model is constructed;
4. expands `nhyperparameters(ℓ)` by 1 (or more, for vector β) so the
   inference loop allocates β alongside the likelihood's other
   hyperparameters.

In effect, "Copy" is sugar over a likelihood-side feature, not a new
latent component class. The latent component listing in the model
struct is unchanged from PR series 2.

### Alternatives considered

**A. θ-aware `apply!`.** Cleaner stylistically — the mapping owns the
projection completely, including any θ-dependent scaling — but
breaks the ADR-017 contract that all ~30 `m.A`-equivalent sites in
`inference/laplace.jl` and `INLASPDERasters.jl` were rewritten
against ten weeks ago. Third-party `AbstractObservationMapping`
implementers (the `LowRankProjector` / `KroneckerMapping` audience)
would need to update their mappings to accept θ even when they don't
use it. Rejected on lock-in cost.

**B. Mutable mapping state with `set_scaling!`.** Avoids the contract
break but introduces a hidden imperative call ordering — the inner
Newton loop has to call `set_scaling!` before each `apply!` or the
mapping returns stale η. Surprises third-party implementers, makes
the seam less type-stable. Rejected on ergonomics.

**C** keeps the seam pure, keeps the Copy implementation local to
likelihoods that actually use it, and matches R-INLA's mental model
where β is a property of the receiving formula's `f(..., copy=...)`,
not of the latent term being copied.

### Consequences

**Good.**

- ADR-017's `apply!` / `apply_adjoke!` contract is unchanged —
  third-party `AbstractObservationMapping` implementers don't have
  to retrofit anything. The 30 rewritten call sites stay rewritten.
- Copy's hyperparameter accounting flows through the existing
  `likelihood_θ_ranges` machinery from PR series 2 — no parallel
  routing. β is just another likelihood hyperparameter.
- The receiving likelihood's `log_density` / `∇_η_log_density` /
  `∇²_η_log_density` methods need no changes: they already operate
  on η. The hook fires before they do; everything downstream is the
  same.
- The Copy component's user-visible constructor stays intact — users
  don't need to know β lives on the likelihood; they pass
  `Copy(target_component; β_prior)` and the model constructor wires
  it through.

**Cost.**

- One new abstract method on `AbstractLikelihood`:
  `add_copy_contributions!`. Every concrete likelihood inherits the
  default no-op (single-line method). The ones that opt in are the
  survival likelihoods (Weibull, Exponential, log-normal, gamma)
  plus Gaussian on the longitudinal arm — five sites at most.
- The model constructor has to thread β-prior info from the
  user-facing `Copy(...)` call onto the right likelihood's
  `θ_ranges`. One pass over `m.likelihoods` at construction.
- The β-source association (`β` on this likelihood reads from the
  random-intercept *of that other component*) needs a stored index
  on the receiving likelihood. Adds one field per opt-in likelihood.

**Escape hatch.** If a future use case needs a Copy whose receiving
*term* isn't a likelihood (e.g. a copy that targets a fixed-effect
slot, not an observation block), the alternative B path is still
available — `set_scaling!` becomes the second mechanism without
disturbing the first.

### Phasing — how this lands

1. **PR-3a — likelihood hook.** Add the no-op
   `add_copy_contributions!(η, ℓ, x, θ)` default and the source-index
   storage on the abstract likelihood interface. Every existing
   concrete likelihood gets the default; oracle suite stays green.
2. **PR-3b — `Copy` ergonomic constructor.** Add the user-visible
   `Copy(target, …)` constructor that wires β-prior onto the receiving
   likelihood. Add a closed-form regression test (β = 1.0 fixed
   should reproduce the unscaled-share oracle result).
3. **PR-3c — Baghfalaki vignette.** Joint Gaussian + Weibull with a
   shared subject-specific random intercept and a Copy-scaled
   contribution into the survival linear predictor. Lands as the
   final Phase G PR; oracle fixture in
   `packages/LatentGaussianModels.jl/test/oracle/fixtures/baghfalaki.jld2`.

### References

- ADR-017 — projector seam decision; this ADR closes the β-plumbing
  question deferred there.
- ADR-018 — censoring as a per-row vector on the survival likelihood
  struct; the same likelihood-side feature pattern that this ADR
  follows for β.
- `plans/replan-2026-04-28.md` Phase G — "Joint-models scaffolding";
  Risks subsection (outside-reviewer prerequisite).
- R-INLA `f(..., copy=..., hyper=list(beta = ...))` — the
  conceptual model this ADR matches.
- Baghfalaki, T., Esfandyari, S. & Nazari, V. (2024). *A Bayesian
  joint modelling of longitudinal and time-to-event data using
  INLA: A tutorial.* — the joint model this ADR's PR-3c vignette
  reproduces.

---

## ADR-022: `IIDND{N}` parameterisation — separable (`log τ_i, atanh ρ_{ij}`) by default, Wishart/InvWishart on the joint precision as alternative

Status: Accepted
Date: 2026-05-02

### Context

Phase I-A opens the multivariate-IID work: the `IID2D` and `IID3D`
families that R-INLA exposes as `model = "2diid"` / `"iid3d"`, used in
joint longitudinal-survival random effects, paired-areal disease
mapping, and bivariate meta-analysis. Before any code lands, the
parameterisation has to be locked because it touches the public kwargs,
the prior interface, the Hessian-at-θ̂ stability, and the matching of
R-INLA's defaults — all four of which are difficult to walk back later.

The component sits on `n × N` latent slots with joint precision
`Λ ∈ ℝ^{N × N}` (the GMRF block is `Λ ⊗ I_n`, so the per-replicate
precision is `Λ` and the cross-replicate structure is independence).
What needs deciding is *how the user supplies Λ*, since that drives the
hyperparameter vector θ_c, the prior interface, and the user-facing
parameters that get reported in the summary.

Three parameterisations were on the table:

A. **Joint Λ with Wishart/InvWishart prior.** θ_c stores the
   `N(N+1)/2` distinct entries of Λ on a Cholesky scale; the prior is a
   single `Wishart(r, V)` (or `InvWishart`). Matches R-INLA's `iid2d`
   model code's default. Requires a new `AbstractJointHyperPrior`-style
   abstract type because the existing `AbstractHyperPrior` is
   documented as scalar-only (`packages/LatentGaussianModels.jl/src/priors/abstract.jl`
   line 18: "Multi-dimensional priors … live in INLASPDE.jl because
   they are inherently coupled").

B. **Marginal precisions × correlations (separable).**
   - For `N = 2`: θ_c = `(log τ_1, log τ_2, atanh ρ)` — three scalars,
     each carrying its own `AbstractHyperPrior`. Matches R-INLA's
     `2diid` model code (the alternate `f(., model="2diid", ...)` form).
   - For `N ≥ 3`: θ_c stores the strictly-lower-triangular Cholesky
     factor of the correlation matrix on a tangent-space scale (Lewandowski-Kurowicka-Joe
     2009 / Stan's `cholesky_factor_corr`). Marginal precisions stay
     scalar.

C. **Marginal precisions × correlation matrix on a sphere
   (Lewandowski).** Strictly more complex than (B); the only material
   difference is the prior shape, and R-INLA's defaults aren't
   expressible cleanly in this form.

### Decision

**Adopt (B) as the default parameterisation. Provide (A) — Wishart /
InvWishart on the joint Λ — as an explicit alternative when the user
supplies one as the `hyperprior` kwarg.**

For `N = 2`:

```julia
IIDND(n, 2;
      hyperprior_precs = (PCPrecision(), PCPrecision()),
      hyperprior_corr  = PCCor0(U = 0.5, α = 0.5))
```

with internal-scale `θ_c = (log τ_1, log τ_2, atanh ρ)`. The
user-facing summary reports `(τ_1, τ_2, ρ)`.

For `N ≥ 3`:

```julia
IIDND(n, 3;
      hyperprior_precs = ntuple(_ -> PCPrecision(), 3),
      hyperprior_corrs = ntuple(_ -> PCCor0(U = 0.5, α = 0.5), 3))
```

with internal-scale θ_c packing the three log-precisions followed by
the `N(N-1)/2 = 3` `atanh ρ_{ij}` entries (i < j); the on-disk Cholesky
factor of the correlation matrix is reconstructed from the `atanh ρ`
entries via the Lewandowski-Kurowicka-Joe stick-breaking step. (The
`atanh-of-each-pairwise-corr` parameterisation is *not* injective onto
positive-definite correlation matrices for N ≥ 4 in general; ADR-022
locks IIDND to N ≤ 3, with N ≥ 4 deferred to a successor ADR if the
need arises. R-INLA's `iid3d` model also stops at N = 3.)

For the Wishart alternative:

```julia
IIDND(n, N; hyperprior = Wishart(r = N + 1, V = Matrix(I, N, N)))
```

θ_c packs the lower-triangular Cholesky factor of Λ; internal scale is
unconstrained `ℝ^{N(N+1)/2}` via the log-Cholesky map (positive
diagonal entries are stored as `log L_{ii}`). This needs a new
`AbstractJointHyperPrior` abstract type and a single
`log_prior_density(prior, θ_block)` method on it; the existing scalar
`AbstractHyperPrior` machinery is left untouched. Wishart and
InvWishart are the only initial concrete subtypes.

The `IIDND` struct dispatches on whichever kwarg path the user took
(separable vs joint) via two distinct `IIDND` types — `IIDND_Sep{N}`
and `IIDND_Joint{N}` — under a single `AbstractIIDND` umbrella. The
public constructor `IIDND(n, N; ...)` selects the right concrete type
by inspecting the kwargs; users don't see the dispatch.

### Why (B) over (A) as default

1. **Matches R-INLA's `2diid` defaults bit-for-bit.** R-INLA's
   `2diid` model uses `loggamma + atanh-ρ-Gaussian` defaults; the
   reference implementation `IntegratedNestedLaplace.jl`'s
   `BivariateIIDModel` does the same. Defaulting to (A) — Wishart on Λ
   — would silently diverge from R-INLA's most-used multivariate-IID
   path (R-INLA's docs list `2diid` as the recommended form when the
   user has scalar precisions + correlation in mind, which is the
   common joint-models case).
2. **Fits the existing `AbstractHyperPrior` infra without a new
   abstract type.** Each of the three (or six, for N=3) hyperparameter
   slots is scalar and gets its own existing prior class. Wishart/
   InvWishart are inherently coupled and *do* need new infra; option
   (B) lets that infra arrive only when the user opts in.
3. **PC priors compose naturally.** `PCCor0` on the correlation slot
   (reference at ρ = 0, penalising departures from independence —
   matches R-INLA's `pc.cor0`), `PCPrecision` on each marginal
   precision — the already-shipped PC priors. The Wishart path doesn't
   have a PC analogue in the literature. (Note: the early drafts of
   this ADR called the prior `PCCor1`; corrected before code landed
   because R-INLA's `pc.cor1` reserves that name for the
   reference-at-ρ=1 prior used by AR(1)'s lag-1 correlation.)
4. **Hessian-at-θ̂ stability is well-understood.** R-INLA's published
   stability results for `2diid` carry over directly. The Cholesky
   parameterisation of the joint Λ has known degenerate-Hessian
   pathologies on the diagonal (the `log L_{ii}` entries) when Λ is
   near-singular; defaulting to that path before we have a
   stress-tested implementation is the wrong order.

### Why (A) is still offered

When the user is genuinely working in the precision-matrix mental
model — typically because they have an informative Wishart prior from
prior elicitation, or because they're following a textbook example
that uses Wishart — forcing them through the separable path is a
violation of "match R-INLA's user mental model". `iid2d` (Wishart
default) and `2diid` (separable default) coexist in R-INLA precisely
because both audiences exist.

### Why not (C)

The sphere parameterisation is strictly more complex than (B) —
Lewandowski-Kurowicka-Joe is the standard tool for this — and the only
material difference is the prior shape. R-INLA's defaults aren't
expressible in (C) without a Jacobian compensation that the user
shouldn't have to think about. Rejected on lock-in cost vs zero
quality benefit.

### Consequences

**Good.**

- `IID2D` is the highest-leverage Phase I-A target, and the separable
  parameterisation makes it (a) bit-for-bit comparable to R-INLA's
  `2diid` oracle, (b) ergonomic for the joint-longitudinal-survival
  use case where the user thinks in terms of "subject-specific
  random intercept and slope, with a correlation between them".
- The Wishart alternative path is opt-in, so the `AbstractJointHyperPrior`
  infra only ships when there's a real caller for it. PR sequencing
  can defer the Wishart path to its own PR.
- PC priors stay primary throughout the package; no backslide on the
  R-INLA defaults-parity track.

**Cost.**

- Two `IIDND` concrete types (`Sep{N}` + `Joint{N}`) instead of one.
  Localised: ~80 LoC each, plus the umbrella abstract type. Both
  share `length`, `nhyperparameters`, `gmrf` defaults via the abstract
  type; only `precision_matrix`, `log_hyperprior`,
  `initial_hyperparameters`, and the public constructor logic differ.
- One new abstract type (`AbstractJointHyperPrior`) plus `Wishart` /
  `InvWishart` concrete subtypes when the Wishart path lands. Three
  new files in `src/priors/`. Doesn't affect any of the existing
  scalar-prior-using components.
- `Cholesky-of-correlation` reconstruction (LKJ stick-breaking) for
  N=3 needs ~30 LoC of arithmetic and a regression test against the
  Stan reference.

**Escape hatch.** If a user case emerges where the separable default
diverges materially from a Wishart default they expected (e.g. they're
porting an R-INLA model that *did* use `iid2d` rather than `2diid`),
they pass `hyperprior = Wishart(...)` and the constructor flips to
the joint path with no further code change.

**Defer.** N ≥ 4 IID is deferred — neither R-INLA nor the joint-models
literature has an active need, and the Lewandowski-Kurowicka-Joe
parameterisation gets fragile at higher N. A successor ADR can lift
this if a real use case appears.

### Phasing — how this lands

1. **PR-1a — `IIDND_Sep{2}` (i.e. `IID2D`).** Add the separable
   constructor, `precision_matrix`, log-prior wiring through the three
   scalar `AbstractHyperPrior` slots, regression tests against a dense
   reference Λ. Add `PCCor0` prior. No Wishart path yet.
2. **PR-1b — `IID3D`.** Same pattern at N=3 with the three-correlation
   LKJ piece. `IID3D` regression test against R-INLA `iid3d` (R-INLA's
   `iid3d` is documented as brittle on small samples; oracle is
   informational, not load-bearing).
3. **PR-1c — Wishart / InvWishart joint path.** Adds
   `AbstractJointHyperPrior`, the two concrete priors, and
   `IIDND_Joint{N}` constructor logic. Lands only when a user case
   asks for it; otherwise sits behind the proposed-status flag.

### References

- ADR-006 — PC priors as the default prior class.
- ADR-021 — recent component ADR; this ADR follows the same shape.
- `plans/replan-2026-04-28.md` Phase I — original IID2D / IID3D scope.
- `plans/phase-i-and-onwards-mighty-emerson.md` §Phase I-A — the
  prereq this ADR closes.
- Simpson, D., Rue, H., Riebler, A., Martins, T. G., & Sørbye, S. H.
  (2017). *Penalising model component complexity: a principled,
  practical approach to constructing priors.* — general PC-prior
  framework underlying `PCCor0` (reference at ρ = 0).
- Sørbye, S. H., & Rue, H. (2017). *Penalised complexity priors for
  stationary autoregressive processes.* — companion construction at
  reference ρ = 1 (R-INLA's `pc.cor1`), used by AR(1) but not by
  IID2D.
- Lewandowski, D., Kurowicka, D., & Joe, H. (2009). *Generating
  random correlation matrices based on vines and extended onion
  method.* — N≥3 Cholesky-of-correlation parameterisation.
- R-INLA `f(., model="2diid", ...)` and `f(., model="iid3d", ...)` —
  the parameterisations this ADR matches.
- `IntegratedNestedLaplace.jl` `BivariateIIDModel` — reference
  implementation using the separable form.

---

## ADR-023: `MEB` and `MEC` measurement-error components — β-via-`Copy` decomposition; non-zero `prior_mean` made load-bearing

Status: Accepted
Date: 2026-05-03

### Context

Phase I-B opens the measurement-error work: R-INLA's `f(w, model="meb", ...)`
(Berkson) and `f(w, model="mec", ...)` (Classical) latent components, used
for errors-in-variables regression in epidemiology, environmental
exposure modelling, and any setting where a covariate is measured with
non-trivial uncertainty (Carroll, Ruppert, Stefanski, Crainiceanu 2006;
Muff, Riebler, Held, Rue, Saner 2015).

Both R-INLA models expose, in the linear predictor, the *scaled* latent
covariate `ν_i = β · x_i` rather than `x_i` itself. Internally the
latent block is the unscaled `x` — a length-`n_unique(w)` Gaussian
vector with prior mean and prior precision determined by the model
flavour and a small set of hyperparameters — and `β` is a per-component
hyperparameter that multiplies the projection from latent to linear
predictor before it lands in `η`. R-INLA registers both models in the
latent-models table alongside `ar1`/`rw1`/`iid` (`models.R`,
`n.required = FALSE`), so the user's mental model is "this is a
component", not "this is a coefficient prior".

The two flavours differ by which way the noise tie runs:

- **MEB (Berkson)**: `x_i = w_i + u_i`, `u_i ~ N(0, (τ_u s_i)⁻¹)`. The
  observed covariate `w` is *fixed* and the latent is `w + u`. The
  latent-block prior is therefore `x ~ N(w, (τ_u D)⁻¹)` with
  `D = diag(s)` after marginalising `u`.
- **MEC (Classical)**: `w_i = x_i + u_i`, `u_i ~ N(0, (τ_u s_i)⁻¹)`,
  with prior `x ~ N(μ_x · 1, (τ_x I)⁻¹)`. The observed `w` is a noisy
  measurement of the latent truth `x`. R-INLA absorbs the Berkson tie
  `w | x ~ N(x, (τ_u D)⁻¹)` into the prior on `x` via Gaussian
  conjugacy: posterior precision becomes `τ_x I + τ_u D`, posterior mean
  becomes `(τ_x I + τ_u D)⁻¹ (τ_x μ_x 1 + τ_u D w)`.

Both flavours fit the existing `AbstractLatentComponent` contract if
two things are true:

1. The component's prior mean is allowed to be non-zero (the optional
   `prior_mean(c, θ)` method documented in `CLAUDE.md` becomes
   load-bearing, where for every other component it has been zero).
2. The β-multiplication of the latent block before it enters η is
   handled somewhere — and we have a choice of where.

`plans/replan-2026-04-28.md` (lines 255–256) originally labelled MEB
as "classical" and MEC as "Berkson". This is the opposite of R-INLA's
convention. The R-INLA LaTeX source
(`inlaprog/doc/latent/{meb,mec}.tex`) is unambiguous: `meb` =
"Berkson Measurement Error", `mec` = "Classical Measurement Error". The
B/C suffix in the model name encodes the error structure, not "Bayesian
vs. classical inference". This ADR locks the correct R-INLA naming;
the same PR that lands this ADR amends the replan in place.

### Decision

**Three coupled choices.**

#### 1. β lives on the *receiving* likelihood via `Copy`, not on the component

Consistent with ADR-021's `Copy` decision. The MEB and MEC components
own only the latent block `x` (with non-zero prior mean and structured
precision); the β-scaling is wired up by the user (or by a constructor
helper) as a `Copy(source_indices; β_prior=...)` on the
`CopyTargetLikelihood` that wraps the receiving observation likelihood.

```julia
# User-facing pattern:
component = MEB(w; scale = s, τ_u_prior = GammaPrecision(1.0, 1.0e-4))
β_copy = Copy(component_range; β_prior = GaussianPrior(1.0, 0.001))
target = CopyTargetLikelihood(GaussianLikelihood(), β_copy)
model = LatentGaussianModel((target,),
                            (Intercept(), component),
                            obs_mapping)
```

The receiving-likelihood placement is the existing infrastructure;
nothing about the inner Newton loop, the η-Jacobian augmentation, or
the hyperparameter accounting changes. The MEB/MEC component itself
contributes zero copies of its own — it is, structurally, an "IID with
non-zero prior mean and per-element precision".

#### 2. Latent layout

Both components store a length-`n_unique(w)` latent block. Observations
with identical `w` (after rounding to the supplied `digits`, matching
R-INLA's `f(w, model = "meb", values = ...)` semantics) share the same
latent slot. The projector from latent to row-space is the natural
"row `i` reads slot `k` where `w[i] = unique_w[k]`" injection — already
expressible as a `LinearProjector` from ADR-017's observation-mapping
seam, no new mapping type required.

`prior_mean(c, θ)` becomes the load-bearing way to expose the non-zero
mean to the inner Newton loop. The CLAUDE.md component-contract entry
already documents `prior_mean` as an optional method with default
`zeros`; this ADR promotes it to "load-bearing for at least two
components" and adds the corresponding hot-path test (the inner Newton
step must read `prior_mean` whenever it is non-zero).

`precision_matrix(c, θ)` is diagonal in both flavours:

- **MEB**: `Q_c = τ_u · D`, `μ_c = w_unique`.
- **MEC**: `Q_c = τ_x I + τ_u D`, `μ_c = Q_c⁻¹ · (τ_x μ_x 1 + τ_u D w_unique)`.

Note the MEC mean depends on θ (through `τ_x`, `τ_u`, `μ_x`), unlike
MEB's θ-constant mean. This is fine — `prior_mean(c, θ)` is dispatched
on θ already; the contract supports it.

#### 3. Hyperparameter list and defaults — bit-for-bit R-INLA

**MEB**:

| slot | name | internal scale | prior | initial | default-fixed |
|---|---|---|---|---|---|
| θ_c[1] | β | identity | `GaussianPrior(1.0, 0.001)` (R-INLA: gaussian, mean=1, prec=0.001) | 1.0 | no |
| θ_c[2] | log τ_u | log | `GammaPrecision(1.0, 1.0e-4)` (R-INLA: loggamma, shape=1, rate=1e-4) | log(1000) ≈ 6.9078 | no |

β lives on the receiving likelihood via `Copy`; θ_c[1] is the Copy's β
slot, *not* the component's. Internally only `log τ_u` belongs to the
component's hyperparameter vector. The "MEB has 2 hyperparameters" line
in R-INLA's docs is a user-facing accounting that includes β; the
Julia accounting is "1 component hyperparameter + 1 β slot via Copy".

**MEC**:

| slot | name | internal scale | prior | initial | default-fixed |
|---|---|---|---|---|---|
| θ_c[1] | β | identity | `GaussianPrior(1.0, 0.001)` | 1.0 | no |
| θ_c[2] | log τ_u | log | `GammaPrecision(1.0, 1.0e-4)` | log(10000) ≈ 9.21 | **yes** |
| θ_c[3] | μ_x | identity | `GaussianPrior(0.0, 1.0e-4)` | 0.0 | **yes** |
| θ_c[4] | log τ_x | log | `GammaPrecision(1.0, 1.0e4)` | -log(10000) ≈ -9.21 | **yes** |

Same β-via-Copy split as MEB. The default-fixed slots (τ_u, μ_x, τ_x)
are R-INLA's "degrades to plain regression unless the user opts in":
τ_u huge ⇒ no measurement error, τ_x tiny ⇒ vague prior on x, μ_x
fixed at 0. The user unfixes whichever slots they want to estimate by
passing `fix_τ_u = false` (etc.) at construction. The Julia-side public
constructor mirrors R-INLA's `f(w, model = "mec", control.fixed =
list(...))` form.

#### Public API

```julia
MEB(w::AbstractVector;
    scale = ones(length(unique(w))),
    digits::Int = 8,
    τ_u_prior::AbstractHyperPrior = GammaPrecision(1.0, 1.0e-4),
    τ_u_init::Real = log(1000.0))
# Returns a `MEB` component (subtype of `AbstractLatentComponent`) plus,
# via the constructor's secondary return, a `Copy` template the user
# attaches to the receiving likelihood.

MEC(w::AbstractVector;
    scale = ones(length(unique(w))),
    digits::Int = 8,
    τ_u_prior = GammaPrecision(1.0, 1.0e-4),
    μ_x_prior = GaussianPrior(0.0, 1.0e-4),
    τ_x_prior = GammaPrecision(1.0, 1.0e4),
    τ_u_init  = log(10000.0),
    μ_x_init  = 0.0,
    τ_x_init  = -log(10000.0),
    fix_τ_u::Bool = true,
    fix_μ_x::Bool = true,
    fix_τ_x::Bool = true)
```

The `digits` kwarg matches R-INLA's `f(...; values = unique(round(w,
digits = ...)))` convention for de-duplicating observed covariate
values into latent slots.

The β slot does not appear in either constructor's kwargs — it lives on
the user-supplied `Copy(...)` attached to the receiving likelihood. To
make this ergonomic, both constructors return a `(component, β_copy_template)`
named tuple where `β_copy_template` is a `Copy` pre-configured with
R-INLA's `gaussian(1, 0.001)` β prior; the user can override by
constructing their own `Copy` against the component's range.

### Why β-via-Copy and not β-on-the-component

1. **Consistency with ADR-021.** That ADR accepted "scaling β lives on
   the receiving likelihood" as the rule. Reopening it for MEB/MEC
   would either fragment the codebase (two β-attachment idioms) or
   force an ADR-021 supersedence. The Copy route is the one we already
   tested.
2. **No change to `add_copy_contributions!` or `_accumulate_copy_jacobian!`.**
   The hot path is already wired for β-times-source-block contributions
   into η. Reusing it for MEB/MEC means zero net new hot-path code
   beyond the component's `precision_matrix` and `prior_mean`.
3. **The "MEB component contributes its own copy" alternative would
   couple component construction to likelihood construction.** Right
   now `LatentGaussianModel` accepts `(likelihoods, components,
   mapping)` independently. A self-copying component would need a
   back-reference from latent index → receiving likelihood, which
   `LatentGaussianModel` does not currently maintain.
4. **R-INLA's posterior of `ν = β·x` is recoverable post-fit as a
   derived quantity from the joint `(x, β)` posterior** — no need to
   carry `ν` as a primary latent. A small accessor in PR-2c
   (`measurement_error_scaled_latent(model, res)`) closes the
   user-facing ergonomics gap.

### Why two distinct components and not a parameterised one

`MEB` and `MEC` differ in:
- prior-mean dependence on θ (constant for MEB, θ-dependent for MEC);
- precision-matrix structure (diagonal `τ_u D` vs `τ_x I + τ_u D`);
- number of hyperparameters (1 vs 3 once β is excluded);
- default-fixed pattern (none for MEB, three of three for MEC).

Folding both into a single `MeasurementError(...; flavour = :berkson|:classical)`
constructor would require the struct's hyperparameter vector to be
parameterised on `flavour`, which forces the field types to be
`Vector` instead of `NTuple` and gives up dispatch granularity for
`precision_matrix`/`prior_mean`. The two-component split is the same
shape as `BYM` vs `BYM2` or `Generic0` vs `Generic1` vs `Generic2` —
distinct components matching distinct R-INLA model strings.

### Why the replan's labels-swapped error matters

Carroll-style measurement-error literature uses the Berkson/Classical
distinction to flag which way the noise tie runs, and the analytical
behaviour differs sharply between them — bias attenuation under
classical error, no-attenuation under Berkson, and entirely different
prior-elicitation requirements. Mislabelling the components in user
documentation would silently push users toward the wrong model for
their data. `plans/replan-2026-04-28.md` lines 255–256 originally
called `MEB(...)` "classical" and `MEC(...)` "Berkson" — the opposite
of R-INLA. The active session-level Phase I-onwards replan
(2026-05-02) inherited the same swap from the parent replan. This ADR
is the single source of truth from this point forward; the committed
replan was corrected in the same PR that landed this ADR, and the
session-level replan picks up the correction via its next sync.

### Consequences

**Good.**

- MEB and MEC ship as standard `AbstractLatentComponent` subtypes with
  no new abstract types, no new prior categories, no Copy-machinery
  changes. The β-attachment idiom is unchanged from ADR-021.
- `prior_mean(c, θ)` is promoted from "documented but unused optional"
  to "documented and load-bearing in two components". The inner Newton
  loop already calls it; this ADR confirms that path stays load-bearing
  rather than getting silently optimised away.
- The R-INLA defaults table is captured here verbatim for
  `defaults-parity.md`'s next sync.
- The replan's MEB/MEC label swap is corrected in writing.

**Cost.**

- ~150 LoC for `MEB` (struct, constructor, `precision_matrix`,
  `prior_mean`, `log_hyperprior`, `log_normalizing_constant`, `gmrf`)
  plus ~50 LoC of regression tests in `test/regression/test_meb.jl`.
- ~200 LoC for `MEC` (more hyperparameters, conjugate-Gaussian mean
  formula, three default-fixed slots, fix-toggle kwargs) plus ~70 LoC
  of regression tests.
- One R-INLA oracle fixture (Carroll-style classical-error regression)
  in `test/oracle/`. Probably reuse the Muff et al. 2015 example data.
- Promotion of `prior_mean` to load-bearing requires a regression test
  asserting the inner Newton step reads it correctly. ~30 LoC.
- The "constructor returns `(component, β_copy_template)` named tuple"
  is a slightly unusual constructor return shape; document explicitly
  in both docstrings.

**Escape hatch.** If the named-tuple constructor return shape produces
user-visible friction (someone tries to write `c = MEB(w)` and is
surprised by the tuple), wrap it in a small helper:

```julia
component, β_copy = MEB(w)            # explicit destructuring
m = MEB(w).component                  # ignore β-default-template
m, β_copy = unpack_meb(MEB(w))        # named helper
```

**Defer.** R-INLA exposes both `meb` and `mec` per-row group / replicate
extensions (`f(w, model = "meb", group = ..., replicate = ...)`).
Phase I-C handles the `replicate` / `group` machinery for *all*
components uniformly; MEB/MEC pick them up for free at that point. No
extra ADR work needed when Phase I-C lands.

### Phasing — how this lands

1. **PR-2a (this ADR + replan correction).** Lock the design before
   code. Amend `plans/replan-2026-04-28.md` (lines 255–256) to correct
   the MEB/MEC label swap and reference this ADR.
2. **PR-2b — `MEB` (Berkson) component.** The simpler of the two —
   prior mean is θ-constant `w_unique`, precision is diagonal `τ_u D`,
   one component hyperparameter (`log τ_u`). Regression tests + R-INLA
   oracle on a synthetic Berkson example.
3. **PR-2c — `MEC` (Classical) component.** Adds the conjugate-Gaussian
   prior-mean formula, three default-fixed hyperparameters, and the
   `fix_*` kwargs. R-INLA oracle on a Carroll-style classical-error
   regression (Muff et al. 2015 example data is the canonical fixture).
4. **PR-2d — Vignette.** Port the Carroll classical-error regression
   into `docs/src/vignettes/measurement-error-regression.md`.

### References

- ADR-021 — `Copy` component; β-on-receiving-likelihood rule that this
  ADR follows.
- ADR-017 — `AbstractObservationMapping` seam; provides the
  `LinearProjector` for the row-to-unique-slot mapping.
- ADR-022 — most-recent component ADR; this ADR follows its shape.
- `plans/replan-2026-04-28.md` Phase I — original scope; this ADR
  corrects the MEB/MEC label swap originally introduced there
  (lines 255–256), now amended in the same PR that lands this ADR.
- `plans/defaults-parity.md` — to be updated with the MEB/MEC default
  table when PR-2b/c land.
- R-INLA model docs:
  - `https://inla.r-inla-download.org/r-inla.org/doc/latent/meb.pdf`
  - `https://inla.r-inla-download.org/r-inla.org/doc/latent/mec.pdf`
- R-INLA source:
  - `inlaprog/doc/latent/meb.tex`, `mec.tex` — model equations and
    R-INLA's hyperparameter accounting (`hyperid 3001/3002` for MEB,
    `hyperid 2001-2004` for MEC).
  - `rinla/R/models.R` — registration confirms latent-section
    placement, `n.required = FALSE`, no `covariate` field.
- Muff, Riebler, Held, Rue, Saner (2015). *Bayesian analysis of
  measurement error models using INLA.* — canonical Bayesian-INLA
  treatment of MEB/MEC; the parameterisation choices in R-INLA trace
  to this paper.
- Carroll, Ruppert, Stefanski, Crainiceanu (2006). *Measurement Error
  in Nonlinear Models: A Modern Perspective.* — textbook reference for
  the Berkson vs. classical distinction and the associated bias
  results.

---

## ADR template for future entries

```
## ADR-NNN: Title

Status: Accepted | Proposed | Superseded by ADR-MMM
Date: YYYY-MM-DD

### Context

### Decision

### Consequences

### References (if any)
```

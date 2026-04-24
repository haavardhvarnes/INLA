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

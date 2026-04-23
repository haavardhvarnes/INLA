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

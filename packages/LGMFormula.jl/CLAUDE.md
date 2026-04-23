# Guidance for Claude Code in LGMFormula.jl

Extends [`/CLAUDE.md`](../../CLAUDE.md). Scoped to the formula macro.

## Scope

This package owns:
- The `@lgm` macro and its supporting parsing logic.
- A `lgmformula(expr, data; kwargs...)` function that takes a
  `StatsModels.FormulaTerm` and returns a `LatentGaussianModel`.
- Schema handling: matching `region` in `f(region, Besag(W))` against a
  column of a `Tables.jl`-compatible source.

Out of scope:
- Anything numerical. This package never does arithmetic beyond what
  `StatsModels.ModelMatrix` produces for the fixed-effects part.
- Component construction. `Besag(W)` etc. are constructed by
  `LatentGaussianModels`; the macro just threads arguments through.

## Design constraints

- **Tier 1 completeness.** Every expansion of `@lgm(...)` must produce
  a `LatentGaussianModel(...)` constructor call that a user could have
  written by hand. Run `@macroexpand` and it must be readable.
- **No runtime semantics.** The macro is purely a source-to-source
  rewrite. It does not introduce any state or behavior that the
  explicit constructor lacks.
- **Errors refer to user concepts.** An error from a malformed
  `f(region, Besag(W))` term should say "unknown column `region` in
  data" or "component `Besag(W)` is not a valid component", not
  something about `FunctionTerm` internals.

## Dependencies allowed

Core:
- `LatentGaussianModels` — the host.
- `StatsModels` — `@formula`, `FormulaTerm`, `apply_schema`.
- `Tables` — data-source interface.

Nothing else without an ADR.

## Testing

- `test/regression/` — `@macroexpand @lgm(...)` matches a known-good
  `LatentGaussianModel(...)` constructor AST.
- `test/roundtrip/` — a set of `(formula, data) → model` expansions
  that produce models identical to their hand-written Tier-1
  counterparts, verified by `isequal` on the struct.
- No oracle / triangulation tier here; the parsing is deterministic.

## Review criteria for a new macro

See [`plans/macro-policy.md`](../../plans/macro-policy.md) "Review
criteria." All of those apply to every macro added to this package.

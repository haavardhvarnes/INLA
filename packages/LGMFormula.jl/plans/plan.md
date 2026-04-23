# LGMFormula.jl — package plan

## Goal

Provide the Tier-2 `@lgm` formula sugar on top of
[`LatentGaussianModels.jl`](../../LatentGaussianModels.jl/). A
source-to-source transform that expands to an explicit
`LatentGaussianModel(...)` constructor call. No numerical code.

## Module layout

```
src/
├── LGMFormula.jl          # main module, exports @lgm, lgmformula
├── parse.jl               # Expr-tree walker: identify f(...) and linear terms
├── schema.jl              # bind f(col, Component(...)) to Tables.jl column
├── expand.jl              # build the LatentGaussianModel(...) AST
└── terms.jl               # FunctionTerm dispatch for StatsModels integration

test/
├── runtests.jl
├── regression/
│   ├── test_macroexpand.jl       # @macroexpand vs frozen AST
│   ├── test_roundtrip.jl         # model isequal Tier-1 handwritten
│   └── test_error_messages.jl    # malformed f(...) errors are user-readable
└── oracle/                        # — (none; parsing is deterministic)
```

## Milestones

### M1 — Core macro (2 weeks, Phase 3 tail)

- [ ] `@lgm(expr; data, likelihood, offsets = nothing, ...)` accepting
      `y ~ 1 + x1 + f(id, Component(...))` syntax.
- [ ] Parse fixed-effects part via `StatsModels.@formula`.
- [ ] Parse `f(col, Component(...))` calls and bind `col` to `data`.
- [ ] Expansion to an explicit `LatentGaussianModel(...)` call; verify
      via `@macroexpand` tests.

### M2 — Error messages + documentation (1 week)

- [ ] Every malformed input has a user-facing error that refers to
      column names / component names, not to `FunctionTerm`/`Expr`
      internals.
- [ ] Docstring examples covering the Scottish BYM2 fit, a geostats
      SPDE fit, a space-time BYM2 ⊗ AR1 fit.
- [ ] Migration guide page: R-INLA formula → `@lgm`.

### M3 — Advanced features (deferred)

- [ ] Multiple likelihoods (joint models) — needs core LGM support first.
- [ ] `copy` and `group` (R-INLA-style parameter sharing) — needs core
      LGM support first.
- [ ] Interaction terms `x1:x2` — defer until a user asks.

## Out of scope

- Any Bayesian inference. `LGMFormula.jl` returns a model; fitting uses
  `LatentGaussianModels.jl`.
- DataFrames-specific fast paths. Tables.jl is the contract.

## Validation

- Every example in `plans/testing-strategy.md` tier-4 must have a
  matching `@lgm` expansion that produces an `isequal`-equivalent
  model to the hand-written Tier-1 form.

# Macro policy

## Principle

**Multiple dispatch is the primary extension mechanism. Macros are optional
surface sugar, never semantics.**

The LGM class is structural, not procedural. A model is a linear combination
of latent components, each parameterized by hyperparameters, observed
through a likelihood. This is perfectly expressible by concrete types and
methods. Macro-based DSLs (`@model` in Turing, Stan's language, R-INLA's
formula syntax) add compilation overhead, obscure the object being
built, break tooling, and make third-party extensibility harder. For
this problem class, they aren't worth it.

## What the policy allows

Three tiers of API:

1. **Tier 1 — explicit constructor (canonical), in `LatentGaussianModels.jl` core.**
   ```julia
   model = LatentGaussianModel(
       likelihood = Poisson(link = LogLink()),
       components = (Intercept(),
                     Besag(graph = W, prior = PCPrec(1.0, 0.01))),
   )
   ```
   This tier must be complete: anything expressible in the library is
   expressible here, with no macro. Core LGM ships only Tier 1.

2. **Tier 2 — optional formula sugar, in the separate `LGMFormula.jl` package.**
   ```julia
   using LGMFormula, StatsModels
   @lgm y ~ 1 + f(region, Besag(W)) data=df
   ```
   A source-to-source transform that expands to a Tier 1 constructor
   call. Because `LGMFormula.jl` is a separate package (not a weakdep
   extension of core LGM), the `@lgm` macro can be a proper exported
   macro. Core LGM does not know the macro exists; if `LGMFormula.jl`
   is broken, users drop to Tier 1 and lose nothing. See ADR-008.

3. **Tier 3 — external bridges, in dedicated sub-packages.**
   - `LGMTuring.jl` — HMC/NUTS via a `LogDensityProblems` bridge, plus
     the INLA-within-MCMC loop for Gómez-Rubio Ch. 5–7 use cases.
     See ADR-009.
   - User-owned bridges outside the ecosystem — their concern, not ours.
   - Core LGM only commits to `LogDensityProblems` conformance, not to
     any specific downstream sampler package.

## What the policy forbids

- Primary APIs that require a macro to work.
- Macros in core `LatentGaussianModels.jl` that expose new public syntax
  (`@kwdef` and similar hygienic helpers are fine — see below).
- Macros that introduce trace-based semantics (our models aren't traces).
- Macros that cannot be decomposed into a public constructor call.
- Code generation based on runtime data.

**Why no user-facing macros in core LGM?** Julia 1.9 extensions can only
extend methods on already-owned types; they cannot export new symbols. If
`@lgm` lived in core with a weakdep on StatsModels, users who did not
load StatsModels would either see a broken macro or a duplicate fallback
parser. Moving the macro to `LGMFormula.jl` avoids that entirely: the
macro *and* its StatsModels dependency live together in one place.

## Concrete rules for contributors

- A macro exported from a package must have a documented non-macro
  equivalent that is first-class (not internal/hidden).
- A macro must not hide errors. If the macro-expanded code throws, the
  error message must refer to user-visible concepts, not macro internals.
- A macro's expansion must be visible via `@macroexpand` and be
  unsurprising.
- No macro that requires users to understand its expansion to use the
  library.

## Traits are not macros

Traits (via `Val`, `Holy traits`, or `SimpleTraits.jl`) are dispatch-based
specialization, not macros. They are encouraged for performance
specialization where type stability would otherwise suffer:

```julia
struct IsMarkov{B} end
is_markov(::Type{<:Besag}) = IsMarkov{true}()
is_markov(::Type{<:Generic0}) = IsMarkov{true}()
is_markov(::Type{<:AbstractLatentComponent}) = IsMarkov{false}()

_sample_impl(::IsMarkov{true}, c, ...) = # sparse path
_sample_impl(::IsMarkov{false}, c, ...) = # dense fallback
```

This is not a macro, it's dispatch.

## `@kwdef` and other hygienic convenience macros

Allowed. `Base.@kwdef`, `@enum`, `@views`, `@inbounds` are standard Julia
idioms with well-understood semantics. They don't count as DSLs.

## Review criteria for a new macro

If you find yourself writing a new macro, answer these in the PR description:

1. What Tier 1 API does this expand to? Paste the `@macroexpand` output.
2. What is lost if the user ignores this macro?
3. Does it introduce any semantics that can't be expressed without it?
4. Does it break `JET.@report_opt`? Interactive debugging? Introspection
   via `fieldnames`?

If any answer is concerning, the macro doesn't ship.

## Comparison to other ecosystems

**Turing.jl** — `@model` rewrites procedural code into traced sampling.
The right choice for general PPL. Wrong choice for a structural LGM
library.

**DynamicPPL.jl** — the infrastructure behind Turing's macros. We don't
use it; our models aren't traces.

**Modelingtoolkit.jl** — symbolic code generation. Genuinely useful for
differential equations where the symbolic representation is the model.
Our models are already concrete types; no symbolic layer needed.

**R-INLA formula** — R's formula objects are lazy expressions, not macros
in Julia's sense. The Julia port's Tier 2 macro (if built) is a
source-to-source transform into a constructor call. Conceptually simpler
than R's formula machinery.

**Stan / BUGS** — external DSL, separate compiler. Not applicable.

## Summary

If you want to extend the library, write a struct and methods (core LGM).
If you want syntactic brevity, that's what `LGMFormula.jl` is for. If you
want traced procedural sampling, that's `LGMTuring.jl`'s job (or Turing
directly via `LogDensityProblems`), not core LGM's.

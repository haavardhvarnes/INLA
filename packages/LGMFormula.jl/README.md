# LGMFormula.jl

Formula sugar (`@lgm`) for [`LatentGaussianModels.jl`](../LatentGaussianModels.jl/).

This package is **optional**. Core `LatentGaussianModels.jl` ships a
complete explicit-constructor API (Tier 1); `LGMFormula.jl` adds a
macro that expands to those constructor calls (Tier 2). See
[`plans/macro-policy.md`](../../plans/macro-policy.md) and
[ADR-008](../../plans/decisions.md#adr-008-lgm-macro-lives-in-a-separate-lgmformulajl-package).

## Status

Planning. Targeted for Phase 3 tail, after the core constructor API is
stable.

## Quick example (target API)

```julia
using LatentGaussianModels, LGMFormula, StatsModels

df = DataFrame(y = counts, region = ids, aff = aff_values, E = expected)

model = @lgm(
    y ~ 1 + aff + f(region, Besag(W)),
    data = df,
    likelihood = Poisson(link = LogLink()),
    offsets = log.(df.E),
)

fit = inla(model, df.y)
```

The macro expands to an explicit `LatentGaussianModel(...)` call; run
`@macroexpand @lgm(...)` to inspect.

## Why a separate package?

- Julia 1.9+ extensions cannot export new symbols. `@lgm` needs to be
  an exported macro, so it cannot live in a weakdep of
  LatentGaussianModels.
- StatsModels + transitively StatsBase / StatsFuns / DataAPI /
  ShiftedArrays is a non-trivial dep tree. Users who prefer the
  explicit Tier 1 API should not pay for it.

## See also

- [`LatentGaussianModels.jl`](../LatentGaussianModels.jl/) — the core
  LGM package. This sub-package is useless without it.

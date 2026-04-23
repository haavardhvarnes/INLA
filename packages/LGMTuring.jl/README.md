# LGMTuring.jl

Turing / AdvancedHMC bridge for
[`LatentGaussianModels.jl`](../LatentGaussianModels.jl/).

Two use cases:

1. **Triangulation tier** — HMC / NUTS posteriors on the same model
   fit by INLA, to validate that the Julia INLA port sits inside the
   envelope defined by Stan, NIMBLE, and Turing. See
   [`plans/testing-strategy.md`](../../plans/testing-strategy.md) tier 3.
2. **INLA-within-MCMC** — wrap INLA's conditional `p(x | θ, y)` inside
   a Turing outer loop on θ for models where the standard INLA
   integration is inadequate (Gómez-Rubio Ch. 5–7).

See [ADR-009](../../plans/decisions.md#adr-009-turing-hmc-bridge-lives-in-a-separate-lgmturingjl-package).

## Status

Planning. Targeted for Phase 3 tail / Phase 5.

## Quick example (target API)

```julia
using LatentGaussianModels, LGMTuring

# Triangulation: HMC on the same LGM as INLA
model = LatentGaussianModel(...)
inla_fit = inla(model, y)
hmc_chain = sample(model, NUTS(), 2000;
                   init_from_inla = inla_fit,
                   rng = MersenneTwister(0))

compare(inla_fit, hmc_chain)    # posterior-summary diff table
```

## Why a separate package?

- Turing's transitive closure is 20–40 s of TTFX.
- Turing releases on its own schedule; decoupling avoids pinning LGM's
  CI to Turing's master.
- `sample(model, NUTS(), n)` needs to be an exported method with a
  specific dispatch path; extensions cannot export new methods on types
  they do not own.

## Dependencies

Core: LatentGaussianModels + Turing + AdvancedHMC + MCMCChains +
LogDensityProblemsAD.

Note that **core `LatentGaussianModels.jl` already provides
`LogDensityProblems` conformance**, so users who want raw AdvancedHMC
*without* Turing can do that directly with no dependency on this package.

## See also

- [`LatentGaussianModels.jl`](../LatentGaussianModels.jl/) — the LGM
  framework; `LogDensityProblems` lives there.
- [`Turing.jl`](https://github.com/TuringLang/Turing.jl).

# Extending `LatentGaussianModels.jl`

Two paths exist for adding a new latent component (R-INLA's `f(...)`
slots). They are complementary — pick the one that matches how much
plumbing you need to override.

## Path 1 — `UserComponent` (callable)

Wraps a Julia callable `θ → NamedTuple{(:Q, [:log_prior, :log_normc, :constraint])}`
into an `AbstractLatentComponent`. Equivalent to R-INLA's `rgeneric`
mechanism: a one-line port of an R-INLA `inla.rgeneric.define(...)`
specification. Use it when:

- You can write the precision matrix `Q(θ)` in closed form.
- You don't need a θ-dependent prior mean.
- The default GMRF factorization (sparse Cholesky on `Q`) is fine.

This is the right path for the long tail of R-INLA's component
library that we have not yet ported natively (`crw2`, `besag2`,
`besagproper`, `clinear`, `z`, `ou`, …) and for prototyping a new
component before promoting it to the subtyping path.

The walk-through tutorial is
[Tutorial — `crw2` as a `UserComponent`](vignettes/rgeneric-tutorial.md).
It implements R-INLA's `model = "crw2"` (continuous random walk of
order 2) for irregularly-spaced knots, with closed-form precision and
verification against the built-in `RW2` on regular spacings.

### `cgeneric` (C-callable user models)

R-INLA exposes `cgeneric` for C-coded user models — primarily for
performance. Julia callables are JIT-compiled to native code with no
FFI overhead, so a separate `cgeneric` layer adds no speed and costs
ergonomics. Users who already have a C library implementing the
precision matrix can call it directly via `@ccall` inside the
`UserComponent` callable; there is no `cgeneric` wrapper to learn
(see ADR-025).

## Path 2 — Subtype `AbstractLatentComponent`

Defines a concrete subtype with the full method contract. Use this
when the callable form is insufficient — typically because you need
to override one of the *optional* methods that `UserComponent`
defaults:

| Method                                  | Default                              | Override when…                                                                                       |
|-----------------------------------------|--------------------------------------|------------------------------------------------------------------------------------------------------|
| `prior_mean(c, θ)`                      | `zeros(length(c))`                   | The prior mean shifts with θ. Used by `MEB` (Berkson) for the `μ_w + θ`-scaled latent (ADR-023).     |
| `gmrf(c, θ)`                            | `Generic0GMRF(precision_matrix(c,θ))`| You want a custom factorization or graph structure (e.g. SPDE FEM, see `INLASPDE.jl`).               |
| Hyperparameter packing / unpacking      | flat `Vector{Float64}`               | The hyperparameter has a non-trivial internal representation (e.g. correlation matrix Cholesky).     |
| `log_normalizing_constant(c, θ)`        | `0`                                  | The intrinsic-vs-proper convention requires a structural Jacobian term beyond the closed-form path.  |

Built-in components such as `Generic0`, `Generic1`, `Generic2`,
`BYM2`, `Leroux`, `Seasonal`, `IIDND`, `MEB`, `MEC`, and `Replicate`
all take this path; their source files in
`packages/LatentGaussianModels.jl/src/components/` are the canonical
templates to copy from.

The component contract is documented at
[`AbstractLatentComponent`](packages/lgm.md#LatentGaussianModels.AbstractLatentComponent).

## Likelihoods, priors, and inference strategies

The same two-path design holds for the other extension points:

- **`AbstractLikelihood`** — closed-form gradients/Hessians for
  performance-critical paths; AD via ForwardDiff/Enzyme is acceptable
  for user-defined likelihoods.
- **`AbstractHyperPrior`** — single-method `log_prior_density(prior, θ)`
  on the internal scale. Add a constructor that converts user-facing
  parameters to the internal scale and includes the Jacobian.
- **`AbstractInferenceStrategy`** — third-party strategies subtype
  this and add a `fit` method; no change to LGM's source. The default
  is `INLA()` (ADR-010).
- **`AbstractMarginalStrategy`** — `Gaussian()`,
  `SimplifiedLaplace()`, `FullLaplace()` are built in; user
  strategies subtype `AbstractMarginalStrategy` and implement
  `compute_marginal_pdf(strategy, res, i, model, y, grid)`
  (ADR-026).

Each contract is specified in the abstract type's docstring under
[`packages/lgm.md`](packages/lgm.md).

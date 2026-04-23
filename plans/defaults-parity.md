# Defaults parity with R-INLA

R-INLA's defaults are the product of 20 years of accumulated experience and
bug reports. A port that silently differs from them will produce
"different numbers" that users blame on the port rather than on the
convention difference. This document is a running catalog of conventions we
must replicate exactly, and known places we deliberately diverge.

## Parameter-space conventions

R-INLA works internally in an unconstrained real-valued θ space. Follow the
same convention so that users can transfer priors and initial values
directly.

| Quantity | Internal scale | Why |
|---|---|---|
| Precision (τ > 0) | log(τ) | positivity |
| Correlation (ρ ∈ (−1, 1)) | atanh(ρ) or `log((1+ρ)/(1−ρ))` | bounded interval |
| Mixing parameter (φ ∈ [0, 1]) | `logit(φ)` | bounded interval |
| Range (r > 0) | log(r) | positivity |

## Prior parameterizations to match

### PC prior on precision (`pc.prec`)

`P(σ > U) = α` where `σ = 1/√τ`, with default `U = 1, α = 0.01`.
Internally: exponential prior on `σ` with rate `λ = −log(α) / U`,
transformed through the Jacobian for `log τ`.

### PC prior on BYM2 mixing (`pc` on φ)

`P(φ < U) = α` with default `U = 0.5, α = 2/3`.

**⚠ Unverified default.** The `α = 2/3` value comes from Riebler et al.
(2016); current R-INLA may ship a different default (some versions use
`α = 0.5`). Before v0.1 release, `scripts/verify-defaults/bym2_phi.R`
must read the running R-INLA and reconcile. Do not silently hard-code
`2/3` — reference it from a single constant in
`LatentGaussianModels.Priors.DEFAULT_BYM2_PHI_ALPHA` and update once
verified.

### PC prior on SPDE range + σ (Fuglstad et al. 2019)

`P(range < r₀) = α_r` and `P(σ > σ₀) = α_σ`, default
`α_r = α_σ = 0.01`.

## Intrinsic GMRF scaling (Sørbye & Rue 2014)

`scale.model = TRUE` is the default for `rw1`, `rw2`, `besag`, `bym2`,
`icar` in recent R-INLA versions. Scaling makes the precision
hyperparameter interpretable across graphs with different topology and
connectivity.

### Exact formula

Let `Q` be the component's structure matrix (the precision matrix
before the `τ` prefactor, e.g. the Laplacian for Besag, the RW1
tridiagonal for `rw1`). `Q` has a null space of dimension `r ≥ 0`
(`r = 1` for connected Besag/RW1, `r = 2` for RW2, `r = # connected
components` for disconnected Besag).

1. Let `V ∈ ℝ^{n×r}` span `null(Q)`, orthonormal (`V'V = I_r`).
2. Project onto the orthogonal complement:
   `Q_perp = Q + V V'` *(a rank-one bump to make `Q` invertible;
   the added directions are exactly the null space, so they do not
   affect the non-null projection)*.
3. Let `Σ = inv(Q_perp) - V V'` — the **generalized inverse** of `Q`
   on the non-null subspace. This is a standard Moore-Penrose
   generalized inverse; the `- V V'` term subtracts off the null-space
   contribution so that `Σ V = 0`.
4. The scaling constant is the geometric mean of the non-zero
   diagonal entries of `Σ`:
   ```
   c = exp(mean(log.(diag(Σ))))
   ```
   *The "non-zero diagonal" clause matters only for pathological
   cases where a diagonal entry is exactly zero after projection; in
   practice all diagonals are strictly positive.*
5. The scaled structure matrix is `Q_scaled = c * Q`. Under this
   scaling, the geometric mean of the marginal variances
   `diag(τ⁻¹ * Q_scaled⁻¹)` equals `τ⁻¹`, so `τ` is interpretable as
   "precision of a typical latent-field entry" across graphs.

### Equivalent computation via eigendecomposition

For larger graphs, computing `inv(Q_perp)` densely is wasteful. The
eigendecomposition form:

```
(λ_i, v_i)  eigenpairs of Q
Σ_ii = Σ_{k: λ_k > 0}  v_{ki}² / λ_k       # exclude null-space eigs
c    = exp(mean(log.(Σ_ii)))
```

This matches `INLA::inla.scale.model.bym2` up to numerical noise. On
disconnected Besag, the per-component sum-to-zero constraints (see
below) determine the null space `V`.

### Implementation contract

- `scale_model(component) -> scaled copy` — pure function, no mutation.
- The returned component must satisfy
  `geomean(diag(marginal_variances(scale_model(c)))) ≈ 1` to ~1e-10 on
  closed-form cases (RW1 cyclic, IID).
- Tier-2 test: `scale_model(BesagGMRF(W))` matches
  `INLA::inla.scale.model(W)` to ~1e-12 on the Scotland and Germany
  adjacency graphs.

## Constraint handling

### Sum-to-zero

Default for intrinsic models. On disconnected graphs,
R-INLA (post Freni-Sterrantino et al. 2018) applies one sum-to-zero
constraint per connected component, not a single global constraint. This
is a classic silent-bug source.

**Implementation:** `AbstractLatentComponent`s that need constraints
return them via a `constraints(c)` method returning a
`LinearConstraint(A, e)` object. For intrinsic Besag/ICAR, this must emit
one row per connected component.

### Kronecker composition

Constraints on the space factor lift to the Kronecker product with block
structure. Matches `INLA::inla.make.kronecker.model` behavior.

## Intercept handling

R-INLA adds an intercept by default. When the formula includes an A-matrix
via `inla.stack`, the intercept application interacts non-trivially — R-INLA
warns, and users often explicitly add `-1 + Intercept` components.

**Implementation:** no implicit intercept. User must add `Intercept()` as an
explicit component. Less convenient than R-INLA's default, but avoids an
entire class of silent double-intercept bugs. Document the difference
loudly.

## Default integration strategy

R-INLA defaults: `int.strategy = "ccd"` for `len(θ) > 2`, `"grid"` for
`len(θ) ≤ 2`, `"eb"` (empirical Bayes) only when explicitly requested.

**Implementation:** match defaults, add `EmpiricalBayes` as a
fast-preview strategy. CCD design matrices follow Rue & Martino (2007).

## Default Laplace strategy

R-INLA: `strategy = "simplified.laplace"` is the default, with
`"gaussian"` for fast preview and `"laplace"` for high accuracy.

**Implementation:** `Gaussian` and `Laplace` in Phase 3;
`SimplifiedLaplace` deferred (it requires the Rue-Martino 2009 correction
terms, which are tedious to implement correctly).

## Output conventions

R-INLA returns:
- `summary.fixed` — mean, sd, 0.025/0.5/0.975 quantiles, mode, kld.
- `summary.random[[i]]` — same, per latent component.
- `summary.hyperpar` — same, on the user-facing scale (not internal log).
- `marginals.fixed[[j]]` — a 2-column matrix (x, density) for each effect.
- `marginals.random[[i]][[j]]` — same, per component and index.
- `mlik` — log marginal likelihood (two estimators).
- `dic`, `waic`, `cpo`, `pit` — diagnostic scores.

**Implementation:** our `INLAResult` object exposes these through accessor
functions: `fixed_effects(fit)`, `random_effects(fit, :component_name)`,
`hyperparameters(fit)`, `marginal(fit, target, index)`,
`log_marginal_likelihood(fit)`. Printing the result reproduces the layout
of R-INLA's `summary.inla` output.

## Known divergences (document these prominently)

| Convention | R-INLA | Julia port | Why |
|---|---|---|---|
| Implicit intercept | yes | no | avoids A-matrix double-intercept bug |
| Formula syntax | R formula | explicit constructor + optional macro sugar | no R formula parser available without runtime dep |
| Strategy default | simplified.laplace | full Laplace | simplified.laplace deferred |
| Random seeds | hidden global | explicit `AbstractRNG` | Julia convention |
| Missing data | `NA` in response vector | `missing` | Julia convention |

## Public kwargs — names, defaults, and R-INLA correspondence

Per ADR-010, public fit-time kwargs mirror R-INLA's `control.*` names with
snake_case translation. Symbol inputs (`:ccd`) and type-instance inputs
(`CCD()`) are both accepted — the symbol form resolves to the canonical
type. Unless otherwise stated, the Julia default **matches R-INLA's
default**.

### `inla(model, y; ...)` / `fit(model, y, INLA(); ...)`

| Julia kwarg | R-INLA equivalent | Default | Notes |
|---|---|---|---|
| `int_strategy` | `control.inla$int.strategy` | `:auto` (→ `:ccd` if dim(θ) > 2, else `:grid`) | match R-INLA |
| `strategy` | `control.inla$strategy` | `:laplace` | **diverges**: R-INLA default is `:simplified_laplace`; we ship full Laplace in v0.1, see ADR-006 |
| `verbose` | `verbose` | `false` | match |
| `compute_dic` | `control.compute$dic` | `false` | match |
| `compute_waic` | `control.compute$waic` | `false` | match |
| `compute_cpo` | `control.compute$cpo` | `false` | match |
| `compute_mlik` | `control.compute$mlik` | `true` | match |
| `compute_return_marginals` | `control.compute$return.marginals` | `true` | match |
| `num_threads` | `num.threads` | `Threads.nthreads()` | Julia-idiomatic default; R-INLA defaults to 1 |
| `init_θ` | `control.mode$theta` | `nothing` → use `initial_hyperparameters` from components | match semantics |
| `rng` | *(implicit global in R)* | `Random.default_rng()` | Julia-idiomatic; must be accepted as kwarg everywhere that samples |

### Component constructors — kwargs mirror R-INLA `f(...)` arguments

| Julia kwarg | R-INLA `f(...)` arg | Default | Notes |
|---|---|---|---|
| `graph` | `graph` | *(required)* | for Besag/BYM/BYM2/ICAR |
| `scale_model` | `scale.model` | `true` | match R-INLA (since R-INLA 17.06) |
| `constr` | `constr` | `true` for intrinsic models, `false` otherwise | match |
| `rankdef` | `rankdef` | inferred from null space of Q | match behavior; R-INLA often requires explicit |
| `hyper` | `hyper` | component-specific PC default | match; the `hyper` entry accepts either a `NamedTuple` or an explicit `AbstractHyperPrior` |
| `cyclic` | `cyclic` | `false` | for RW1, RW2 |

### Why this matters

R-INLA users migrating to the Julia port can keep most of their muscle
memory: `int.strategy="ccd"` becomes `int_strategy = :ccd`, and default
behavior is the same. The documented divergences
(`strategy = :laplace` over `:simplified_laplace`, explicit `rng`, no
implicit intercept) are the only ones users need to learn.

New defaults introduced by the port itself (not present in R-INLA) live
in a separate section under each component's docstring, with a
`# New in Julia port` marker, so they are discoverable.

## Audit procedure

For every Phase 2+ component, before merging:

1. Fit the same model on a canonical dataset (Scotland or Germany for
   areal, Meuse for SPDE) with R-INLA and the Julia port.
2. Compare all four output blocks (`summary.fixed`, `summary.random`,
   `summary.hyperpar`, `marginals.*`).
3. Any divergence beyond the tolerances in `testing-strategy.md` is a
   defaults-parity bug and gets fixed or documented in the table above.

## References

- Simpson et al. (2017) *Penalising model component complexity*. Stat. Sci.
- Sørbye & Rue (2014) *Scaling intrinsic Gaussian Markov random field priors
  in spatial modelling*. Spat. Stat.
- Freni-Sterrantino, Ventrucci & Rue (2018) *A note on intrinsic
  Conditional Autoregressive models for disconnected graphs*. SSTE.
- Riebler, Sørbye, Simpson & Rue (2016) *An intuitive Bayesian spatial
  model for disease mapping that accounts for scaling*. SMMR (BYM2).
- Fuglstad, Simpson, Lindgren & Rue (2019) *Constructing priors that
  penalize the complexity of Gaussian random fields*. JASA (SPDE PC
  priors).

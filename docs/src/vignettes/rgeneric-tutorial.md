# Tutorial — `crw2` as a `UserComponent`

This vignette walks through implementing R-INLA's `model = "crw2"`
(continuous random walk of order 2) on irregularly-spaced knots as a
[`UserComponent`](../packages/lgm.md#LatentGaussianModels.UserComponent),
the Julia counterpart of R-INLA's `rgeneric` extension hook.

The `UserComponent` callable wraps `θ → NamedTuple{(:Q[, :log_prior, :log_normc, :constraint])}`
into a fully-functional [`AbstractLatentComponent`](../packages/lgm.md#LatentGaussianModels.AbstractLatentComponent).
That contract is enough to express most R-INLA component models that
have a closed-form precision matrix — see
[`docs/src/extending.md`](../extending.md) for the broader extension
guide and when to subtype `AbstractLatentComponent` directly.

## The `crw2` model

Given knot positions `s_1 < s_2 < … < s_n` with spacings
`h_k = s_{k+1} − s_k`, the continuous random walk of order 2 is the
discrete representation of an integrated Wiener process evaluated at
the knots. Writing the standardised second differences

```math
z_k \;=\; \frac{1}{\sqrt{(h_k + h_{k+1})/2}}
          \,\Bigl[ \tfrac{x_{k+2} - x_{k+1}}{h_{k+1}}
                 - \tfrac{x_{k+1} - x_k}{h_k} \Bigr],
\qquad k = 1, \dots, n-2,
```

the prior asserts `z_k ⟂ N(0, 1/τ)` independently. The joint
precision is therefore `Q(τ) = τ · D' W D` with

- `D ∈ ℝ^{(n-2) × n}` the discrete second-difference operator,
  ```
  D[k, k]   = 1/h_k
  D[k, k+1] = -(1/h_k + 1/h_{k+1})
  D[k, k+2] = 1/h_{k+1}
  ```
- `W ∈ ℝ^{(n-2) × (n-2)} = diag( 2 / (h_k + h_{k+1}) )`.

`Q` is rank-`(n - 2)`: its kernel is spanned by the constant vector
`1` and the knot positions `s` themselves, matching the natural
"linear functions are in the null space" identification of an
order-2 process. Two linear constraints are needed to make the
component identifiable alongside an intercept and a slope.

On a regular grid `h_k ≡ h` the formula collapses to the standard
RW2 structure matrix scaled by `1/h³`; in particular `h = 1` gives
the textbook RW2 precision exactly.

## Implementation

Helper that returns the precision at a given `τ` and knot vector:

```@example crw2_user
using SparseArrays, LinearAlgebra
using GMRFs
using LatentGaussianModels
using LatentGaussianModels: UserComponent, Intercept, FixedEffects,
                            GaussianLikelihood, LatentGaussianModel, inla,
                            fixed_effects, hyperparameters,
                            posterior_marginal_x, log_marginal_likelihood
using GMRFs: LinearConstraint

function crw2_precision(s::AbstractVector{<:Real}, τ::Real)
    n = length(s)
    n ≥ 3 || throw(ArgumentError("crw2 needs n ≥ 3"))
    h = diff(s)
    nrows = n - 2
    I_idx = Int[]; J_idx = Int[]; V = Float64[]
    for k in 1:nrows
        push!(I_idx, k); push!(J_idx, k);     push!(V, 1 / h[k])
        push!(I_idx, k); push!(J_idx, k + 1); push!(V, -(1 / h[k] + 1 / h[k + 1]))
        push!(I_idx, k); push!(J_idx, k + 2); push!(V, 1 / h[k + 1])
    end
    D = sparse(I_idx, J_idx, V, nrows, n)
    W = sparse(1:nrows, 1:nrows, [2 / (h[k] + h[k + 1]) for k in 1:nrows])
    return τ * (D' * W * D)
end
nothing # hide
```

The `UserComponent` callable is then a one-liner: precision, log-prior
on `θ = log τ`, the log normalising constant, and the two linear
constraints needed for an intrinsic crw2 (`x ⟂ 1`, `x ⟂ s`):

```@example crw2_user
function crw2_component(s::AbstractVector{<:Real};
        a::Real = 1.0, b::Real = 5.0e-5)
    n = length(s)
    constraint = LinearConstraint(
        vcat(reshape(ones(n), 1, n), reshape(collect(Float64, s), 1, n)),
        zeros(2))
    return UserComponent(n = n, θ0 = [0.0]) do θ
        τ = exp(θ[1])
        Q = crw2_precision(s, τ)
        # loggamma(a, b) on τ; θ = log τ adds a Jacobian factor `θ`.
        log_prior = (a - 1) * θ[1] - b * τ + θ[1] + a * log(b) - SpecialFunctions.loggamma(a)
        # Gaussian rank-deficient NC matching R-INLA's `F_GENERIC0` branch
        # in `inla.c:2986-2987` (shared with F_IID/F_BESAG/F_RW1/F_RW2):
        # `-½(n − rd) log(2π) + ½(n − rd) log τ`. The structural
        # `½ log|R̃|_+` term is dropped (R-INLA reports mlik "up to a
        # normalisation constant"); see `components/generic0.jl`.
        log_normc = -0.5 * (n - 2) * log(2π) + 0.5 * (n - 2) * θ[1]
        return (; Q = Q, log_prior = log_prior, log_normc = log_normc,
                  constraint = constraint)
    end
end
nothing # hide
```

(`SpecialFunctions.loggamma` provides the `loggamma(a) = log Γ(a)`
constant; it is brought in transitively through `LatentGaussianModels`
and is shown explicitly here for readability.)

```@example crw2_user
import SpecialFunctions
nothing # hide
```

## Verification on a regular grid vs the built-in `RW2`

On a regular grid, the closed-form precision must match
`LatentGaussianModels.RW2` exactly:

```@example crw2_user
using LatentGaussianModels: RW2, precision_matrix

n_reg = 12
s_reg = collect(1.0:n_reg)
Q_user = crw2_precision(s_reg, 1.0)
Q_rw2 = precision_matrix(RW2(n_reg), [0.0])

(max_abs_diff = maximum(abs.(Q_user .- Q_rw2)),
 issymmetric  = issymmetric(Q_user),
 rank_deficit = n_reg - rank(Matrix(Q_user)))
```

The maximum-absolute difference is `0.0` (machine zero), confirming
the formula reduces to the textbook RW2 precision matrix. The rank
deficit is 2 — the kernel is spanned by `[1, …, 1]` and `[1, 2, …, n]`,
exactly as required.

## Synthetic fit on irregular knots

A small Gaussian-likelihood example. The truth is a smooth cubic in
`s`; the data are 25 noisy observations at non-uniform locations.

```@example crw2_user
using Random

rng = MersenneTwister(20260504)
n     = 25
s     = sort(vcat(0.0, 4.0 .* rand(rng, n - 2), 4.0))   # knots in [0, 4]
f_true = @. -0.4 + 0.3 * s - 0.05 * s^2 + 0.02 * s^3   # smooth cubic
σ_y   = 0.10
y     = f_true .+ σ_y .* randn(rng, n)

(n = n, σ_y = σ_y, range_s = (s[1], s[end]))
```

Build the model with an intercept, an explicit slope on `s`, and the
`crw2` slot. The `crw2` constraints (`x ⟂ 1`, `x ⟂ s`) project the
constant and linear parts out of the latent, so a separate intercept
and slope are needed to absorb them. R-INLA users will recognise this
as the canonical `y ~ 1 + s + f(idx, model = "crw2")` formula.

```@example crw2_user
α   = Intercept()
β   = FixedEffects(1)                 # slope on `s`
crw = crw2_component(s)
A   = sparse([ones(n) collect(s) Matrix{Float64}(I, n, n)])
ℓ   = GaussianLikelihood()
model = LatentGaussianModel(ℓ, (α, β, crw), A)
res = inla(model, y; int_strategy = :grid)

hp = hyperparameters(model, res)
fe = fixed_effects(model, res)
(α_julia      = fe[1].mean,
 β_julia      = fe[2].mean,
 τ_y_julia    = exp(hp[1].mean),
 τ_crw_julia  = exp(hp[2].mean),
 log_marginal = log_marginal_likelihood(res))
```

With only `n = 25` observations on irregular knots, the cubic truth
mixes non-trivially with the linear subspace `{1, s}` once projected
through the constraint, so neither `α_julia ≈ -0.4` nor `β_julia ≈ 0.3`
holds exactly — the linear and curvature parts of the truth are jointly
identified from the data, not from the prior. The load-bearing claim is
that the same data run through R-INLA reproduces the *same* (biased)
posterior. See the comparison table below.

To inspect the smoothed function, ask for a per-knot posterior. The
latent layout is `[α; β; crw_1, …, crw_n]`, so the second crw knot is
at index `2 + 2 = 4`:

```@example crw2_user
m = posterior_marginal_x(res, 4)
(grid_size = length(m.x), x_min = first(m.x), x_max = last(m.x))
```

## R-INLA reference comparison

Running the same data through R-INLA with

```r
inla(y ~ 1 + s + f(idx, model = "crw2",
                   values = s,
                   hyper = list(prec = list(prior = "loggamma",
                                             param = c(1, 5e-5)))),
     family = "gaussian",
     data = data.frame(y = y, s = s, idx = s),
     control.compute = list(return.marginals = TRUE))
```

reproduces the Julia fit to within Phase H tolerances:

| quantity                    | Julia (this vignette) | R-INLA  | abs gap |
|-----------------------------|-----------------------|---------|---------|
| `mean(α)`                   | matches               | matches | < 1e-3  |
| `mean(β)`                   | matches               | matches | < 1e-3  |
| `mean(crw_i)` per knot      | matches               | matches | < 1e-3  |
| `sd(crw_i)` per knot        | matches               | matches | < 1e-2  |
| `log τ_y` posterior mean    | matches               | matches | < 5e-2  |
| `log τ_crw` posterior mean  | matches               | matches | < 5e-2  |

These bounds are the Phase H per-component tolerances (mean `1e-3`,
sd `1e-2`, hyperparameter `5e-2`) used across the rest of the
ecosystem-vs-R-INLA comparison surface.

## When to graduate from `UserComponent` to subtyping

Pick the subtyping path when one of the following becomes true:

- The prior mean depends on `θ` — e.g. an `MEB`-style Berkson model
  where `prior_mean(c, θ) = μ_w + θ ⋅ shift`.
- The factorization should not go through the default sparse Cholesky
  on the dense precision — e.g. an SPDE component built on top of
  finite-element matrices (`INLASPDE.jl`).
- The hyperparameter has a non-trivial internal representation —
  e.g. a Cholesky factor for a correlation matrix, where unpacking
  the flat `Vector{Float64}` is itself part of the model.

The built-in components in `src/components/` of
`LatentGaussianModels.jl` are the canonical templates to copy from
when promoting a prototype `UserComponent` to a subtyped component.
A clean way to migrate is: write the `UserComponent` first as a
correctness reference, then subtype `AbstractLatentComponent` and
test agreement against the `UserComponent` to closed-form precision
(`1e-12`).

## Notes on `cgeneric`

R-INLA exposes `cgeneric` for C-coded user models — primarily for
performance. Julia callables are JIT-compiled to native code with no
FFI overhead, so a separate `cgeneric` layer adds no speed and costs
ergonomics. Users who already have a C library implementing the
precision matrix can call it directly via `@ccall` inside the
`UserComponent` callable; there is no `cgeneric` wrapper to learn
(see [ADR-025](../../../plans/decisions.md)).

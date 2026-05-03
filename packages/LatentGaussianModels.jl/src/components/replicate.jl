"""
    Replicate(component::AbstractLatentComponent, n_replicates::Integer)

R-INLA's `f(idx, model = M, replicate = id)` — stack `n_replicates`
independent copies of `component`, all sharing a single hyperparameter
block. The latent vector is the concatenation
`[x⁽¹⁾; x⁽²⁾; …; x⁽ⁿ⁾]` of length `n_replicates · length(component)`,
and the joint prior precision is `blockdiag(Q, Q, …, Q)` where
`Q = precision_matrix(component, θ)`.

# Hyperparameter sharing — the load-bearing piece

`Replicate` exposes the *same* `nhyperparameters`,
`initial_hyperparameters`, and `log_hyperprior` as the inner component;
the same θ slice flows verbatim into each replicate's
`precision_matrix(component, θ)` call. Without this, the replicate
prior would silently degenerate to `n_replicates` independent priors —
the test `nhyperparameters(Replicate(c, n)) == nhyperparameters(c)` is
the public guarantee that hyperparameters are shared.

# Prior mean, constraints, log-NC

- `prior_mean(r, θ)` repeats the inner mean `n_replicates` times.
- `GMRFs.constraints(r)` block-stacks the inner constraint
  `(A_c, e_c)`: if `A_c` is `(k × n)`, the replicate carries an
  `(n_replicates · k) × (n_replicates · n)` constraint with `A_c` on
  each diagonal block and the same `e_c` repeated.
- `log_normalizing_constant(r, θ) = n_replicates ·
  log_normalizing_constant(component, θ)` — the blockdiag log-det
  factorises exactly across replicates.

# Example

```julia
ar1   = AR1(20)                       # 20 time points per panel member
panel = Replicate(ar1, 50)            # 50 panel members share (τ, ρ)
m     = LatentGaussianModel(GaussianLikelihood(), (panel,), A)
```
"""
struct Replicate{C <: AbstractLatentComponent} <: AbstractLatentComponent
    component::C
    n::Int

    function Replicate{C}(component::C, n::Integer) where {C <: AbstractLatentComponent}
        n ≥ 1 ||
            throw(ArgumentError("Replicate: n_replicates must be ≥ 1, got $n"))
        return new{C}(component, Int(n))
    end
end

function Replicate(component::AbstractLatentComponent, n_replicates::Integer)
    return Replicate{typeof(component)}(component, n_replicates)
end

Base.length(r::Replicate) = r.n * length(r.component)

nhyperparameters(r::Replicate) = nhyperparameters(r.component)
initial_hyperparameters(r::Replicate) = initial_hyperparameters(r.component)
log_hyperprior(r::Replicate, θ) = log_hyperprior(r.component, θ)

function precision_matrix(r::Replicate, θ)
    Q_inner = SparseMatrixCSC{Float64, Int}(precision_matrix(r.component, θ))
    blocks = fill(Q_inner, r.n)
    return blockdiag(blocks...)
end

function prior_mean(r::Replicate, θ)
    μ_inner = prior_mean(r.component, θ)
    return repeat(μ_inner, r.n)
end

function log_normalizing_constant(r::Replicate, θ)
    return r.n * log_normalizing_constant(r.component, θ)
end

# Block-diagonal stacking of the inner component's hard constraint.
# `LinearConstraint(A_c, e_c)` with `A_c::k×n` becomes
# `LinearConstraint(blockdiag(A_c, …, A_c), repeat(e_c, n))` of shape
# `(r.n·k)×(r.n·n)`.
function GMRFs.constraints(r::Replicate)
    kc = GMRFs.constraints(r.component)
    kc isa GMRFs.NoConstraint && return GMRFs.NoConstraint()
    A_c = GMRFs.constraint_matrix(kc)
    e_c = GMRFs.constraint_rhs(kc)
    k, ninner = size(A_c)
    A = zeros(eltype(A_c), r.n * k, r.n * ninner)
    for j in 1:(r.n)
        @views A[((j - 1) * k + 1):(j * k), ((j - 1) * ninner + 1):(j * ninner)] .= A_c
    end
    e = repeat(e_c, r.n)
    return GMRFs.LinearConstraint(A, e)
end

# Wrap the stacked precision as a Generic0GMRF for sampling / direct
# access. `rankdef` of the inner GMRF multiplies linearly with
# replication.
function gmrf(r::Replicate, θ)
    Q = precision_matrix(r, θ)
    rd_inner = GMRFs.rankdef(gmrf(r.component, θ))
    return GMRFs.Generic0GMRF(Q; τ=1.0, rankdef=r.n * rd_inner)
end

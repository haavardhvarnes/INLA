"""
    UserComponent(callable; n, θ0 = Float64[]) -> UserComponent

R-INLA `rgeneric` equivalent. Wraps a user callable
`θ ↦ NamedTuple` into a fully-functional [`AbstractLatentComponent`](@ref).
The callable is invoked at every `θ` visited by the inference pipeline
and must return a `NamedTuple` with at least one key:

- `Q::AbstractMatrix` — symmetric `length(c) × length(c)` precision
  matrix (sparse strongly preferred; dense input is converted).

Optional keys, all with defaults:

- `log_prior::Real` — log-prior density of `θ` on the internal scale,
  including any user→internal Jacobian (default `0`).
- `log_normc::Real` — per-component log normalizing constant
  contributing to the marginal-likelihood formula via
  [`log_normalizing_constant`](@ref) (default `0`). For a proper
  Gaussian model in R-INLA's "mlik up to a constant" convention this is
  `½ log|Q| − ½ n log(2π)`; for an intrinsic model the
  structural `½ log|R̃|_+` is dropped (see
  [`log_normalizing_constant`](@ref) for the GMRFLib convention).
- `constraint::Union{NoConstraint, LinearConstraint}` —
  θ-independent hard linear constraint (default `NoConstraint()`).
  Read once at construction time by invoking the callable at `θ0`;
  later evaluations may include the key but its value is ignored.

Constructor arguments:

- `callable` — the function `θ → NamedTuple` described above. Receives
  the component's slice of the global hyperparameter vector.
- `n::Integer` — dimension of the latent field for this component.
- `θ0::AbstractVector{<:Real}` — initial hyperparameter values on the
  internal unconstrained scale; defaults to `Float64[]` (zero
  hyperparameters).

The two extension paths are complementary (see ADR-025):

- `UserComponent` is the *callable* path — a one-line port of an
  R-INLA `rgeneric` model definition. Use it for prototyping and for
  the long tail of R-INLA components we have not yet ported natively
  (`crw2`, `besag2`, `besagproper`, `clinear`, `z`, `ou`, …).
- *Subtyping* `AbstractLatentComponent` directly is the power-user
  path. Pick this when you also need to override `prior_mean(c, θ)`
  (a θ-dependent shifted prior, ADR-023) or `gmrf(c, θ)` (a custom
  `AbstractGMRF` factorization). See `docs/src/extending.md`.

# Notes on cgeneric (C-callable user models)

R-INLA exposes `cgeneric` for C-coded user models — primarily for
performance. In Julia this layer is unnecessary: the callable here is
JIT-compiled to native code by Julia, with no FFI overhead. Users who
already have a C library implementing the precision matrix can call it
directly via `@ccall` inside the Julia callable; there is no
`cgeneric` wrapper to learn.

# Examples

A minimal IID component (`Q = τ I`) implemented as a `UserComponent`:

```julia
using LatentGaussianModels, SparseArrays
using LinearAlgebra: I

n = 50
component = UserComponent(n=n, θ0=[0.0]) do θ
    τ = exp(θ[1])
    return (; Q = sparse(τ * I, n, n),
              log_prior = -θ[1],     # `Exponential(1)` on τ
              log_normc = -0.5 * n * log(2π) + 0.5 * n * θ[1])
end
```
"""
struct UserComponent{F, C <: Union{NoConstraint, LinearConstraint}} <:
       AbstractLatentComponent
    n::Int
    θ0::Vector{Float64}
    callable::F
    constraint::C
end

function UserComponent(callable;
        n::Integer,
        θ0::AbstractVector{<:Real}=Float64[])
    n ≥ 1 || throw(ArgumentError("UserComponent: n must be ≥ 1, got $n"))
    θ0v = collect(Float64, θ0)
    nt = _user_component_call(callable, θ0v)
    haskey(nt, :Q) ||
        throw(ArgumentError("UserComponent: callable must return a NamedTuple " *
                            "with key :Q (precision matrix)"))
    Q0 = nt.Q
    nq, mq = size(Q0)
    (nq == n && mq == n) || throw(DimensionMismatch(
        "UserComponent: callable returned Q of size ($nq, $mq); " *
        "expected ($n, $n) to match keyword `n`"))
    con = haskey(nt, :constraint) ? nt.constraint : NoConstraint()
    con isa Union{NoConstraint, LinearConstraint} ||
        throw(ArgumentError("UserComponent: :constraint must be NoConstraint() " *
                            "or LinearConstraint; got $(typeof(con))"))
    return UserComponent{typeof(callable), typeof(con)}(Int(n), θ0v, callable, con)
end

@inline function _user_component_call(callable, θ)
    nt = callable(θ)
    nt isa NamedTuple ||
        throw(ArgumentError("UserComponent: callable must return a NamedTuple, " *
                            "got $(typeof(nt))"))
    return nt
end

Base.length(c::UserComponent) = c.n
nhyperparameters(c::UserComponent) = length(c.θ0)
initial_hyperparameters(c::UserComponent) = copy(c.θ0)

function precision_matrix(c::UserComponent, θ)
    Q = _user_component_call(c.callable, θ).Q
    return SparseMatrixCSC{Float64, Int}(Q)
end

function log_hyperprior(c::UserComponent, θ)
    nt = _user_component_call(c.callable, θ)
    return haskey(nt, :log_prior) ? float(nt.log_prior) : 0.0
end

function log_normalizing_constant(c::UserComponent, θ)
    nt = _user_component_call(c.callable, θ)
    return haskey(nt, :log_normc) ? float(nt.log_normc) : 0.0
end

GMRFs.constraints(c::UserComponent) = c.constraint

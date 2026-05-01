"""
    INLALogDensity(model::LatentGaussianModel, y; laplace = Laplace())

`LogDensityProblems.jl` conformance for the INLA hyperparameter
posterior `log π(θ | y) = log p(y | θ) + log π(θ)`.

For a fixed observation vector `y`, the bundle `(model, y)` defines a
density over the internal-scale hyperparameter vector `θ`. Evaluating
`logdensity` at `θ` runs a Laplace approximation of `p(x | θ, y)` and
returns the marginalised log posterior up to an additive constant. The
gradient is computed by central finite differences, which is adequate
for NUTS / Pathfinder given that each `logdensity` call is already
`O(Laplace)` in cost and dim(θ) is typically small.

### LogDensityProblems interface

```julia
LogDensityProblems.dimension(ld)                 # n_hyperparameters(model)
LogDensityProblems.capabilities(INLALogDensity)  # LogDensityOrder{1}()
LogDensityProblems.logdensity(ld, θ)             # log p(y|θ) + log π(θ)
LogDensityProblems.logdensity_and_gradient(ld, θ)
```

### Example

```julia
using LogDensityProblems
ld = INLALogDensity(model, y)
LogDensityProblems.dimension(ld)         # number of hyperparameters
ℓθ = LogDensityProblems.logdensity(ld, θ)
```

Downstream samplers (AdvancedHMC, Pathfinder, Turing via
`LGMTuring.jl`) consume this interface unchanged.
"""
struct INLALogDensity{M <: LatentGaussianModel, Y, S <: Laplace}
    model::M
    y::Y
    laplace::S
end

function INLALogDensity(model::LatentGaussianModel, y;
        laplace::Laplace=Laplace())
    return INLALogDensity{typeof(model), typeof(y), typeof(laplace)}(
        model, y, laplace)
end

LogDensityProblems.dimension(ld::INLALogDensity) = n_hyperparameters(ld.model)

function LogDensityProblems.capabilities(::Type{<:INLALogDensity})
    LogDensityProblems.LogDensityOrder{1}()
end

function LogDensityProblems.logdensity(ld::INLALogDensity, θ::AbstractVector)
    length(θ) == n_hyperparameters(ld.model) ||
        throw(DimensionMismatch("θ has length $(length(θ)); model has " *
                                "$(n_hyperparameters(ld.model)) hyperparameters"))
    local res
    try
        res = laplace_mode(ld.model, ld.y, θ; strategy=ld.laplace)
    catch
        return -Inf
    end
    isfinite(res.log_marginal) || return -Inf
    return res.log_marginal + log_hyperprior(ld.model, θ)
end

function LogDensityProblems.logdensity_and_gradient(ld::INLALogDensity,
        θ::AbstractVector)
    ℓ = LogDensityProblems.logdensity(ld, θ)
    f = θ_ -> LogDensityProblems.logdensity(ld, θ_)
    g = FiniteDiff.finite_difference_gradient(f, collect(Float64, θ))
    return ℓ, g
end

"""
    sample_conditional(model, θ, y; rng = Random.default_rng(),
                       laplace = Laplace()) -> Vector{Float64}
    sample_conditional(model, θ, y, n_samples; rng, laplace) -> Matrix{Float64}

Draw samples of the latent vector `x` from the Laplace approximation to
`p(x | θ, y)`. Paired with
[`INLALogDensity`](@ref) for INLA-within-MCMC (ADR-009) — each NUTS /
HMC step on `θ` is followed by a draw from the conditional surface at
that `θ`.

### Algorithm

1. Run [`laplace_mode`](@ref) at `θ` to get mode `x̂` and posterior
   precision `H`.
2. Draw `z ∼ N(0, I)` and solve `L'ᵀ x₀ = z` where `L` is the sparse
   Cholesky factor of `H` (so `Cov(x₀) = H⁻¹`).
3. If the component stack declared hard linear constraints
   `Cx = e`, subtract the kriging correction so the draw satisfies the
   constraint to working precision.

### Return value

- Single-sample call: `Vector{Float64}` of length `n_latent(model)`.
- `n_samples` positional argument: `Matrix{Float64}` of size
  `(n_latent(model), n_samples)`.

### Example

```julia
rng = Random.Xoshiro(42)
θ = initial_hyperparameters(model)
x = sample_conditional(model, θ, y; rng)
X = sample_conditional(model, θ, y, 100; rng)
```
"""
function sample_conditional(model::LatentGaussianModel, θ::AbstractVector, y;
        rng::Random.AbstractRNG=Random.default_rng(),
        laplace::Laplace=Laplace())
    lp = laplace_mode(model, y, θ; strategy=laplace)
    return _sample_laplace(rng, lp)
end

function sample_conditional(model::LatentGaussianModel, θ::AbstractVector, y,
        n_samples::Integer;
        rng::Random.AbstractRNG=Random.default_rng(),
        laplace::Laplace=Laplace())
    n_samples ≥ 1 || throw(ArgumentError("n_samples must be ≥ 1"))
    lp = laplace_mode(model, y, θ; strategy=laplace)
    X = Matrix{Float64}(undef, length(lp.mode), n_samples)
    for s in 1:n_samples
        @views X[:, s] .= _sample_laplace(rng, lp)
    end
    return X
end

"""
    Generic0(R; hyperprior = PCPrecision(), rankdef = 0, scale_model = false)

User-supplied structure-matrix component, precision `Q = τ · R` with
`τ = exp(θ[1])`. Direct LGM wrapper around
[`GMRFs.Generic0GMRF`](@ref).

- `R` is symmetric non-negative-definite.
- `rankdef` is the claimed dimension of `R`'s null space; when
  non-zero the caller is responsible for supplying a matching
  constraint. Unlike [`Besag`](@ref) we cannot infer the null-space
  directions generically.
- `scale_model = true` applies the Sørbye-Rue (2014) geometric-mean
  scaling to `R` at construction time. Matches R-INLA's
  `generic0 + scale.model = TRUE`.

One hyperparameter on the internal scale `θ = log(τ)`.
"""
struct Generic0{P <: AbstractHyperPrior, T <: Real} <: AbstractLatentComponent
    R::SparseMatrixCSC{T, Int}    # structure matrix, possibly scaled
    hyperprior::P
    rankdef::Int
    scale_model::Bool
end

function Generic0(R::AbstractMatrix;
                  hyperprior::AbstractHyperPrior = PCPrecision(),
                  rankdef::Integer = 0,
                  scale_model::Bool = false)
    # Delegate validation + (optional) Sørbye-Rue scaling to Generic0GMRF.
    stub = GMRFs.Generic0GMRF(R; τ = one(eltype(float(one(eltype(R))))),
                              rankdef = rankdef, scale_model = scale_model)
    Rs = stub.R
    T = eltype(Rs)
    return Generic0{typeof(hyperprior), T}(Rs, hyperprior, Int(rankdef), scale_model)
end

Base.length(c::Generic0) = size(c.R, 1)
nhyperparameters(::Generic0) = 1
initial_hyperparameters(::Generic0) = [0.0]

precision_matrix(c::Generic0, θ) = exp(θ[1]) .* c.R
log_hyperprior(c::Generic0, θ) = log_prior_density(c.hyperprior, θ[1])

# `Q = τ R`. Drop the τ-independent `½ log|R̃|_+` (intrinsic-component
# convention, applied uniformly so `Generic0` does not depend on a
# Cholesky of `R` at construction). When `rankdef = 0` we further
# include the `-½ n log(2π)` reference-measure term so the global
# formula reduces to the standard proper-Gaussian Laplace identity.
function log_normalizing_constant(c::Generic0, θ)
    n = size(c.R, 1)
    proper_dim = n - c.rankdef
    base = 0.5 * proper_dim * θ[1]
    return c.rankdef == 0 ? -0.5 * n * log(2π) + base : base
end

function gmrf(c::Generic0, θ)
    # scale_model = false since c.R is already in its post-scaling form.
    return GMRFs.Generic0GMRF(c.R; τ = exp(θ[1]), rankdef = c.rankdef,
                              scale_model = false)
end

"""
    Generic1(R; hyperprior = PCPrecision(), rankdef = 0)

Like [`Generic0`](@ref) but rescales `R` at construction so that its
largest eigenvalue is `1`. Matches the structure-matrix normalisation
used by R-INLA's `generic1` before multiplying by `τ`.

Unlike R-INLA's full `generic1`, this component does **not** introduce
the mixing parameter `β ∈ (0, 1)` — only the eigenvalue rescaling. The
β-mixing flavour is deferred; see
[packages/GMRFs.jl/plans/plan.md M2 note](packages/GMRFs.jl/plans/plan.md).

One hyperparameter on the internal scale `θ = log(τ)`.
"""
struct Generic1{P <: AbstractHyperPrior, T <: Real} <: AbstractLatentComponent
    R::SparseMatrixCSC{T, Int}      # rescaled so λ_max(R) = 1
    hyperprior::P
    rankdef::Int
    λ_max_original::T               # original λ_max(R) before rescaling
end

function Generic1(R::AbstractMatrix;
                  hyperprior::AbstractHyperPrior = PCPrecision(),
                  rankdef::Integer = 0)
    n, m = size(R)
    n == m || throw(DimensionMismatch("Generic1: R must be square, got $(n)×$(m)"))
    issymmetric(R) || throw(ArgumentError("Generic1: R must be symmetric"))
    rankdef ≥ 0 || throw(ArgumentError("Generic1: rankdef must be ≥ 0, got $rankdef"))

    # Largest eigenvalue via a dense eigendecomposition. Acceptable at
    # v0.1 scale: this matches the dense fall-back used by
    # `GMRFs._generic_scale_factor`.
    λ_max = maximum(eigvals(Symmetric(Matrix(R))))
    λ_max > 0 || throw(ArgumentError("Generic1: largest eigenvalue of R must be positive, got $λ_max"))

    T = typeof(float(λ_max))
    Rs_scaled = SparseMatrixCSC{T, Int}(R) ./ T(λ_max)
    return Generic1{typeof(hyperprior), T}(Rs_scaled, hyperprior, Int(rankdef), T(λ_max))
end

Base.length(c::Generic1) = size(c.R, 1)
nhyperparameters(::Generic1) = 1
initial_hyperparameters(::Generic1) = [0.0]

precision_matrix(c::Generic1, θ) = exp(θ[1]) .* c.R
log_hyperprior(c::Generic1, θ) = log_prior_density(c.hyperprior, θ[1])

# Same convention as `Generic0`: drop `½ log|R̃|_+` (τ-independent), and
# include `-½ n log(2π)` only for the proper case.
function log_normalizing_constant(c::Generic1, θ)
    n = size(c.R, 1)
    proper_dim = n - c.rankdef
    base = 0.5 * proper_dim * θ[1]
    return c.rankdef == 0 ? -0.5 * n * log(2π) + base : base
end

function gmrf(c::Generic1, θ)
    return GMRFs.Generic0GMRF(c.R; τ = exp(θ[1]), rankdef = c.rankdef,
                              scale_model = false)
end

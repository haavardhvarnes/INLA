"""
    LatentGaussianModel(likelihoods, components, mapping)
    LatentGaussianModel(likelihood,  components, mapping_or_A)
    LatentGaussianModel(likelihood,  component,  mapping_or_A)

Bayesian latent Gaussian model:

    y | η, θ_ℓ ∼ p(y | η, θ_ℓ),     η = mapping * x,
    x | θ     ∼ N(0, Q(θ)⁻¹),       Q(θ) = blockdiag(Q_1(θ_1), ...),
    θ         ∼ π(θ).

# Arguments

- `likelihoods::Tuple{Vararg{AbstractLikelihood}}` — one observation
  model per block. Each block applies to a contiguous slice of rows of
  `η`, with the slice ranges supplied by the `mapping`. For a
  single-likelihood model the convenience signature
  `LatentGaussianModel(ℓ, components, mapping_or_A)` accepts a scalar
  `ℓ::AbstractLikelihood` and promotes it internally to `(ℓ,)`.
- `components::Tuple{Vararg{AbstractLatentComponent}}` — latent
  components, concatenated into a single `x` of length
  `sum(length, components)`.
- `mapping::AbstractObservationMapping` — projector from the stacked
  latent vector `x` to the linear predictor `η`. The convenience
  signature `LatentGaussianModel(ℓ, components, A::AbstractMatrix)`
  wraps the matrix in [`LinearProjector`](@ref) automatically (the
  v0.1 default).

For multi-likelihood models the mapping must be a [`StackedMapping`](@ref)
whose number of blocks equals `length(likelihoods)`; the `k`-th block
of `mapping` defines the row range owned by `likelihoods[k]`.

Internally we store the latent slices (index ranges of each component
in `x`), the per-component hyperparameter slices in the full `θ`, and
the per-likelihood hyperparameter slices in the full `θ`. The full `θ`
layout is `[θ_ℓ_1, ..., θ_ℓ_K, θ_c_1, ..., θ_c_M]` — likelihood
hyperparameters first (in block order), component hyperparameters
after.
"""
struct LatentGaussianModel{L <: Tuple{Vararg{AbstractLikelihood}},
    C <: Tuple{Vararg{AbstractLatentComponent}},
    M <: AbstractObservationMapping}
    likelihoods::L
    components::C
    mapping::M
    latent_ranges::Vector{UnitRange{Int}}            # per component, in x
    θ_ranges::Vector{UnitRange{Int}}                 # per component, in θ
    likelihood_θ_ranges::Vector{UnitRange{Int}}      # per likelihood, in θ
    block_rows::Vector{UnitRange{Int}}               # per likelihood, in 1:n_obs
    n_x::Int
    n_θ::Int
end

function LatentGaussianModel(likelihoods::Tuple{Vararg{AbstractLikelihood}},
        components::Tuple{Vararg{AbstractLatentComponent}},
        mapping::AbstractObservationMapping)
    isempty(likelihoods) &&
        throw(ArgumentError("LatentGaussianModel needs at least one likelihood"))

    nc = length(components)
    latent_ranges = Vector{UnitRange{Int}}(undef, nc)
    offset = 0
    for (i, c) in enumerate(components)
        latent_ranges[i] = (offset + 1):(offset + length(c))
        offset += length(c)
    end
    n_x = offset

    ncols(mapping) == n_x ||
        throw(DimensionMismatch("mapping has ncols=$(ncols(mapping)); components span $n_x"))

    block_rows = _build_block_rows(mapping, likelihoods)
    n_obs = nrows(mapping)
    if !isempty(block_rows)
        last(last(block_rows)) == n_obs ||
            throw(DimensionMismatch("block_rows cover 1:$(last(last(block_rows))); " *
                                    "mapping has $n_obs rows"))
    end

    likelihood_θ_ranges = Vector{UnitRange{Int}}(undef, length(likelihoods))
    θ_offset = 0
    for (k, ℓ) in enumerate(likelihoods)
        nh = nhyperparameters(ℓ)
        likelihood_θ_ranges[k] = (θ_offset + 1):(θ_offset + nh)
        θ_offset += nh
    end

    θ_ranges = Vector{UnitRange{Int}}(undef, nc)
    for (i, c) in enumerate(components)
        k = nhyperparameters(c)
        θ_ranges[i] = (θ_offset + 1):(θ_offset + k)
        θ_offset += k
    end
    n_θ = θ_offset

    return LatentGaussianModel{typeof(likelihoods), typeof(components), typeof(mapping)}(
        likelihoods, components, mapping, latent_ranges, θ_ranges,
        likelihood_θ_ranges, block_rows, n_x, n_θ)
end

# Block-row layout. Single-block mappings own all rows; StackedMapping
# carries its own per-block ranges.
_build_block_rows(m::AbstractObservationMapping, ℓs::Tuple) = _single_block_rows(m, ℓs)

function _single_block_rows(m::AbstractObservationMapping, ℓs::Tuple)
    length(ℓs) == 1 ||
        throw(ArgumentError("$(typeof(m)) is single-block; received " *
                            "$(length(ℓs)) likelihoods. Use StackedMapping for " *
                            "multi-likelihood models."))
    return [1:nrows(m)]
end

function _build_block_rows(m::StackedMapping, ℓs::Tuple)
    length(ℓs) == length(m.blocks) ||
        throw(ArgumentError("StackedMapping has $(length(m.blocks)) blocks; " *
                            "received $(length(ℓs)) likelihoods"))
    return copy(m.rows)
end

# v0.1 compatibility: scalar likelihood is wrapped in a 1-tuple.
function LatentGaussianModel(likelihood::AbstractLikelihood,
        components::Tuple{Vararg{AbstractLatentComponent}},
        mapping::AbstractObservationMapping)
    LatentGaussianModel((likelihood,), components, mapping)
end

# v0.1 compatibility: AbstractMatrix wraps in LinearProjector. Existing
# `LatentGaussianModel(ℓ, components, A)` calls keep working unchanged.
function LatentGaussianModel(likelihoods::Tuple{Vararg{AbstractLikelihood}},
        components::Tuple{Vararg{AbstractLatentComponent}},
        A::AbstractMatrix)
    LatentGaussianModel(likelihoods, components, LinearProjector(A))
end

function LatentGaussianModel(likelihood::AbstractLikelihood,
        components::Tuple{Vararg{AbstractLatentComponent}},
        A::AbstractMatrix)
    LatentGaussianModel((likelihood,), components, LinearProjector(A))
end

# Convenience: single component, mapping or matrix supplied.
function LatentGaussianModel(ℓ::AbstractLikelihood, c::AbstractLatentComponent,
        mapping_or_A::Union{AbstractObservationMapping, AbstractMatrix})
    LatentGaussianModel((ℓ,), (c,), mapping_or_A)
end

function LatentGaussianModel(ℓs::Tuple{Vararg{AbstractLikelihood}},
        c::AbstractLatentComponent,
        mapping_or_A::Union{AbstractObservationMapping, AbstractMatrix})
    LatentGaussianModel(ℓs, (c,), mapping_or_A)
end

# Back-compat property accessor: `m.likelihood` resolves to the single
# block's likelihood. Errors on multi-likelihood models. The struct
# field is `likelihoods` (plural).
function Base.getproperty(m::LatentGaussianModel, name::Symbol)
    if name === :likelihood
        ℓs = getfield(m, :likelihoods)
        length(ℓs) == 1 ||
            error("`m.likelihood` is ambiguous on multi-likelihood models with " *
                  "$(length(ℓs)) blocks; use `m.likelihoods[k]` instead.")
        return ℓs[1]
    end
    return getfield(m, name)
end

"""
    n_latent(m) -> Int

Length of the stacked latent vector `x`.
"""
n_latent(m::LatentGaussianModel) = m.n_x

"""
    n_observations(m) -> Int

Number of observation rows. Equal to `nrows(m.mapping)`.
"""
n_observations(m::LatentGaussianModel) = nrows(m.mapping)

"""
    n_hyperparameters(m) -> Int

Number of hyperparameters (all likelihoods + all components).
"""
n_hyperparameters(m::LatentGaussianModel) = m.n_θ

"""
    n_likelihoods(m) -> Int

Number of likelihood blocks. Equal to `length(m.likelihoods)`.
"""
n_likelihoods(m::LatentGaussianModel) = length(m.likelihoods)

"""
    n_likelihood_hyperparameters(m) -> Int

Total number of likelihood-attached hyperparameters across all blocks.
"""
n_likelihood_hyperparameters(m::LatentGaussianModel) = isempty(m.likelihood_θ_ranges) ? 0 :
                                                       last(last(m.likelihood_θ_ranges))

"""
    initial_hyperparameters(m::LatentGaussianModel) -> Vector{Float64}

Stack the initial internal-scale hyperparameters of all likelihoods and
components, in the canonical `[θ_ℓ; θ_c]` order.
"""
function initial_hyperparameters(m::LatentGaussianModel)
    θ0 = Float64[]
    for ℓ in m.likelihoods
        append!(θ0, initial_hyperparameters(ℓ))
    end
    for c in m.components
        append!(θ0, initial_hyperparameters(c))
    end
    return θ0
end

"""
    joint_precision(m::LatentGaussianModel, θ) -> SparseMatrixCSC

Block-diagonal latent precision `Q(θ) = blockdiag(Q_1, ..., Q_K)` where
`Q_i = precision_matrix(components[i], θ[θ_ranges[i]])`.
"""
function joint_precision(m::LatentGaussianModel, θ)
    blocks = SparseMatrixCSC{Float64, Int}[]
    for (i, c) in enumerate(m.components)
        θ_i = θ[m.θ_ranges[i]]
        Qi = precision_matrix(c, θ_i)
        push!(blocks, SparseMatrixCSC{Float64, Int}(Qi))
    end
    return blockdiag(blocks...)
end

"""
    joint_prior_mean(m::LatentGaussianModel, θ) -> Vector

Stacked latent prior mean `μ(θ) = [μ_1(θ_1); ...; μ_K(θ_K)]` where
`μ_i = prior_mean(components[i], θ[θ_ranges[i]])`. The default per
component is zeros, so this returns a zero vector for every model
that does not use measurement-error or other shifted-prior components.

ADR-023 promotes `prior_mean` from documented-optional to load-bearing
in the inner Newton loop: the latent quadratic is
`½ (x - μ)' Q(θ) (x - μ)`, not `½ x' Q x`.
"""
function joint_prior_mean(m::LatentGaussianModel, θ)
    μ = zeros(Float64, m.n_x)
    for (i, c) in enumerate(m.components)
        θ_i = θ[m.θ_ranges[i]]
        μi = prior_mean(c, θ_i)
        @views μ[m.latent_ranges[i]] .= μi
    end
    return μ
end

"""
    log_hyperprior(m::LatentGaussianModel, θ) -> Real

Sum of log-priors on the internal scale for all likelihoods and
components.
"""
function log_hyperprior(m::LatentGaussianModel, θ)
    lp = zero(eltype(θ))
    for (k, ℓ) in enumerate(m.likelihoods)
        lp += log_hyperprior(ℓ, θ[m.likelihood_θ_ranges[k]])
    end
    for (i, c) in enumerate(m.components)
        lp += log_hyperprior(c, θ[m.θ_ranges[i]])
    end
    return lp
end

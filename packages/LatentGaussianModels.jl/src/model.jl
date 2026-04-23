"""
    LatentGaussianModel(likelihood, components, A; component_projectors = nothing)

Bayesian latent Gaussian model:

    y | η, θ_ℓ ∼ p(y | η, θ_ℓ),     η = A x,
    x | θ     ∼ N(0, Q(θ)⁻¹),       Q(θ) = blockdiag(Q_1(θ_1), ...),
    θ         ∼ π(θ).

- `likelihood::AbstractLikelihood` — observation model. Carries any
  observation-level hyperparameters (e.g. Gaussian `τ`).
- `components::Tuple{Vararg{AbstractLatentComponent}}` — latent
  components, concatenated into a single `x` of length
  `sum(length, components)`.
- `A::SparseMatrixCSC` — projector from the stacked latent vector to
  the linear predictor `η`, shape `(n_obs, n_x)`. The caller is
  responsible for building `A`; for models built component-by-component
  use `build_projector`.

Internally we store the latent slices (index ranges of each component
in `x`) and hyperparameter slices (ranges of each component's `θ_i`
in the full θ vector which also leads with the likelihood's
hyperparameters).
"""
struct LatentGaussianModel{L <: AbstractLikelihood,
                           C <: Tuple{Vararg{AbstractLatentComponent}},
                           A <: AbstractMatrix}
    likelihood::L
    components::C
    A::A
    latent_ranges::Vector{UnitRange{Int}}   # range of each component in x
    θ_ranges::Vector{UnitRange{Int}}        # range of each component in θ (after likelihood θ)
    n_x::Int
    n_θ::Int
end

function LatentGaussianModel(likelihood::AbstractLikelihood,
                             components::Tuple{Vararg{AbstractLatentComponent}},
                             A::AbstractMatrix)
    nc = length(components)
    latent_ranges = Vector{UnitRange{Int}}(undef, nc)
    offset = 0
    for (i, c) in enumerate(components)
        latent_ranges[i] = (offset + 1):(offset + length(c))
        offset += length(c)
    end
    n_x = offset

    size(A, 2) == n_x ||
        throw(DimensionMismatch("projector A has $(size(A,2)) columns; components span $n_x"))

    θ_ranges = Vector{UnitRange{Int}}(undef, nc)
    θ_offset = nhyperparameters(likelihood)
    for (i, c) in enumerate(components)
        k = nhyperparameters(c)
        θ_ranges[i] = (θ_offset + 1):(θ_offset + k)
        θ_offset += k
    end
    n_θ = θ_offset

    return LatentGaussianModel{typeof(likelihood), typeof(components), typeof(A)}(
        likelihood, components, A, latent_ranges, θ_ranges, n_x, n_θ)
end

# Convenience: single component, projector supplied.
LatentGaussianModel(ℓ::AbstractLikelihood, c::AbstractLatentComponent, A::AbstractMatrix) =
    LatentGaussianModel(ℓ, (c,), A)

"""
    n_latent(m) -> Int

Length of the stacked latent vector `x`.
"""
n_latent(m::LatentGaussianModel) = m.n_x

"""
    n_hyperparameters(m) -> Int

Number of hyperparameters (likelihood + all components).
"""
n_hyperparameters(m::LatentGaussianModel) = m.n_θ

"""
    initial_hyperparameters(m::LatentGaussianModel) -> Vector{Float64}

Stack the initial internal-scale hyperparameters of the likelihood and
each component.
"""
function initial_hyperparameters(m::LatentGaussianModel)
    θ0 = Float64[]
    append!(θ0, initial_hyperparameters(m.likelihood))
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
    log_hyperprior(m::LatentGaussianModel, θ) -> Real

Sum of log-priors on the internal scale for the likelihood and all
components.
"""
function log_hyperprior(m::LatentGaussianModel, θ)
    n_ℓ = nhyperparameters(m.likelihood)
    lp = log_hyperprior(m.likelihood, θ[1:n_ℓ])
    for (i, c) in enumerate(m.components)
        lp += log_hyperprior(c, θ[m.θ_ranges[i]])
    end
    return lp
end

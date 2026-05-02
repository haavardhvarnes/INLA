# Per-block likelihood routing for `LatentGaussianModel`.
#
# A multi-likelihood model carries `likelihoods::Tuple{Vararg{<:AbstractLikelihood}}`
# and a [`StackedMapping`](@ref) whose row blocks identify which observations
# belong to each likelihood. The helpers in this file evaluate joint
# log-density and η-derivatives by routing each block through its own
# `AbstractLikelihood` interface and concatenating the per-block results.
#
# Hot-path invariants:
#
# - For a single-likelihood model (`length(m.likelihoods) == 1`) every
#   helper bottoms out to one call against the existing
#   `log_density(ℓ, y, η, θ_ℓ)` / `∇_η_log_density(...)` interface — no
#   allocations beyond what the per-likelihood method already does.
# - The block layout is fixed at construction (cached on the model), so
#   the hot path never queries the mapping for row ranges.
# - Per-block dispatches resolve at compile time when `likelihoods` is a
#   concrete `Tuple` of <=3 distinct types (Julia's union-splitting
#   threshold). Multi-block models with >3 distinct likelihood types
#   fall through to dynamic dispatch — fine for now; revisit if a
#   real fixture needs more blocks than that.

"""
    joint_log_density(m::LatentGaussianModel, y, η, θ) -> Real

Sum of per-block likelihood log-densities, routed through `m.likelihoods`
and `m.block_rows`. Reduces to a single `log_density(m.likelihoods[1],
y, η, θ_ℓ)` call when there is exactly one block.
"""
function joint_log_density(m::LatentGaussianModel,
        y::AbstractVector, η::AbstractVector, θ::AbstractVector)
    s = zero(eltype(η))
    for k in eachindex(m.likelihoods)
        ℓ_k = m.likelihoods[k]
        rows_k = m.block_rows[k]
        θ_k = view(θ, m.likelihood_θ_ranges[k])
        s += log_density(ℓ_k, view(y, rows_k), view(η, rows_k), θ_k)
    end
    return s
end

"""
    joint_∇_η_log_density(m::LatentGaussianModel, y, η, θ) -> AbstractVector

Length-`n_obs` gradient of `Σ_k log p(y_k | η_k, θ_ℓ_k)` w.r.t. `η`,
assembled by routing each block through its `∇_η_log_density`.
"""
function joint_∇_η_log_density(m::LatentGaussianModel,
        y::AbstractVector, η::AbstractVector, θ::AbstractVector)
    if length(m.likelihoods) == 1
        ℓ = m.likelihoods[1]
        return ∇_η_log_density(ℓ, y, η, view(θ, m.likelihood_θ_ranges[1]))
    end
    out = Vector{promote_type(eltype(η), Float64)}(undef, length(η))
    for k in eachindex(m.likelihoods)
        ℓ_k = m.likelihoods[k]
        rows_k = m.block_rows[k]
        θ_k = view(θ, m.likelihood_θ_ranges[k])
        out[rows_k] .= ∇_η_log_density(ℓ_k, view(y, rows_k), view(η, rows_k), θ_k)
    end
    return out
end

"""
    joint_∇²_η_log_density(m::LatentGaussianModel, y, η, θ) -> AbstractVector

Length-`n_obs` diagonal of the `η`-Hessian. Same routing pattern as
[`joint_∇_η_log_density`](@ref).
"""
function joint_∇²_η_log_density(m::LatentGaussianModel,
        y::AbstractVector, η::AbstractVector, θ::AbstractVector)
    if length(m.likelihoods) == 1
        ℓ = m.likelihoods[1]
        return ∇²_η_log_density(ℓ, y, η, view(θ, m.likelihood_θ_ranges[1]))
    end
    out = Vector{promote_type(eltype(η), Float64)}(undef, length(η))
    for k in eachindex(m.likelihoods)
        ℓ_k = m.likelihoods[k]
        rows_k = m.block_rows[k]
        θ_k = view(θ, m.likelihood_θ_ranges[k])
        out[rows_k] .= ∇²_η_log_density(ℓ_k, view(y, rows_k), view(η, rows_k), θ_k)
    end
    return out
end

"""
    joint_∇³_η_log_density(m::LatentGaussianModel, y, η, θ) -> AbstractVector

Length-`n_obs` diagonal of the third-derivative tensor of the joint
likelihood w.r.t. `η`. Used by simplified-Laplace mean-shift and
posterior-marginal skewness.
"""
function joint_∇³_η_log_density(m::LatentGaussianModel,
        y::AbstractVector, η::AbstractVector, θ::AbstractVector)
    if length(m.likelihoods) == 1
        ℓ = m.likelihoods[1]
        return ∇³_η_log_density(ℓ, y, η, view(θ, m.likelihood_θ_ranges[1]))
    end
    out = Vector{promote_type(eltype(η), Float64)}(undef, length(η))
    for k in eachindex(m.likelihoods)
        ℓ_k = m.likelihoods[k]
        rows_k = m.block_rows[k]
        θ_k = view(θ, m.likelihood_θ_ranges[k])
        out[rows_k] .= ∇³_η_log_density(ℓ_k, view(y, rows_k), view(η, rows_k), θ_k)
    end
    return out
end

"""
    joint_pointwise_log_density(m::LatentGaussianModel, y, η, θ) -> AbstractVector

Length-`n_obs` per-observation `log p(y_i | η_i, θ_ℓ)`. Sums to
[`joint_log_density`](@ref).
"""
function joint_pointwise_log_density(m::LatentGaussianModel,
        y::AbstractVector, η::AbstractVector, θ::AbstractVector)
    if length(m.likelihoods) == 1
        ℓ = m.likelihoods[1]
        return pointwise_log_density(ℓ, y, η, view(θ, m.likelihood_θ_ranges[1]))
    end
    out = Vector{promote_type(eltype(η), Float64)}(undef, length(η))
    for k in eachindex(m.likelihoods)
        ℓ_k = m.likelihoods[k]
        rows_k = m.block_rows[k]
        θ_k = view(θ, m.likelihood_θ_ranges[k])
        out[rows_k] .= pointwise_log_density(ℓ_k, view(y, rows_k), view(η, rows_k), θ_k)
    end
    return out
end

"""
    joint_apply_copy_contributions!(η, m::LatentGaussianModel, x, θ) -> η

Apply each likelihood's [`add_copy_contributions!`](@ref) hook to its
own slice of `η`, in place, and return `η`. Called after every
`η = mapping * x` evaluation in the inner Newton loop and posterior
sampler so that `Copy`-scaled latent shares fold into the linear
predictor before likelihood derivatives are computed.

The default `add_copy_contributions!` is a no-op, so single-likelihood
models without copies pay only the per-block dispatch loop — concrete
`Tuple{Vararg{<:AbstractLikelihood}}` likelihoods union-split at
compile time, so the no-op specialisation lowers to a no-op.
"""
function joint_apply_copy_contributions!(η::AbstractVector,
        m::LatentGaussianModel,
        x::AbstractVector,
        θ::AbstractVector)
    for k in eachindex(m.likelihoods)
        ℓ_k = m.likelihoods[k]
        rows_k = m.block_rows[k]
        θ_k = view(θ, m.likelihood_θ_ranges[k])
        add_copy_contributions!(view(η, rows_k), ℓ_k, x, θ_k)
    end
    return η
end

"""
    joint_pointwise_cdf(m::LatentGaussianModel, y, η, θ) -> AbstractVector

Length-`n_obs` per-observation predictive CDF `F(y_i | η_i, θ_ℓ)`. Used
by PIT diagnostics. Each block must implement
[`pointwise_cdf`](@ref) on its likelihood.
"""
function joint_pointwise_cdf(m::LatentGaussianModel,
        y::AbstractVector, η::AbstractVector, θ::AbstractVector)
    if length(m.likelihoods) == 1
        ℓ = m.likelihoods[1]
        return pointwise_cdf(ℓ, y, η, view(θ, m.likelihood_θ_ranges[1]))
    end
    out = Vector{promote_type(eltype(η), Float64)}(undef, length(η))
    for k in eachindex(m.likelihoods)
        ℓ_k = m.likelihoods[k]
        rows_k = m.block_rows[k]
        θ_k = view(θ, m.likelihood_θ_ranges[k])
        out[rows_k] .= pointwise_cdf(ℓ_k, view(y, rows_k), view(η, rows_k), θ_k)
    end
    return out
end

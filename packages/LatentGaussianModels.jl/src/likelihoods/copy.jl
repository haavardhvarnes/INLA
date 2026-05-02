# `Copy` — share a latent component into another linear-predictor block.
#
# ADR-021: β (the scaling) is a hyperparameter of the *receiving*
# likelihood, not of the projection mapping. The user wraps the
# receiving likelihood in `CopyTargetLikelihood(base, copies)`; the
# wrapper folds `β * x[source_indices]` into η via the
# `add_copy_contributions!` hook from PR-3a, then forwards every
# likelihood method to `base`.
#
# All η-derivatives forward unchanged: the Copy contribution is
# additive in η, so `∇_η log p`, `∇²_η log p`, `∇³_η log p`, and
# `pointwise_log_density` are unaffected by the wrapping (they evaluate
# at the *adjusted* η that already contains the copy term). The wrapper
# only needs to plumb β through `nhyperparameters`,
# `initial_hyperparameters`, and `log_hyperprior`.

"""
    Copy(source_indices::UnitRange{Int};
         β_prior = GaussianPrior(1.0, 0.1),
         β_init  = 1.0,
         fixed   = false)

Specifier for one Copy contribution attached to a
[`CopyTargetLikelihood`](@ref). The receiving likelihood adds
`β * x[source_indices][i]` to its η-row `i` (one-to-one row mapping
between observations and source-component slots).

`source_indices` is the range of the source component inside the
model's stacked latent vector `x`. For the common case where the
source is the `j`-th declared component, that is
`m.latent_ranges[j]` after model construction; users can compute it
inline with `cumsum(length, components_before)` or the helper
[`component_range`](@ref) once the model exists.

`β_prior` is on the unconstrained user scale (β = β-internal). Default
is R-INLA's `gaussian(mean=1, prec=100)` — `GaussianPrior(1.0, 0.1)`
in this package's parameterisation.

If `fixed = true`, β is held at `β_init` for the duration of inference
(no hyperparameter slot is allocated). Useful for the closed-form
regression test that pins β to 1.0 to recover the unscaled-share
result, and for sensitivity studies that want to ablate β.
"""
struct Copy{P}
    source_indices::UnitRange{Int}
    β_prior::P
    β_init::Float64
    fixed::Bool
end

function Copy(source_indices::AbstractUnitRange{<:Integer};
        β_prior=GaussianPrior(1.0, 0.1),
        β_init::Real=1.0,
        fixed::Bool=false)
    return Copy{typeof(β_prior)}(UnitRange{Int}(source_indices),
        β_prior, Float64(β_init), fixed)
end

"""
    CopyTargetLikelihood(base::AbstractLikelihood, copies::Tuple{Vararg{Copy}})
    CopyTargetLikelihood(base, copy::Copy)

Wrap `base` with one or more [`Copy`](@ref) contributions. Forwards
every required `AbstractLikelihood` method to `base` after the
inference loop folds the Copy contributions into η via
[`add_copy_contributions!`](@ref).

Each non-fixed `Copy` adds one β slot to the wrapper's hyperparameter
vector, in the order they are supplied. The full layout is
`[θ_base..., β_1, β_2, ...]`, matching how the model constructor
accounts for `nhyperparameters(ℓ)`.

# Example

```julia
# Two-block joint Gaussian + Poisson model where the Poisson arm
# *copies* the Gaussian arm's IID random effect with an estimated β.
ℓ_g = GaussianLikelihood()
ℓ_p_with_copy = CopyTargetLikelihood(
    PoissonLikelihood(; E = fill(1.0, n)),
    (Copy(2:(n + 1); β_prior = GaussianPrior(1.0, 0.5)),))
model = LatentGaussianModel(
    (ℓ_g, ℓ_p_with_copy),
    (Intercept(), IID(n)),
    StackedMapping(...))
```
"""
struct CopyTargetLikelihood{B <: AbstractLikelihood, T <: Tuple{Vararg{Copy}}} <:
       AbstractLikelihood
    base::B
    copies::T

    function CopyTargetLikelihood{B, T}(
            base::B, copies::T) where {
            B <: AbstractLikelihood, T <: Tuple{Vararg{Copy}}}
        isempty(copies) &&
            throw(ArgumentError("CopyTargetLikelihood needs at least one Copy"))
        return new{B, T}(base, copies)
    end
end

function CopyTargetLikelihood(base::AbstractLikelihood, copies::Tuple{Vararg{Copy}})
    return CopyTargetLikelihood{typeof(base), typeof(copies)}(base, copies)
end

function CopyTargetLikelihood(base::AbstractLikelihood, copy::Copy)
    return CopyTargetLikelihood(base, (copy,))
end

# --- forwarding to base -------------------------------------------------
# The Copy contributions enter η additively and are folded in by
# `add_copy_contributions!` before any of these are called. So every
# η-derivative evaluates correctly at the adjusted η without further
# modification.

link(ℓ::CopyTargetLikelihood) = link(ℓ.base)

function log_density(ℓ::CopyTargetLikelihood, y, η, θ_ℓ)
    return log_density(ℓ.base, y, η, _base_θ(ℓ, θ_ℓ))
end

function ∇_η_log_density(ℓ::CopyTargetLikelihood, y, η, θ_ℓ)
    return ∇_η_log_density(ℓ.base, y, η, _base_θ(ℓ, θ_ℓ))
end

function ∇²_η_log_density(ℓ::CopyTargetLikelihood, y, η, θ_ℓ)
    return ∇²_η_log_density(ℓ.base, y, η, _base_θ(ℓ, θ_ℓ))
end

function ∇³_η_log_density(ℓ::CopyTargetLikelihood, y, η, θ_ℓ)
    return ∇³_η_log_density(ℓ.base, y, η, _base_θ(ℓ, θ_ℓ))
end

function pointwise_log_density(ℓ::CopyTargetLikelihood, y, η, θ_ℓ)
    return pointwise_log_density(ℓ.base, y, η, _base_θ(ℓ, θ_ℓ))
end

function pointwise_cdf(ℓ::CopyTargetLikelihood, y, η, θ_ℓ)
    return pointwise_cdf(ℓ.base, y, η, _base_θ(ℓ, θ_ℓ))
end

# --- hyperparameter accounting ------------------------------------------

function nhyperparameters(ℓ::CopyTargetLikelihood)
    return nhyperparameters(ℓ.base) + _n_free_copies(ℓ)
end

function initial_hyperparameters(ℓ::CopyTargetLikelihood)
    base_init = collect(initial_hyperparameters(ℓ.base))
    for c in ℓ.copies
        c.fixed && continue
        push!(base_init, c.β_init)
    end
    return base_init
end

function log_hyperprior(ℓ::CopyTargetLikelihood, θ_ℓ)
    n_base = nhyperparameters(ℓ.base)
    lp = log_hyperprior(ℓ.base, view(θ_ℓ, 1:n_base))
    j = n_base
    for c in ℓ.copies
        c.fixed && continue
        j += 1
        lp += log_prior_density(c.β_prior, θ_ℓ[j])
    end
    return lp
end

# --- the hook -----------------------------------------------------------

function add_copy_contributions!(η::AbstractVector,
        ℓ::CopyTargetLikelihood,
        x::AbstractVector,
        θ_ℓ)
    n_base = nhyperparameters(ℓ.base)
    free_idx = 0
    for c in ℓ.copies
        if c.fixed
            β = c.β_init
        else
            free_idx += 1
            β = θ_ℓ[n_base + free_idx]
        end
        # The mapping has already populated η for every observation in
        # this block. Add `β * x[source_indices]` row-by-row. We assume
        # `length(c.source_indices) == length(η)` — the constructor
        # validates this lazily at the first call.
        length(c.source_indices) == length(η) ||
            throw(DimensionMismatch("Copy source span $(length(c.source_indices)) " *
                                    "does not match η-block length $(length(η))"))
        @inbounds for i in eachindex(η)
            η[i] += β * x[c.source_indices[i]]
        end
    end
    return η
end

# --- Jacobian augmentation ---------------------------------------------
# The η-hook above handles the forward pass `η += β * x[source]`. The
# inner Newton step also needs `J = dη/dx` to include the same `β` rows
# so the gradient `Jᵀ ∇η` and Hessian `Q + Jᵀ D J` are consistent.
# `joint_effective_jacobian` (joint_likelihood.jl) calls these hooks per
# block and folds the contributions in.

_has_copy_contribution(::CopyTargetLikelihood) = true

function _accumulate_copy_jacobian!(Is::AbstractVector, Js::AbstractVector,
        Vs::AbstractVector, ℓ::CopyTargetLikelihood,
        block_rows::AbstractVector, θ_ℓ)
    n_base = nhyperparameters(ℓ.base)
    free_idx = 0
    for c in ℓ.copies
        if c.fixed
            β = c.β_init
        else
            free_idx += 1
            β = θ_ℓ[n_base + free_idx]
        end
        length(c.source_indices) == length(block_rows) ||
            throw(DimensionMismatch("Copy source span $(length(c.source_indices)) " *
                                    "does not match block-row length $(length(block_rows))"))
        @inbounds for i in eachindex(block_rows)
            push!(Is, block_rows[i])
            push!(Js, c.source_indices[i])
            push!(Vs, β)
        end
    end
    return nothing
end

# --- helpers ------------------------------------------------------------

# Slice of θ_ℓ that belongs to the base likelihood (excluding β slots).
_base_θ(ℓ::CopyTargetLikelihood, θ_ℓ) = view(θ_ℓ, 1:nhyperparameters(ℓ.base))

_n_free_copies(ℓ::CopyTargetLikelihood) = count(c -> !c.fixed, ℓ.copies)

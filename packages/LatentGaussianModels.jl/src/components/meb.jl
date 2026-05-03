"""
    MEB(values; scale = ones(length(values)),
        τ_u_prior = GammaPrecision(1.0, 1.0e-4),
        τ_u_init = log(1000.0))

R-INLA's `model = "meb"` — Berkson measurement-error component
(ADR-023). The latent block `x` is one slot per supplied `values` entry
with prior

    x ~ N(values, (τ_u · diag(scale))⁻¹).

The Berkson tie `x_i = w_i + u_i`, `u_i ~ N(0, (τ_u s_i)⁻¹)` has been
marginalised so that the latent prior carries the observed (proxy) `w`
as a θ-constant prior mean.

# Arguments

- `values`: length-`n` vector of per-slot prior means (the observed
  proxy `w`, deduplicated by the caller if appropriate).
- `scale`: length-`n` per-slot diagonal scaling. Matches R-INLA's
  `scale = ...` argument. Default is all-ones.
- `τ_u_prior`: scalar prior on `θ[1] = log τ_u`. Default
  `GammaPrecision(1.0, 1.0e-4)` (matches R-INLA's `loggamma(1, 1e-4)`).
- `τ_u_init`: initial value for `θ[1]`. Default `log(1000.0)`, matching
  R-INLA's `initial = log(1000)`.

# Hyperparameters

| slot | name | internal scale | default prior | default initial |
|---|---|---|---|---|
| `θ[1]` | `log τ_u` | `log` | `GammaPrecision(1.0, 1.0e-4)` | `log(1000)` |

The β scaling that multiplies `x` before it lands in `η` lives on the
*receiving* likelihood as a [`Copy`](@ref) (per ADR-021/ADR-023), not
on the component. The R-INLA β default is
`GaussianPrior(1.0, 0.001)` on the user scale; users attach it as

```julia
c = MEB(w)
m = LatentGaussianModel(...)        # `c` placed in the component tuple
range_c = component_range(m, c_idx) # 1-indexed position in m.components
β_copy = Copy(range_c; β_prior = GaussianPrior(1.0, 0.001), β_init = 1.0)
target = CopyTargetLikelihood(receiving_likelihood, β_copy)
```

# Note on `gmrf(c, θ)`

The `gmrf(c::MEB, θ)` accessor returns a [`GMRFs.Generic0GMRF`](@ref)
carrying *only* the precision; the non-zero prior mean is exposed
separately via [`prior_mean(c, θ)`](@ref). Inference inside the LGM
package reads `prior_mean(c, θ)` through
[`joint_prior_mean`](@ref) and is therefore correct regardless of how
`gmrf` represents the mean. Direct `rand(rng, gmrf(c, θ))` calls draw
from the centred prior — add `prior_mean(c, θ)` if you want the
shifted draw.
"""
struct MEB{P <: AbstractHyperPrior} <: AbstractLatentComponent
    values::Vector{Float64}        # per-slot prior mean (Berkson w)
    scale::Vector{Float64}         # per-slot diagonal scale (D = diag(scale))
    τ_u_prior::P
    τ_u_init::Float64
end

function MEB(values::AbstractVector{<:Real};
        scale::AbstractVector{<:Real}=ones(Float64, length(values)),
        τ_u_prior::AbstractHyperPrior=GammaPrecision(1.0, 1.0e-4),
        τ_u_init::Real=log(1000.0))
    n = length(values)
    n > 0 || throw(ArgumentError("MEB: values must be non-empty"))
    length(scale) == n ||
        throw(DimensionMismatch("MEB: scale has length $(length(scale)); " *
                                "must equal length(values) = $n"))
    all(>(0), scale) ||
        throw(ArgumentError("MEB: scale entries must be > 0"))
    return MEB{typeof(τ_u_prior)}(collect(Float64, values),
        collect(Float64, scale),
        τ_u_prior, Float64(τ_u_init))
end

Base.length(c::MEB) = length(c.values)
nhyperparameters(::MEB) = 1
initial_hyperparameters(c::MEB) = [c.τ_u_init]

function precision_matrix(c::MEB, θ)
    τ_u = exp(θ[1])
    return spdiagm(0 => τ_u .* c.scale)
end

prior_mean(c::MEB, θ) = copy(c.values)

log_hyperprior(c::MEB, θ) = log_prior_density(c.τ_u_prior, θ[1])

# Proper N(values, (τ_u · diag(scale))⁻¹) prior on `n` independent slots.
# log|Q| = log|τ_u D| = n log τ_u + Σ log scale_i, so
#
#   ½ log|Q|_+ = ½ n log τ_u + ½ Σ log scale_i.
#
# Following the F_GENERIC0 / F_BYM2 convention shared with the rest of
# the package, the user-independent `½ Σ log scale_i` term is dropped —
# this keeps Julia's `mlik` aligned with R-INLA's "up to a constant"
# form.
function log_normalizing_constant(c::MEB, θ)
    n = length(c)
    return -0.5 * n * log(2π) + 0.5 * n * θ[1]
end

function gmrf(c::MEB, θ)
    n = length(c)
    τ_u = exp(θ[1])
    D = spdiagm(0 => c.scale)
    return GMRFs.Generic0GMRF(D; τ=τ_u, rankdef=0)
end

"""
    MEC(values; scale = ones(length(values)),
        τ_u_prior = GammaPrecision(1.0, 1.0e-4),
        μ_x_prior = GaussianPrior(0.0, 1.0e-4),
        τ_x_prior = GammaPrecision(1.0, 1.0e4),
        τ_u_init  = log(10000.0),
        μ_x_init  = 0.0,
        τ_x_init  = -log(10000.0),
        fix_τ_u::Bool = true,
        fix_μ_x::Bool = true,
        fix_τ_x::Bool = true)

R-INLA's `model = "mec"` — Classical measurement-error component
(ADR-023). The latent block `x` is one slot per supplied `values` entry
with prior `x ~ N(μ_x · 1, (τ_x I)⁻¹)` and an observed proxy
`w | x ~ N(x, (τ_u D)⁻¹)` with `D = diag(scale)`. Gaussian conjugacy
absorbs the Berkson tie into the prior:

    Q̂(θ) = τ_x I + τ_u D
    μ̂(θ) = Q̂⁻¹ · (τ_x μ_x 1 + τ_u D · values)

so the LGM-level latent prior is `x ~ N(μ̂(θ), Q̂(θ)⁻¹)`. Unlike MEB's
θ-constant mean, MEC's `prior_mean(c, θ)` depends on θ through
`(τ_u, μ_x, τ_x)`.

# Arguments

- `values`: length-`n` vector of observed proxy `w` (deduplicated by
  the caller if appropriate).
- `scale`: length-`n` per-slot diagonal scaling. Default all-ones.
- `τ_u_prior`, `μ_x_prior`, `τ_x_prior`: scalar priors. Defaults match
  R-INLA's `mec.tex`.
- `τ_u_init`, `μ_x_init`, `τ_x_init`: initial values. Defaults match
  R-INLA.
- `fix_τ_u`, `fix_μ_x`, `fix_τ_x`: toggle whether each slot is
  estimated. R-INLA's default is `fix_*  = true` for all three —
  the model degrades to plain regression unless the user opts in.

# Hyperparameters

Per ADR-023, the component carries up to three internal slots in
canonical order `(log τ_u, μ_x, log τ_x)`. Fixed slots are excluded
from the θ vector and held at their `*_init` values. The β scaling
that multiplies `x` before it lands in `η` lives on the *receiving*
likelihood as a [`Copy`](@ref) (per ADR-021/ADR-023), not on the
component. The R-INLA β default is `GaussianPrior(1.0, 0.001)` on the
user scale; users attach it as

```julia
c = MEC(w)
m = LatentGaussianModel(...)        # `c` placed in the component tuple
range_c = component_range(m, c_idx) # 1-indexed position in m.components
β_copy = Copy(range_c; β_prior = GaussianPrior(1.0, 0.001), β_init = 1.0)
target = CopyTargetLikelihood(receiving_likelihood, β_copy)
```

# Note on `gmrf(c, θ)`

`gmrf(c::MEC, θ)` returns a [`GMRFs.Generic0GMRF`](@ref) carrying
*only* the precision `Q̂(θ)`; the non-zero, θ-dependent prior mean is
exposed separately via [`prior_mean(c, θ)`](@ref). LGM inference reads
`prior_mean(c, θ)` through [`joint_prior_mean`](@ref) and is correct
regardless of how `gmrf` represents the mean.
"""
struct MEC{Pu <: AbstractHyperPrior,
    Pmu <: AbstractHyperPrior,
    Px <: AbstractHyperPrior} <: AbstractLatentComponent
    values::Vector{Float64}        # observed proxy w (per slot)
    scale::Vector{Float64}         # diagonal scale (D = diag(scale))
    τ_u_prior::Pu
    μ_x_prior::Pmu
    τ_x_prior::Px
    τ_u_init::Float64
    μ_x_init::Float64
    τ_x_init::Float64
    fix_τ_u::Bool
    fix_μ_x::Bool
    fix_τ_x::Bool
end

function MEC(values::AbstractVector{<:Real};
        scale::AbstractVector{<:Real}=ones(Float64, length(values)),
        τ_u_prior::AbstractHyperPrior=GammaPrecision(1.0, 1.0e-4),
        μ_x_prior::AbstractHyperPrior=GaussianPrior(0.0, 1.0e-4),
        τ_x_prior::AbstractHyperPrior=GammaPrecision(1.0, 1.0e4),
        τ_u_init::Real=log(10000.0),
        μ_x_init::Real=0.0,
        τ_x_init::Real=-log(10000.0),
        fix_τ_u::Bool=true,
        fix_μ_x::Bool=true,
        fix_τ_x::Bool=true)
    n = length(values)
    n > 0 || throw(ArgumentError("MEC: values must be non-empty"))
    length(scale) == n ||
        throw(DimensionMismatch("MEC: scale has length $(length(scale)); " *
                                "must equal length(values) = $n"))
    all(>(0), scale) ||
        throw(ArgumentError("MEC: scale entries must be > 0"))
    return MEC{typeof(τ_u_prior), typeof(μ_x_prior), typeof(τ_x_prior)}(
        collect(Float64, values), collect(Float64, scale),
        τ_u_prior, μ_x_prior, τ_x_prior,
        Float64(τ_u_init), Float64(μ_x_init), Float64(τ_x_init),
        fix_τ_u, fix_μ_x, fix_τ_x)
end

Base.length(c::MEC) = length(c.values)

function nhyperparameters(c::MEC)
    n = 0
    c.fix_τ_u || (n += 1)
    c.fix_μ_x || (n += 1)
    c.fix_τ_x || (n += 1)
    return n
end

function initial_hyperparameters(c::MEC)
    θ0 = Float64[]
    c.fix_τ_u || push!(θ0, c.τ_u_init)
    c.fix_μ_x || push!(θ0, c.μ_x_init)
    c.fix_τ_x || push!(θ0, c.τ_x_init)
    return θ0
end

# Resolve `(τ_u, μ_x, τ_x)` from the internal θ vector + the fixed
# `*_init` values. Order in θ matches `initial_hyperparameters`:
# log τ_u (if free), μ_x (if free), log τ_x (if free).
function _mec_unpack(c::MEC, θ)
    j = 0
    log_τ_u = if c.fix_τ_u
        c.τ_u_init
    else
        j += 1
        θ[j]
    end
    μ_x = if c.fix_μ_x
        c.μ_x_init
    else
        j += 1
        θ[j]
    end
    log_τ_x = if c.fix_τ_x
        c.τ_x_init
    else
        j += 1
        θ[j]
    end
    return exp(log_τ_u), μ_x, exp(log_τ_x)
end

function precision_matrix(c::MEC, θ)
    τ_u, _, τ_x = _mec_unpack(c, θ)
    return spdiagm(0 => τ_x .+ τ_u .* c.scale)
end

# Conjugate-Gaussian posterior mean of `x` given the Berkson tie
# `w | x ~ N(x, (τ_u D)⁻¹)` and prior `x ~ N(μ_x · 1, (τ_x I)⁻¹)`:
#   μ̂_i = (τ_x μ_x + τ_u s_i w_i) / (τ_x + τ_u s_i).
function prior_mean(c::MEC, θ)
    τ_u, μ_x, τ_x = _mec_unpack(c, θ)
    diag_Q = τ_x .+ τ_u .* c.scale
    rhs = τ_x .* μ_x .+ τ_u .* c.scale .* c.values
    return rhs ./ diag_Q
end

function log_hyperprior(c::MEC, θ)
    j = 0
    s = 0.0
    if !c.fix_τ_u
        j += 1
        s += log_prior_density(c.τ_u_prior, θ[j])
    end
    if !c.fix_μ_x
        j += 1
        s += log_prior_density(c.μ_x_prior, θ[j])
    end
    if !c.fix_τ_x
        j += 1
        s += log_prior_density(c.τ_x_prior, θ[j])
    end
    return s
end

# Proper Gaussian prior `N(μ̂, Q̂⁻¹)` with `Q̂ = τ_x I + τ_u D`:
#
#   log NC = -½ n log(2π) + ½ log|Q̂|
#          = -½ n log(2π) + ½ Σ log(τ_x + τ_u s_i).
#
# Both terms can be θ-dependent (through τ_u, τ_x), so neither can be
# absorbed into the user-independent constant the way MEB drops
# `½ Σ log s_i`. We keep the full log-determinant.
function log_normalizing_constant(c::MEC, θ)
    τ_u, _, τ_x = _mec_unpack(c, θ)
    diag_Q = τ_x .+ τ_u .* c.scale
    n = length(c)
    return -0.5 * n * log(2π) + 0.5 * sum(log, diag_Q)
end

function gmrf(c::MEC, θ)
    τ_u, _, τ_x = _mec_unpack(c, θ)
    diag_Q = τ_x .+ τ_u .* c.scale
    return GMRFs.Generic0GMRF(spdiagm(0 => diag_Q); τ=1.0, rankdef=0)
end

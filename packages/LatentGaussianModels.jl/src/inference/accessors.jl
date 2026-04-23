# R-INLA-shaped accessors on INLAResult.
#
# R-INLA returns a result object with `$summary.fixed`, `$summary.random`,
# `$summary.hyperpar`, and `$marginals.*` lists. We mirror that layout with
# Julia-native NamedTuples keyed by component, so users familiar with the R
# interface can navigate the result object without translation.

"""
    component_range(model, i::Int) -> UnitRange{Int}

Index range of the `i`-th component in the stacked latent vector `x`.
"""
component_range(m::LatentGaussianModel, i::Integer) = m.latent_ranges[i]

"""
    fixed_effects(model, res; level = 0.95) -> Vector{@NamedTuple{...}}

Posterior summaries for each scalar "fixed-effect"-shaped component —
those whose latent dimension is 1 (e.g. `Intercept`, scalar slopes). Each
element has fields `(name, mean, sd, lower, upper)` with the
`(1-level)/2`-quantile intervals computed from the Gaussian at
`(x_mean, x_var)`.

Components of length > 1 are surfaced through [`random_effects`](@ref).
"""
function fixed_effects(m::LatentGaussianModel, res::INLAResult; level::Real = 0.95)
    rows = NamedTuple{(:name, :mean, :sd, :lower, :upper),
                       Tuple{String, Float64, Float64, Float64, Float64}}[]
    α = (1 - level) / 2
    z = _normal_quantile(1 - α)
    for (i, c) in enumerate(m.components)
        length(c) == 1 || continue
        idx = first(m.latent_ranges[i])
        μ = res.x_mean[idx]
        sd = sqrt(max(res.x_var[idx], 0.0))
        push!(rows, (name = _component_name(c, i),
                     mean = μ, sd = sd,
                     lower = μ - z * sd, upper = μ + z * sd))
    end
    return rows
end

"""
    random_effects(model, res; level = 0.95)
      -> Dict{String, @NamedTuple{mean, sd, lower, upper}}

Per-component posterior summaries for vector-valued components (length > 1).
Each entry is a NamedTuple of vectors — one entry per latent coordinate
within that component.
"""
function random_effects(m::LatentGaussianModel, res::INLAResult; level::Real = 0.95)
    α = (1 - level) / 2
    z = _normal_quantile(1 - α)
    out = Dict{String, NamedTuple{(:mean, :sd, :lower, :upper),
                                    NTuple{4, Vector{Float64}}}}()
    for (i, c) in enumerate(m.components)
        length(c) > 1 || continue
        rng = m.latent_ranges[i]
        μ = res.x_mean[rng]
        sd = sqrt.(max.(res.x_var[rng], 0.0))
        out[_component_name(c, i)] = (mean = μ, sd = sd,
                                       lower = μ .- z .* sd,
                                       upper = μ .+ z .* sd)
    end
    return out
end

"""
    hyperparameters(model, res; level = 0.95) -> Vector{@NamedTuple{...}}

Gaussian-approximation summaries of the hyperparameters on the internal
scale: one row per entry of `θ`, ordered as (likelihood, component_1, ..).

Each row has `(name, mean, sd, lower, upper)`. For display on user scale,
apply the relevant `user_scale` transforms (e.g. `exp` for log-precisions).
"""
function hyperparameters(m::LatentGaussianModel, res::INLAResult; level::Real = 0.95)
    names = _hyperparameter_names(m)
    α = (1 - level) / 2
    z = _normal_quantile(1 - α)
    rows = NamedTuple{(:name, :mean, :sd, :lower, :upper),
                       Tuple{String, Float64, Float64, Float64, Float64}}[]
    for j in eachindex(res.θ̂)
        μ = res.θ̂[j]
        sd = sqrt(max(res.Σθ[j, j], 0.0))
        push!(rows, (name = names[j], mean = μ, sd = sd,
                     lower = μ - z * sd, upper = μ + z * sd))
    end
    return rows
end

"""
    log_marginal_likelihood(res::INLAResult) -> Float64

Estimate of `log p(y)` (model evidence) from the importance-sampling
normalising constant.
"""
log_marginal_likelihood(res::INLAResult) = res.log_marginal

# ---------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", res::INLAResult)
    print(io, "INLAResult\n")
    print(io, "  n_x           = ", length(res.x_mean), "\n")
    print(io, "  dim(θ)        = ", length(res.θ̂), "\n")
    print(io, "  θ̂             = ", res.θ̂, "\n")
    print(io, "  #design points= ", length(res.θ_points), "\n")
    print(io, "  log p(y)      = ", res.log_marginal, "\n")
    print(io, "See fixed_effects, random_effects, hyperparameters, " *
              "posterior_marginal_x, posterior_marginal_θ.")
end

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

# Approximate inverse standard-normal CDF using Acklam's rational
# approximation. Avoids a Distributions.quantile call inside the hot path.
function _normal_quantile(p::Real)
    p ≤ 0 && return -Inf
    p ≥ 1 && return Inf
    # Coefficients from Peter Acklam's algorithm.
    a = (-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00)
    b = (-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00)
    d = ( 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00)
    plow = 0.02425
    phigh = 1 - plow
    if p < plow
        q = sqrt(-2 * log(p))
        return (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
               ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
    elseif p ≤ phigh
        q = p - 0.5
        r = q * q
        return (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q /
               (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1)
    else
        q = sqrt(-2 * log(1 - p))
        return -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
                ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
    end
end

_component_name(c::AbstractLatentComponent, i::Integer) =
    string(nameof(typeof(c)), "[", i, "]")

function _hyperparameter_names(m::LatentGaussianModel)
    names = String[]
    n_ℓ = nhyperparameters(m.likelihood)
    for k in 1:n_ℓ
        push!(names, "likelihood[$k]")
    end
    for (i, c) in enumerate(m.components)
        n_c = nhyperparameters(c)
        base = _component_name(c, i)
        if n_c == 1
            push!(names, base)
        else
            for k in 1:n_c
                push!(names, string(base, "[", k, "]"))
            end
        end
    end
    return names
end

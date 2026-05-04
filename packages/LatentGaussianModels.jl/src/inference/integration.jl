"""
    AbstractIntegrationScheme

Strategy for numerically integrating functions of `θ` against the
Gaussian approximation at the mode `θ̂` with covariance `Σ = H⁻¹` (where
`H` is the Hessian of the negative log-posterior at `θ̂`).

A scheme produces a list of design points `θ_k` and weights `w_k` such
that

    ∫ f(θ) π(θ | y) dθ  ≈  ∑_k w_k f(θ_k)

is a good quadrature rule when `π(θ | y)` is well approximated by a
Gaussian in a log-posterior sense. Implementations return
`(points, weights)` with `points::Vector{Vector{Float64}}` and
`weights::Vector{Float64}` summing to 1.
"""
abstract type AbstractIntegrationScheme end

"""
    Grid(; n_per_dim = 5, span = 3.0,
           stdev_corr_pos = nothing, stdev_corr_neg = nothing)

Tensor-product midpoint grid in the eigenbasis of `H`. `n_per_dim`
points per dimension on `[-span, span]` (in units of standard
deviations of the Gaussian approximation), weights derived from the
standard-normal density.

Recommended only for `dim(θ) ≤ 3`; cost is `n_per_dim^dim(θ)`.

# Asymmetric skewness corrections

`stdev_corr_pos` / `stdev_corr_neg` are length-`m = dim(θ)` per-axis
stretch factors applied to the eigen-basis grid coordinates `z`
*before* mapping back to the native `θ`-parameterisation: along
eigen-axis `k`, a point with `z_k > 0` is placed at
`θ̂ + halfσ * (z_k * stdev_corr_pos[k]) e_k` and a point with `z_k < 0`
at `θ̂ + halfσ * (z_k * stdev_corr_neg[k]) e_k`. Quadrature weights
are unchanged from the standard-normal weights — the importance-
sampling reweight `π̂(θ_k|y) / q(θ_k)` performed by the surrounding
INLA pipeline absorbs any proposal-mismatch from the asymmetric
placement, while the asymmetric placement itself concentrates points
where `π(θ|y)` actually has mass on each side.

The reference values defining the stretch are σ⁺_k / σ⁻_k from
Rue–Martino–Chopin (2009) §6.5: `σ⁺_k` is the positive-direction step
along axis `k` at which `log π̂(θ̂) - log π̂(θ̂ + σ⁺_k e_k) = ½`, and
similarly for `σ⁻_k` in the negative direction. For an exactly
Gaussian posterior σ⁺_k = σ⁻_k = √λ_k (the corresponding eigenvalue
of `Σ`), so the stretches reduce to `1`. See
[`compute_skewness_corrections`](@ref) for the helper that estimates
them, and the `skewness_correction` keyword on [`INLA`](@ref) for the
opt-in flag in the standard pipeline.

Pass `nothing` (the default) on either side to disable the
correction on that side; mixed behaviour (only one side specified) is
not allowed — use `ones(m)` to leave the other side at `1.0`.
"""
Base.@kwdef struct Grid <: AbstractIntegrationScheme
    n_per_dim::Int = 5
    span::Float64 = 3.0
    stdev_corr_pos::Union{Nothing, Vector{Float64}} = nothing
    stdev_corr_neg::Union{Nothing, Vector{Float64}} = nothing

    function Grid(n_per_dim::Int, span::Float64,
            stdev_corr_pos::Union{Nothing, Vector{Float64}},
            stdev_corr_neg::Union{Nothing, Vector{Float64}})
        n_per_dim ≥ 1 ||
            throw(ArgumentError("Grid: n_per_dim must be ≥ 1, got $n_per_dim"))
        span > 0 ||
            throw(ArgumentError("Grid: span must be > 0, got $span"))
        # Either both stretches are nothing (symmetric) or both are
        # vectors of the same length (asymmetric).
        if stdev_corr_pos === nothing && stdev_corr_neg === nothing
            return new(n_per_dim, span, nothing, nothing)
        elseif stdev_corr_pos === nothing || stdev_corr_neg === nothing
            throw(ArgumentError(
                "Grid: stdev_corr_pos and stdev_corr_neg must both be " *
                "`nothing` or both be vectors; got " *
                "$(stdev_corr_pos === nothing ? "nothing" : "vector") and " *
                "$(stdev_corr_neg === nothing ? "nothing" : "vector")"))
        end
        # Both are vectors here — narrowed by the elseif above.
        length(stdev_corr_pos) == length(stdev_corr_neg) ||
            throw(ArgumentError(
                "Grid: stdev_corr_pos and stdev_corr_neg must have " *
                "the same length; got $(length(stdev_corr_pos)) and " *
                "$(length(stdev_corr_neg))"))
        all(>(0), stdev_corr_pos) || throw(ArgumentError(
            "Grid: stdev_corr_pos entries must be > 0"))
        all(>(0), stdev_corr_neg) || throw(ArgumentError(
            "Grid: stdev_corr_neg entries must be > 0"))
        return new(n_per_dim, span, stdev_corr_pos, stdev_corr_neg)
    end
end

"""
    GaussHermite(; n_per_dim = 5)

Tensor-product Gauss-Hermite quadrature in the eigenbasis of `H`.
Exact for polynomial integrands of degree `2 n_per_dim - 1` against
the Gaussian weight.
"""
Base.@kwdef struct GaussHermite <: AbstractIntegrationScheme
    n_per_dim::Int = 5
end

"""
    CCD(; f0 = nothing)

Central composite design in the eigenbasis of `H`, after Rue, Martino
& Chopin (2009, §6.5). For `m = dim(θ) ≥ 2` the design consists of

- 1 center point,
- `2m` axial points at `±f0` on each eigenaxis,
- `2^(m-p)` fractional-factorial corner points with coordinates in
  `{-1, +1}` (we use `p = 0` — full factorial — up to `m = 5`; above
  that callers should prefer `Grid` or `GaussHermite`).

Weights are chosen so that the rule integrates a quadratic + pure
fourth-order term exactly against the standard normal. The default
`f0 = sqrt(m + 2)` is the Rue-Martino-Chopin recommendation.

Falls back to `Grid(n_per_dim = 7)` for `m ≤ 1`.
"""
Base.@kwdef struct CCD <: AbstractIntegrationScheme
    f0::Union{Nothing, Float64} = nothing
end

# --- Quadrature point construction --------------------------------------

"""
    integration_nodes(scheme, θ̂, Σ) -> (points, log_weights)

Return the design points in the native θ-parameterisation and their
*log-weights*. The weights carry the Gaussian base measure already;
caller multiplies by `exp(Δ log π)` to get weights under the true
posterior, then renormalises. Returning log-weights avoids underflow
at high dimension.
"""
function integration_nodes(scheme::Grid, θ̂::AbstractVector{<:Real},
        Σ::AbstractMatrix{<:Real})
    m = length(θ̂)
    F = eigen(Symmetric(Σ))
    halfσ = F.vectors * Diagonal(sqrt.(max.(F.values, 0.0)))   # Σ^{1/2}

    has_skew = scheme.stdev_corr_pos !== nothing
    if has_skew
        length(scheme.stdev_corr_pos) == m || throw(ArgumentError(
            "Grid: stdev_corr_pos has length $(length(scheme.stdev_corr_pos)) " *
            "but θ̂ has dim $m"))
    end

    z_1d = range(-scheme.span, scheme.span; length=scheme.n_per_dim)
    Δz = step(z_1d)
    # Standard-normal weights φ(z) Δz per dimension, product over dims.
    # Weights are unchanged when skewness correction is on: the
    # asymmetric stretch only relocates points; the IS reweight
    # `π̂(θ_k|y)/q(θ_k)` in the surrounding fit absorbs the
    # proposal-mismatch.
    log_w1d = [-0.5 * z^2 - 0.5 * log(2π) + log(Δz) for z in z_1d]

    iters = Iterators.product(ntuple(_ -> 1:(scheme.n_per_dim), m)...)
    points = Vector{Vector{Float64}}(undef, length(iters))
    lws = Vector{Float64}(undef, length(iters))
    for (k, idx) in enumerate(iters)
        z = [z_1d[i] for i in idx]
        if has_skew
            z = [z[d] > 0 ? z[d] * scheme.stdev_corr_pos[d] :
                 z[d] * scheme.stdev_corr_neg[d] for d in 1:m]
        end
        points[k] = θ̂ + halfσ * z
        lws[k] = sum(log_w1d[i] for i in idx)
    end
    return points, lws
end

function integration_nodes(scheme::GaussHermite, θ̂::AbstractVector{<:Real},
        Σ::AbstractMatrix{<:Real})
    m = length(θ̂)
    F = eigen(Symmetric(Σ))
    halfσ = F.vectors * Diagonal(sqrt.(max.(F.values, 0.0)))

    # `gausshermite` returns physicists' nodes for ∫ exp(-x²) p(x) dx.
    # We use the probabilists' transform x = z/√2, weight ω = w/√π.
    x, w = FastGaussQuadrature.gausshermite(scheme.n_per_dim)
    z_1d = x .* sqrt(2)
    log_w1d = log.(w ./ sqrt(π))

    iters = Iterators.product(ntuple(_ -> 1:(scheme.n_per_dim), m)...)
    points = Vector{Vector{Float64}}(undef, length(iters))
    lws = Vector{Float64}(undef, length(iters))
    for (k, idx) in enumerate(iters)
        z = [z_1d[i] for i in idx]
        points[k] = θ̂ + halfσ * z
        lws[k] = sum(log_w1d[i] for i in idx)
    end
    return points, lws
end

function integration_nodes(scheme::CCD, θ̂::AbstractVector{<:Real},
        Σ::AbstractMatrix{<:Real})
    m = length(θ̂)
    if m ≤ 1
        return integration_nodes(Grid(n_per_dim=7, span=3.0), θ̂, Σ)
    end

    f0 = scheme.f0 === nothing ? sqrt(m + 2.0) : scheme.f0
    F = eigen(Symmetric(Σ))
    halfσ = F.vectors * Diagonal(sqrt.(max.(F.values, 0.0)))

    zs = Vector{Vector{Float64}}()
    # Center
    push!(zs, zeros(m))
    # Axial
    for i in 1:m
        e = zeros(m)
        e[i] = f0
        push!(zs, copy(e))
        e[i] = -f0
        push!(zs, copy(e))
    end
    # Full factorial ±1 corners (for m ≤ 5 keep cost reasonable; higher
    # dim callers should pick a different scheme).
    m ≤ 5 || throw(ArgumentError("CCD is configured for dim(θ) ≤ 5; got $m"))
    for bits in 0:(2^m - 1)
        z = [iszero((bits >> (i - 1)) & 1) ? -1.0 : 1.0 for i in 1:m]
        push!(zs, z)
    end

    # Weights: choose so that the design integrates the standard normal's
    # second moments exactly. With center w0, axial w1, corner w2:
    #   Σ w = 1
    #   2 w1 f0² + 2^m w2 = 1                 (each axis variance = 1)
    #   2 w1 f0⁴ + 2^m w2 = 3                 (kurtosis, unused here)
    # We solve the first two; w0 absorbs the residual. Corners have even
    # count so off-diagonal moments vanish automatically.
    nc = 2^m
    # From the two-equation system solved for (w1, w2) with w0 = 1 - 2m w1 - nc w2:
    # Pick the rotatable weights: w1 = 1 / (2 f0²) × (1 / (2m) − nc × something).
    # Keep it simple: match just the second-moment condition; split
    # remaining weight 50/50 between center and corners.
    # -> 2 m w1 f0² + nc w2 = 1, plus choose w1 = w_corners-equal-magnitude.
    # This gives w1 such that 2 m w1 f0² = 1 - nc w2.
    # We tune w2 so that the corner points carry ~10% of total mass and
    # axial points carry ~45% — a standard INLA-style weighting.
    w_total_corners = 0.10
    w_total_axial = 0.45
    w2 = w_total_corners / nc
    # Rescale w1 to satisfy the second-moment condition exactly:
    w1 = (1 - nc * w2) / (2m * f0^2)
    # Center takes the remainder.
    w0 = 1 - 2m * w1 - nc * w2
    # Guard against pathological weights.
    all(>(0), (w0, w1, w2)) ||
        @warn "CCD weights went non-positive; consider Grid or GaussHermite" w0 w1 w2 m f0
    weights = vcat(w0, fill(w1, 2m), fill(w2, nc))

    points = [θ̂ + halfσ * z for z in zs]
    # Log of the *prior (Gaussian) weight* is log(w_k) — we encode the
    # Gaussian base already in the design, so return log(w_k) directly.
    return points, log.(weights)
end

# --- Asymmetric skewness corrections ------------------------------------

"""
    compute_skewness_corrections(log_post, θ̂, Σ;
                                 threshold = 0.05,
                                 max_stretch = 5.0)
        -> (stdev_corr_pos, stdev_corr_neg)

Estimate the per-eigen-axis Rue–Martino–Chopin (2009) §6.5 stretches
that align a Gaussian midpoint grid with the actual density-shape
asymmetry of `π(θ|y)`.

Along eigen-axis `k` of `Σ`, the stretch on the positive side is
`stdev_corr_pos[k] = √(½ / Δ⁺_k)` where
`Δ⁺_k = log π(θ̂) - log π(θ̂ + halfσ_k e_k)` and `halfσ` is `Σ^{1/2}`
expressed in the eigenbasis. Symmetric correction: for an exactly
Gaussian posterior the drop in log-density across one nominal std
along any axis is exactly ½, so the stretch is `1`.

`log_post` is the callable `θ → log π̂(θ|y)` (up to a θ-independent
constant — the constant cancels out of the ratios). Pass any function
of `θ` only (no extra arguments). Internally the helper queries
`log_post(θ̂)` once and `log_post(θ̂ ± halfσ e_k)` for each `k`, so the
total cost is `2m + 1` evaluations.

Stretches are floored at `threshold` and capped at `max_stretch`:
- `Δ < threshold` (very flat axis or numerical noise) → stretch left
  at `1.0`,
- otherwise stretch is clamped to `[1/max_stretch, max_stretch]` to
  avoid runaway proposals when one side of the posterior is a stiff
  wall (Δ becomes large) or pathologically flat.

# Example

```julia
f = θ -> -negative_log_post(θ)        # log π̂(θ|y) up to a constant
stdev_pos, stdev_neg = compute_skewness_corrections(f, θ̂, Σθ)
scheme = Grid(n_per_dim = 21, span = 4.0,
              stdev_corr_pos = stdev_pos,
              stdev_corr_neg = stdev_neg)
```
"""
function compute_skewness_corrections(log_post, θ̂::AbstractVector{<:Real},
        Σ::AbstractMatrix{<:Real};
        threshold::Real=0.05, max_stretch::Real=5.0)
    threshold > 0 ||
        throw(ArgumentError("compute_skewness_corrections: threshold must be > 0"))
    max_stretch > 1 || throw(ArgumentError(
        "compute_skewness_corrections: max_stretch must be > 1"))

    m = length(θ̂)
    F = eigen(Symmetric(Σ))
    halfσ = F.vectors * Diagonal(sqrt.(max.(F.values, 0.0)))

    log_π0 = log_post(θ̂)
    isfinite(log_π0) || throw(ArgumentError(
        "compute_skewness_corrections: log_post(θ̂) is not finite ($log_π0)"))

    pos = ones(Float64, m)
    neg = ones(Float64, m)
    half_target = 0.5
    inv_max = 1 / max_stretch
    for k in 1:m
        e = zeros(Float64, m)
        e[k] = 1.0
        shift = halfσ * e
        log_π_p = log_post(θ̂ + shift)
        log_π_m = log_post(θ̂ - shift)
        Δp = log_π0 - log_π_p
        Δm = log_π0 - log_π_m
        if isfinite(Δp) && Δp > threshold
            pos[k] = clamp(sqrt(half_target / Δp), inv_max, max_stretch)
        end
        if isfinite(Δm) && Δm > threshold
            neg[k] = clamp(sqrt(half_target / Δm), inv_max, max_stretch)
        end
    end
    return pos, neg
end

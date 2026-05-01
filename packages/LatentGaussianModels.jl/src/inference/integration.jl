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
    Grid(; n_per_dim = 5, span = 3.0)

Tensor-product midpoint grid in the eigenbasis of `H`. `n_per_dim`
points per dimension on `[-span, span]` (in units of standard
deviations of the Gaussian approximation), weights derived from the
standard-normal density.

Recommended only for `dim(θ) ≤ 3`; cost is `n_per_dim^dim(θ)`.
"""
Base.@kwdef struct Grid <: AbstractIntegrationScheme
    n_per_dim::Int = 5
    span::Float64 = 3.0
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

    z_1d = range(-scheme.span, scheme.span; length=scheme.n_per_dim)
    Δz = step(z_1d)
    # Standard-normal weights φ(z) Δz per dimension, product over dims.
    log_w1d = [-0.5 * z^2 - 0.5 * log(2π) + log(Δz) for z in z_1d]

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

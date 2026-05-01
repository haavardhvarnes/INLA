"""
    ExponentialLikelihood(; link = LogLink(), censoring = nothing, time_hi = nothing)

`y_i | η_i ∼ Exponential(rate = g⁻¹(η_i))`. With the canonical
`LogLink`, the rate is `λ_i = exp(η_i)`, the mean is `1/λ_i = exp(-η_i)`,
and `y_i > 0` is required.

`censoring` is an optional `AbstractVector{Censoring}` of length
`length(y)`; when `nothing` (default), every observation is uncensored.
For `INTERVAL` rows, `time_hi[i]` is the upper bound (with `y[i]` the
lower bound); `time_hi` is otherwise unread and may be `nothing`.

No likelihood hyperparameters. `θ` is ignored.

Closed-form derivatives are provided for `LogLink` for all four
censoring modes; see ADR-018 for the contract.

# Example

```julia
# Right-censored survival data
ℓ = ExponentialLikelihood(censoring = [NONE, RIGHT, NONE, RIGHT])

# Interval-censored
ℓ = ExponentialLikelihood(
    censoring = [NONE, INTERVAL],
    time_hi   = [0.0, 5.0],
)
```
"""
struct ExponentialLikelihood{
    L <: AbstractLinkFunction,
    C <: Union{Nothing, AbstractVector{Censoring}},
    V <: Union{Nothing, AbstractVector{<:Real}}
} <: AbstractLikelihood
    link::L
    censoring::C
    time_hi::V
end

function ExponentialLikelihood(; link::AbstractLinkFunction=LogLink(),
        censoring=nothing,
        time_hi::Union{Nothing, AbstractVector{<:Real}}=nothing)
    return ExponentialLikelihood(link, _coerce_censoring(censoring), time_hi)
end

link(ℓ::ExponentialLikelihood) = ℓ.link
nhyperparameters(::ExponentialLikelihood) = 0
initial_hyperparameters(::ExponentialLikelihood) = Float64[]

# --- log-link, all-uncensored fast path -------------------------------
# The typed `Nothing` field collapses the per-row censoring branch at
# compile time, giving the simplest code-gen path for the common case.

function log_density(::ExponentialLikelihood{LogLink, Nothing}, y, η, θ)
    s = zero(promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        # log f(t) = log λ - λ t = η - exp(η) t
        s += η[i] - exp(η[i]) * y[i]
    end
    return s
end

function ∇_η_log_density(::ExponentialLikelihood{LogLink, Nothing}, y, η, θ)
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        # ∂η log p = 1 - λ t
        out[i] = 1 - exp(η[i]) * y[i]
    end
    return out
end

function ∇²_η_log_density(::ExponentialLikelihood{LogLink, Nothing}, y, η, θ)
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        # ∂²η log p = -λ t
        out[i] = -exp(η[i]) * y[i]
    end
    return out
end

function ∇³_η_log_density(::ExponentialLikelihood{LogLink, Nothing}, y, η, θ)
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        # ∂³η log p = -λ t (derivative of ∂²η, since u' = u for log link)
        out[i] = -exp(η[i]) * y[i]
    end
    return out
end

# --- log-link, mixed censoring ----------------------------------------
# Closed-form derivatives per censoring mode. Convention: u = λt,
# Δ = u_hi - u_lo. Under log link, dη u = u, so we get the η-derivatives
# of `h(u)` from the chain rule:
#   ∂η  h(u)   = h'(u) u
#   ∂²η h(u)   = h''(u) u² + h'(u) u
#   ∂³η h(u)   = h'''(u) u³ + 3 h''(u) u² + h'(u) u
#
# NONE     :  log p = η - u            (h = -u + log u + const)
# RIGHT    :  log p = -u
# LEFT     :  log p = log(1 - exp(-u)) = log(-expm1(-u))
# INTERVAL :  log p = -u_lo + log(1 - exp(-Δ))
#
# For LEFT and INTERVAL the derivatives use `expm1` to retain digits
# near `u → 0⁺`. With the survival likelihood always evaluated on
# strictly positive event times (validated at `fit`-time), we never
# hit u = 0 exactly.

function log_density(ℓ::ExponentialLikelihood{LogLink, <:AbstractVector{Censoring}},
        y, η, θ)
    s = zero(promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        c = ℓ.censoring[i]
        η_i = η[i]
        t_lo = y[i]
        if c === NONE
            s += η_i - exp(η_i) * t_lo
        elseif c === RIGHT
            s += -exp(η_i) * t_lo
        elseif c === LEFT
            u = exp(η_i) * t_lo
            s += log(-expm1(-u))
        else  # INTERVAL
            t_hi = ℓ.time_hi[i]
            λ = exp(η_i)
            u_lo = λ * t_lo
            Δ = λ * (t_hi - t_lo)
            s += -u_lo + log(-expm1(-Δ))
        end
    end
    return s
end

function ∇_η_log_density(ℓ::ExponentialLikelihood{LogLink, <:AbstractVector{Censoring}},
        y, η, θ)
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        c = ℓ.censoring[i]
        η_i = η[i]
        t_lo = y[i]
        if c === NONE
            out[i] = 1 - exp(η_i) * t_lo
        elseif c === RIGHT
            out[i] = -exp(η_i) * t_lo
        elseif c === LEFT
            u = exp(η_i) * t_lo
            out[i] = u / expm1(u)
        else  # INTERVAL
            t_hi = ℓ.time_hi[i]
            λ = exp(η_i)
            u_lo = λ * t_lo
            Δ = λ * (t_hi - t_lo)
            out[i] = -u_lo + Δ / expm1(Δ)
        end
    end
    return out
end

function ∇²_η_log_density(ℓ::ExponentialLikelihood{LogLink, <:AbstractVector{Censoring}},
        y, η, θ)
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        c = ℓ.censoring[i]
        η_i = η[i]
        t_lo = y[i]
        if c === NONE || c === RIGHT
            out[i] = -exp(η_i) * t_lo
        elseif c === LEFT
            u = exp(η_i) * t_lo
            em1 = expm1(u)
            eu = exp(u)
            # ∂²η log p = u/(eᵘ - 1) - u² eᵘ / (eᵘ - 1)²
            out[i] = u / em1 - u^2 * eu / em1^2
        else  # INTERVAL
            t_hi = ℓ.time_hi[i]
            λ = exp(η_i)
            u_lo = λ * t_lo
            Δ = λ * (t_hi - t_lo)
            em1 = expm1(Δ)
            eΔ = exp(Δ)
            out[i] = -u_lo + Δ / em1 - Δ^2 * eΔ / em1^2
        end
    end
    return out
end

function ∇³_η_log_density(ℓ::ExponentialLikelihood{LogLink, <:AbstractVector{Censoring}},
        y, η, θ)
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        c = ℓ.censoring[i]
        η_i = η[i]
        t_lo = y[i]
        if c === NONE || c === RIGHT
            out[i] = -exp(η_i) * t_lo
        elseif c === LEFT
            u = exp(η_i) * t_lo
            em1 = expm1(u)
            eu = exp(u)
            # ∂³η log p = u/em1 - 3u² eᵘ/em1² + u³ eᵘ(eᵘ + 1)/em1³
            out[i] = u / em1 - 3 * u^2 * eu / em1^2 +
                     u^3 * eu * (eu + 1) / em1^3
        else  # INTERVAL
            t_hi = ℓ.time_hi[i]
            λ = exp(η_i)
            u_lo = λ * t_lo
            Δ = λ * (t_hi - t_lo)
            em1 = expm1(Δ)
            eΔ = exp(Δ)
            out[i] = -u_lo + Δ / em1 - 3 * Δ^2 * eΔ / em1^2 +
                     Δ^3 * eΔ * (eΔ + 1) / em1^3
        end
    end
    return out
end

# --- pointwise log-density --------------------------------------------
# Sum equals `log_density` regardless of censoring mix.

function pointwise_log_density(::ExponentialLikelihood{LogLink, Nothing}, y, η, θ)
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        out[i] = η[i] - exp(η[i]) * y[i]
    end
    return out
end

function pointwise_log_density(
        ℓ::ExponentialLikelihood{LogLink, <:AbstractVector{Censoring}}, y, η, θ)
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        c = ℓ.censoring[i]
        η_i = η[i]
        t_lo = y[i]
        if c === NONE
            out[i] = η_i - exp(η_i) * t_lo
        elseif c === RIGHT
            out[i] = -exp(η_i) * t_lo
        elseif c === LEFT
            u = exp(η_i) * t_lo
            out[i] = log(-expm1(-u))
        else  # INTERVAL
            t_hi = ℓ.time_hi[i]
            λ = exp(η_i)
            u_lo = λ * t_lo
            Δ = λ * (t_hi - t_lo)
            out[i] = -u_lo + log(-expm1(-Δ))
        end
    end
    return out
end

# --- pointwise CDF (for PIT diagnostics) ------------------------------
# v0.1: defined only for all-uncensored data. Censored PIT
# (Henderson-Crowther) is deferred to v0.2 per ADR-018.

function pointwise_cdf(::ExponentialLikelihood{LogLink, Nothing}, y, η, θ)
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        out[i] = -expm1(-exp(η[i]) * y[i])
    end
    return out
end

function pointwise_cdf(
        ℓ::ExponentialLikelihood{LogLink, <:AbstractVector{Censoring}}, y, η, θ)
    if any(!=(NONE), ℓ.censoring)
        throw(ArgumentError(
            "pointwise_cdf is undefined for censored observations; " *
            "PIT under censoring (Henderson-Crowther) is deferred to v0.2"))
    end
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        out[i] = -expm1(-exp(η[i]) * y[i])
    end
    return out
end

"""
    WeibullLikelihood(; link = LogLink(), censoring = nothing, time_hi = nothing,
                       hyperprior = GammaPrecision(1.0, 0.001))

Weibull survival likelihood in the **proportional-hazards** (PH)
parameterisation, matching R-INLA's `family = "weibullsurv"` (variant 0).
With shape `α > 0` and the canonical `LogLink`,

```
λ_i  = exp(η_i)
h(t) = α λ_i t^(α-1)
S(t) = exp(-λ_i t^α)
f(t) = α λ_i t^(α-1) exp(-λ_i t^α)
```

so the cumulative hazard at `t` is `u(t) = λ_i t^α`. The shape `α` is
the single likelihood hyperparameter, carried internally as
`θ_ℓ = [log α]`. The default hyperprior matches R-INLA's `weibullsurv`
default — `loggamma(1, 0.001)` on `log α`, encoded here as
`GammaPrecision(1.0, 0.001)`. (`PCAlphaW` (Sørbye-Rue 2017) lands in a
follow-up PR per ADR-018; until then, `GammaPrecision` is the placeholder.)

`censoring` is an optional `AbstractVector{Censoring}` of length
`length(y)`; when `nothing` (default), every observation is uncensored.
For `INTERVAL` rows, `time_hi[i]` is the upper bound (with `y[i]` the
lower bound); `time_hi` is otherwise unread and may be `nothing`.

Closed-form derivatives are provided for `LogLink` for all four
censoring modes; the η-derivative shapes coincide with `ExponentialLikelihood`
once `u = λ t` is replaced by `u = λ t^α` and time differences by
`(t_hi^α − t_lo^α)`. See ADR-018 for the contract.

# Example

```julia
# Right-censored survival data with shape α
ℓ = WeibullLikelihood(censoring = [NONE, RIGHT, NONE, RIGHT])

# Interval-censored
ℓ = WeibullLikelihood(
    censoring = [NONE, INTERVAL],
    time_hi   = [0.0, 5.0],
)
```
"""
struct WeibullLikelihood{
        L <: AbstractLinkFunction,
        C <: Union{Nothing, AbstractVector{Censoring}},
        V <: Union{Nothing, AbstractVector{<:Real}},
        P <: AbstractHyperPrior,
} <: AbstractLikelihood
    link::L
    censoring::C
    time_hi::V
    hyperprior::P
end

function WeibullLikelihood(; link::AbstractLinkFunction = LogLink(),
        censoring = nothing,
        time_hi::Union{Nothing, AbstractVector{<:Real}} = nothing,
        hyperprior::AbstractHyperPrior = GammaPrecision(1.0, 0.001))
    link isa LogLink ||
        throw(ArgumentError(
            "WeibullLikelihood: only LogLink is supported, got $(typeof(link))"))
    return WeibullLikelihood(link, _coerce_censoring(censoring), time_hi, hyperprior)
end

link(ℓ::WeibullLikelihood) = ℓ.link
nhyperparameters(::WeibullLikelihood) = 1
initial_hyperparameters(::WeibullLikelihood) = [0.0]   # log α = 0, α = 1
log_hyperprior(ℓ::WeibullLikelihood, θ) = log_prior_density(ℓ.hyperprior, θ[1])

# --- log-link, all-uncensored fast path -------------------------------
# log f(t) = log α + η + (α-1) log t - exp(η) t^α
# ∂η log f = 1 - u,  ∂²η = -u,  ∂³η = -u   (u = exp(η) t^α)

function log_density(ℓ::WeibullLikelihood{LogLink, Nothing}, y, η, θ)
    α = exp(θ[1])
    log_α = θ[1]
    s = zero(promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        y[i] > 0 || return -Inf
        u = exp(η[i]) * y[i]^α
        s += log_α + η[i] + (α - 1) * log(y[i]) - u
    end
    return s
end

function ∇_η_log_density(ℓ::WeibullLikelihood{LogLink, Nothing}, y, η, θ)
    α = exp(θ[1])
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        out[i] = 1 - exp(η[i]) * y[i]^α
    end
    return out
end

function ∇²_η_log_density(ℓ::WeibullLikelihood{LogLink, Nothing}, y, η, θ)
    α = exp(θ[1])
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        out[i] = -exp(η[i]) * y[i]^α
    end
    return out
end

function ∇³_η_log_density(ℓ::WeibullLikelihood{LogLink, Nothing}, y, η, θ)
    α = exp(θ[1])
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        out[i] = -exp(η[i]) * y[i]^α
    end
    return out
end

# --- log-link, mixed censoring ----------------------------------------
# Same η-derivative skeleton as ExponentialLikelihood; substitute
# u  = exp(η) t^α   (in place of exp(η) t)
# Δ  = exp(η) (t_hi^α - t_lo^α)
# The NONE branch carries the extra `log α + (α-1) log t` constants in
# `log_density` (these are η-independent so they drop out of derivatives).

function log_density(ℓ::WeibullLikelihood{LogLink, <:AbstractVector{Censoring}},
        y, η, θ)
    α = exp(θ[1])
    log_α = θ[1]
    s = zero(promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        y[i] > 0 || return -Inf
        c = ℓ.censoring[i]
        η_i = η[i]
        t_lo = y[i]
        t_lo_α = t_lo^α
        if c === NONE
            u = exp(η_i) * t_lo_α
            s += log_α + η_i + (α - 1) * log(t_lo) - u
        elseif c === RIGHT
            s += -exp(η_i) * t_lo_α
        elseif c === LEFT
            u = exp(η_i) * t_lo_α
            s += log(-expm1(-u))
        else  # INTERVAL
            t_hi = ℓ.time_hi[i]
            λ = exp(η_i)
            u_lo = λ * t_lo_α
            Δ = λ * (t_hi^α - t_lo_α)
            s += -u_lo + log(-expm1(-Δ))
        end
    end
    return s
end

function ∇_η_log_density(
        ℓ::WeibullLikelihood{LogLink, <:AbstractVector{Censoring}}, y, η, θ)
    α = exp(θ[1])
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        c = ℓ.censoring[i]
        η_i = η[i]
        t_lo = y[i]
        t_lo_α = t_lo^α
        if c === NONE
            out[i] = 1 - exp(η_i) * t_lo_α
        elseif c === RIGHT
            out[i] = -exp(η_i) * t_lo_α
        elseif c === LEFT
            u = exp(η_i) * t_lo_α
            out[i] = u / expm1(u)
        else  # INTERVAL
            t_hi = ℓ.time_hi[i]
            λ = exp(η_i)
            u_lo = λ * t_lo_α
            Δ = λ * (t_hi^α - t_lo_α)
            out[i] = -u_lo + Δ / expm1(Δ)
        end
    end
    return out
end

function ∇²_η_log_density(
        ℓ::WeibullLikelihood{LogLink, <:AbstractVector{Censoring}}, y, η, θ)
    α = exp(θ[1])
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        c = ℓ.censoring[i]
        η_i = η[i]
        t_lo = y[i]
        t_lo_α = t_lo^α
        if c === NONE || c === RIGHT
            out[i] = -exp(η_i) * t_lo_α
        elseif c === LEFT
            u = exp(η_i) * t_lo_α
            em1 = expm1(u)
            eu = exp(u)
            out[i] = u / em1 - u^2 * eu / em1^2
        else  # INTERVAL
            t_hi = ℓ.time_hi[i]
            λ = exp(η_i)
            u_lo = λ * t_lo_α
            Δ = λ * (t_hi^α - t_lo_α)
            em1 = expm1(Δ)
            eΔ = exp(Δ)
            out[i] = -u_lo + Δ / em1 - Δ^2 * eΔ / em1^2
        end
    end
    return out
end

function ∇³_η_log_density(
        ℓ::WeibullLikelihood{LogLink, <:AbstractVector{Censoring}}, y, η, θ)
    α = exp(θ[1])
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        c = ℓ.censoring[i]
        η_i = η[i]
        t_lo = y[i]
        t_lo_α = t_lo^α
        if c === NONE || c === RIGHT
            out[i] = -exp(η_i) * t_lo_α
        elseif c === LEFT
            u = exp(η_i) * t_lo_α
            em1 = expm1(u)
            eu = exp(u)
            out[i] = u / em1 - 3 * u^2 * eu / em1^2 +
                     u^3 * eu * (eu + 1) / em1^3
        else  # INTERVAL
            t_hi = ℓ.time_hi[i]
            λ = exp(η_i)
            u_lo = λ * t_lo_α
            Δ = λ * (t_hi^α - t_lo_α)
            em1 = expm1(Δ)
            eΔ = exp(Δ)
            out[i] = -u_lo + Δ / em1 - 3 * Δ^2 * eΔ / em1^2 +
                     Δ^3 * eΔ * (eΔ + 1) / em1^3
        end
    end
    return out
end

# --- pointwise log-density --------------------------------------------

function pointwise_log_density(ℓ::WeibullLikelihood{LogLink, Nothing}, y, η, θ)
    α = exp(θ[1])
    log_α = θ[1]
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        if y[i] > 0
            u = exp(η[i]) * y[i]^α
            out[i] = log_α + η[i] + (α - 1) * log(y[i]) - u
        else
            out[i] = T(-Inf)
        end
    end
    return out
end

function pointwise_log_density(
        ℓ::WeibullLikelihood{LogLink, <:AbstractVector{Censoring}}, y, η, θ)
    α = exp(θ[1])
    log_α = θ[1]
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        c = ℓ.censoring[i]
        η_i = η[i]
        t_lo = y[i]
        if !(t_lo > 0)
            out[i] = T(-Inf)
            continue
        end
        t_lo_α = t_lo^α
        if c === NONE
            u = exp(η_i) * t_lo_α
            out[i] = log_α + η_i + (α - 1) * log(t_lo) - u
        elseif c === RIGHT
            out[i] = -exp(η_i) * t_lo_α
        elseif c === LEFT
            u = exp(η_i) * t_lo_α
            out[i] = log(-expm1(-u))
        else  # INTERVAL
            t_hi = ℓ.time_hi[i]
            λ = exp(η_i)
            u_lo = λ * t_lo_α
            Δ = λ * (t_hi^α - t_lo_α)
            out[i] = -u_lo + log(-expm1(-Δ))
        end
    end
    return out
end

# --- pointwise CDF (PIT) ---------------------------------------------
# Closed form: F(t) = 1 - exp(-λ t^α). Defined for all-uncensored data only;
# censored PIT (Henderson-Crowther) is deferred to v0.2 per ADR-018.

function pointwise_cdf(::WeibullLikelihood{LogLink, Nothing}, y, η, θ)
    α = exp(θ[1])
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        out[i] = -expm1(-exp(η[i]) * y[i]^α)
    end
    return out
end

function pointwise_cdf(
        ℓ::WeibullLikelihood{LogLink, <:AbstractVector{Censoring}}, y, η, θ)
    if any(!=(NONE), ℓ.censoring)
        throw(ArgumentError(
            "pointwise_cdf is undefined for censored observations; " *
            "PIT under censoring (Henderson-Crowther) is deferred to v0.2"))
    end
    α = exp(θ[1])
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        out[i] = -expm1(-exp(η[i]) * y[i]^α)
    end
    return out
end

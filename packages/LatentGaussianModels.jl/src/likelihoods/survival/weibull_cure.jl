"""
    WeibullCureLikelihood(; link = LogLink(), censoring = nothing,
                            time_hi = nothing,
                            hyperprior_alpha = GammaPrecision(1.0, 0.001),
                            hyperprior_p     = LogitBeta(1.0, 1.0))

Weibull mixture-cure survival likelihood, matching R-INLA's
`family = "weibullcure"`. A latent fraction `p ∈ (0, 1)` of the
population is *cured* (will never experience the event); the
remaining `1 − p` follow a Weibull distribution in the proportional-
hazards parameterisation. With shape `α > 0`, cure fraction `p`, and
the canonical `LogLink`,

```
λ_i  = exp(η_i),    u_i = λ_i t^α
S(t) = p + (1 − p) exp(−u_i)
F(t) = (1 − p) (1 − exp(−u_i))
f(t) = (1 − p) α t^(α−1) λ_i exp(−u_i)
```

Two likelihood hyperparameters, carried internally as
`θ_ℓ = [log α, logit p]`. Defaults: `loggamma(1, 0.001)` on `log α`
matching `weibullsurv`, and a uniform prior on `p` (i.e.
`LogitBeta(1, 1)` on the logit scale).

`censoring` is an optional `AbstractVector{Censoring}` of length
`length(y)`; when `nothing` (default), every observation is uncensored
(each contributes `f(t)`, *not* the cured-mass marginal — there is no
information about cured-vs-event without censoring). For `INTERVAL`
rows, `time_hi[i]` is the upper bound (with `y[i]` the lower bound);
`time_hi` is otherwise unread and may be `nothing`.

η-derivatives for the `NONE`/`LEFT`/`INTERVAL` branches coincide with
`WeibullLikelihood` because the `(1 − p)` factor in `f`/`F`/`F_hi − F_lo`
is η-independent. Only the `RIGHT` branch differs: with
`v = exp(−u)` and `D = p + (1 − p) v`, the chain rule gives
`∂η log D = D'/D`, `∂²η log D = D''/D − (D'/D)²`, and
`∂³η log D = D'''/D − 3 D' D''/D² + 2 (D'/D)³`, where

```
D'   = −(1 − p) u v
D''  =  (1 − p) u v (u − 1)
D''' =  (1 − p) u v (3u − 1 − u²)
```

These reduce to plain Weibull's RIGHT branch as `p → 0`. See ADR-018
for the contract.
"""
struct WeibullCureLikelihood{
    L <: AbstractLinkFunction,
    C <: Union{Nothing, AbstractVector{Censoring}},
    V <: Union{Nothing, AbstractVector{<:Real}},
    Pα <: AbstractHyperPrior,
    Pp <: AbstractHyperPrior
} <: AbstractLikelihood
    link::L
    censoring::C
    time_hi::V
    hyperprior_alpha::Pα
    hyperprior_p::Pp
end

function WeibullCureLikelihood(; link::AbstractLinkFunction=LogLink(),
        censoring=nothing,
        time_hi::Union{Nothing, AbstractVector{<:Real}}=nothing,
        hyperprior_alpha::AbstractHyperPrior=GammaPrecision(1.0, 0.001),
        hyperprior_p::AbstractHyperPrior=LogitBeta(1.0, 1.0))
    link isa LogLink ||
        throw(ArgumentError(
            "WeibullCureLikelihood: only LogLink is supported, got $(typeof(link))"))
    return WeibullCureLikelihood(
        link, _coerce_censoring(censoring), time_hi,
        hyperprior_alpha, hyperprior_p)
end

link(ℓ::WeibullCureLikelihood) = ℓ.link
nhyperparameters(::WeibullCureLikelihood) = 2

# Initial: log α = 0 (α = 1) and logit p = logit(0.1) ≈ −2.197 (mild prior pull
# toward a small cure fraction without committing to the boundary).
initial_hyperparameters(::WeibullCureLikelihood) = [0.0, log(0.1 / 0.9)]

function log_hyperprior(ℓ::WeibullCureLikelihood, θ)
    return log_prior_density(ℓ.hyperprior_alpha, θ[1]) +
           log_prior_density(ℓ.hyperprior_p, θ[2])
end

# Map internal θ_ℓ = [log α, logit p] → user-scale (α, p).
@inline _wc_unpack(θ) = (exp(θ[1]), inv(one(θ[2]) + exp(-θ[2])))

# --- log-link, all-uncensored fast path --------------------------------
# log f(t) = log(1-p) + log α + (α-1) log t + η - exp(η) t^α
# η-derivatives match plain Weibull NONE: ∂η = 1 - u, ∂²η = -u, ∂³η = -u.

function log_density(ℓ::WeibullCureLikelihood{LogLink, Nothing}, y, η, θ)
    α, p = _wc_unpack(θ)
    log_α = θ[1]
    log_1mp = log1p(-p)
    s = zero(promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        y[i] > 0 || return -Inf
        u = exp(η[i]) * y[i]^α
        s += log_1mp + log_α + η[i] + (α - 1) * log(y[i]) - u
    end
    return s
end

function ∇_η_log_density(ℓ::WeibullCureLikelihood{LogLink, Nothing}, y, η, θ)
    α, _ = _wc_unpack(θ)
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        out[i] = 1 - exp(η[i]) * y[i]^α
    end
    return out
end

function ∇²_η_log_density(ℓ::WeibullCureLikelihood{LogLink, Nothing}, y, η, θ)
    α, _ = _wc_unpack(θ)
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        out[i] = -exp(η[i]) * y[i]^α
    end
    return out
end

function ∇³_η_log_density(ℓ::WeibullCureLikelihood{LogLink, Nothing}, y, η, θ)
    α, _ = _wc_unpack(θ)
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        out[i] = -exp(η[i]) * y[i]^α
    end
    return out
end

# --- log-link, mixed censoring -----------------------------------------
# RIGHT  : log S = log[p + (1-p) v]    with v = exp(-u), u = exp(η) t^α
# LEFT   : log F = log(1-p) + log(1 - v)         (η-derivs match plain Weibull)
# INTERVAL: log[F(t_hi) - F(t_lo)]
#         = log(1-p) + log(v_lo - v_hi)            (η-derivs match plain Weibull)

function log_density(ℓ::WeibullCureLikelihood{LogLink, <:AbstractVector{Censoring}},
        y, η, θ)
    α, p = _wc_unpack(θ)
    log_α = θ[1]
    log_1mp = log1p(-p)
    s = zero(promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        y[i] > 0 || return -Inf
        c = ℓ.censoring[i]
        η_i = η[i]
        t_lo = y[i]
        t_lo_α = t_lo^α
        if c === NONE
            u = exp(η_i) * t_lo_α
            s += log_1mp + log_α + η_i + (α - 1) * log(t_lo) - u
        elseif c === RIGHT
            u = exp(η_i) * t_lo_α
            v = exp(-u)
            s += log(p + (1 - p) * v)
        elseif c === LEFT
            u = exp(η_i) * t_lo_α
            s += log_1mp + log(-expm1(-u))
        else  # INTERVAL
            t_hi = ℓ.time_hi[i]
            λ = exp(η_i)
            u_lo = λ * t_lo_α
            Δ = λ * (t_hi^α - t_lo_α)
            s += log_1mp - u_lo + log(-expm1(-Δ))
        end
    end
    return s
end

function ∇_η_log_density(
        ℓ::WeibullCureLikelihood{LogLink, <:AbstractVector{Censoring}}, y, η, θ)
    α, p = _wc_unpack(θ)
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        c = ℓ.censoring[i]
        η_i = η[i]
        t_lo = y[i]
        t_lo_α = t_lo^α
        if c === NONE
            out[i] = 1 - exp(η_i) * t_lo_α
        elseif c === RIGHT
            u = exp(η_i) * t_lo_α
            v = exp(-u)
            D = p + (1 - p) * v
            Dp = -(1 - p) * u * v
            out[i] = Dp / D
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
        ℓ::WeibullCureLikelihood{LogLink, <:AbstractVector{Censoring}}, y, η, θ)
    α, p = _wc_unpack(θ)
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        c = ℓ.censoring[i]
        η_i = η[i]
        t_lo = y[i]
        t_lo_α = t_lo^α
        if c === NONE
            out[i] = -exp(η_i) * t_lo_α
        elseif c === RIGHT
            u = exp(η_i) * t_lo_α
            v = exp(-u)
            D = p + (1 - p) * v
            Dp = -(1 - p) * u * v
            Dpp = (1 - p) * u * v * (u - 1)
            out[i] = Dpp / D - (Dp / D)^2
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
        ℓ::WeibullCureLikelihood{LogLink, <:AbstractVector{Censoring}}, y, η, θ)
    α, p = _wc_unpack(θ)
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        c = ℓ.censoring[i]
        η_i = η[i]
        t_lo = y[i]
        t_lo_α = t_lo^α
        if c === NONE
            out[i] = -exp(η_i) * t_lo_α
        elseif c === RIGHT
            u = exp(η_i) * t_lo_α
            v = exp(-u)
            D = p + (1 - p) * v
            Dp = -(1 - p) * u * v
            Dpp = (1 - p) * u * v * (u - 1)
            Dppp = (1 - p) * u * v * (3 * u - 1 - u^2)
            r1 = Dp / D
            r2 = Dpp / D
            out[i] = Dppp / D - 3 * r1 * r2 + 2 * r1^3
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

# --- pointwise log-density ---------------------------------------------

function pointwise_log_density(ℓ::WeibullCureLikelihood{LogLink, Nothing},
        y, η, θ)
    α, p = _wc_unpack(θ)
    log_α = θ[1]
    log_1mp = log1p(-p)
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        if y[i] > 0
            u = exp(η[i]) * y[i]^α
            out[i] = log_1mp + log_α + η[i] + (α - 1) * log(y[i]) - u
        else
            out[i] = T(-Inf)
        end
    end
    return out
end

function pointwise_log_density(
        ℓ::WeibullCureLikelihood{LogLink, <:AbstractVector{Censoring}}, y, η, θ)
    α, p = _wc_unpack(θ)
    log_α = θ[1]
    log_1mp = log1p(-p)
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
            out[i] = log_1mp + log_α + η_i + (α - 1) * log(t_lo) - u
        elseif c === RIGHT
            u = exp(η_i) * t_lo_α
            v = exp(-u)
            out[i] = log(p + (1 - p) * v)
        elseif c === LEFT
            u = exp(η_i) * t_lo_α
            out[i] = log_1mp + log(-expm1(-u))
        else  # INTERVAL
            t_hi = ℓ.time_hi[i]
            λ = exp(η_i)
            u_lo = λ * t_lo_α
            Δ = λ * (t_hi^α - t_lo_α)
            out[i] = log_1mp - u_lo + log(-expm1(-Δ))
        end
    end
    return out
end

# --- pointwise CDF (PIT) ----------------------------------------------
# The mixture-cure population CDF is `F(t) = (1−p)(1 − exp(−u))`, which is
# bounded above by `1 − p < 1`. Defined for all-uncensored data only;
# censored PIT (Henderson-Crowther) deferred to v0.2 per ADR-018.

function pointwise_cdf(::WeibullCureLikelihood{LogLink, Nothing}, y, η, θ)
    α, p = _wc_unpack(θ)
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        out[i] = (1 - p) * (-expm1(-exp(η[i]) * y[i]^α))
    end
    return out
end

function pointwise_cdf(
        ℓ::WeibullCureLikelihood{LogLink, <:AbstractVector{Censoring}}, y, η, θ)
    if any(!=(NONE), ℓ.censoring)
        throw(ArgumentError(
            "pointwise_cdf is undefined for censored observations; " *
            "PIT under censoring (Henderson-Crowther) is deferred to v0.2"))
    end
    α, p = _wc_unpack(θ)
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        out[i] = (1 - p) * (-expm1(-exp(η[i]) * y[i]^α))
    end
    return out
end

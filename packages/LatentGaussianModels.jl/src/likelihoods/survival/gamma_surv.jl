"""
    GammaSurvLikelihood(; link = LogLink(), censoring = nothing,
                          time_hi = nothing,
                          hyperprior = GammaPrecision(1.0, 5.0e-5))

Gamma survival likelihood matching R-INLA's `family = "gammasurv"`. Uses
R-INLA's mean-precision parameterisation: with shape `a = φ` and rate
`b = φ/μ` where `μ = exp(η)` under the canonical `LogLink`,

```
f(t)     = b^a t^(a-1) exp(-b t) / Γ(a)
S(t)     = Γ(a, b t) / Γ(a)             (regularised upper-incomplete)
F(t)     = γ(a, b t) / Γ(a)             (regularised lower-incomplete)
```

The single hyperparameter is the precision `φ > 0`, carried internally as
`θ_ℓ = [log φ]`. The default hyperprior matches R-INLA's `family = "gamma"`
default — `loggamma(1, 5e-5)` on `log φ`, encoded here as
`GammaPrecision(1.0, 5.0e-5)`.

`censoring` is an optional `AbstractVector{Censoring}` of length
`length(y)`; when `nothing` (default), every observation is uncensored.
For `INTERVAL` rows, `time_hi[i]` is the upper bound (with `y[i]` the
lower bound).

Closed-form derivatives are provided for `LogLink` for all four
censoring modes via the unitless ratio
`g(x) = x · δ(x) / Γ(a, x)` (the gamma analog of the inverse-Mills
ratio), where `δ(x) = x^(a-1) exp(-x) / Γ(a)` and `x = b t`. The
analogous `h(x) = x · δ(x) / γ(a, x)` covers the `LEFT`/`INTERVAL`
branches. See ADR-018 for the contract.
"""
struct GammaSurvLikelihood{
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

function GammaSurvLikelihood(;
        link::AbstractLinkFunction = LogLink(),
        censoring = nothing,
        time_hi::Union{Nothing, AbstractVector{<:Real}} = nothing,
        hyperprior::AbstractHyperPrior = GammaPrecision(1.0, 5.0e-5))
    link isa LogLink ||
        throw(ArgumentError(
            "GammaSurvLikelihood: only LogLink is supported, got $(typeof(link))"))
    return GammaSurvLikelihood(
        link, _coerce_censoring(censoring), time_hi, hyperprior)
end

link(ℓ::GammaSurvLikelihood) = ℓ.link
nhyperparameters(::GammaSurvLikelihood) = 1
initial_hyperparameters(::GammaSurvLikelihood) = [0.0]   # log φ = 0, φ = 1
log_hyperprior(ℓ::GammaSurvLikelihood, θ) = log_prior_density(ℓ.hyperprior, θ[1])

# --- gamma incomplete ratio helpers -----------------------------------
# Built on SpecialFunctions: one `gamma_inc(a, x)` call returns the
# regularised pair `(P, Q) = (P(a,x), Q(a,x)) = (γ(a,x)/Γ(a), Γ(a,x)/Γ(a))`.
# Per-call cost: a single `gamma_inc` plus one `loggamma(a)` cached at the
# top of each method (no `Gamma(a, 1)` object construction).

@inline function _gamma_logS(a, x)
    _, Q = SpecialFunctions.gamma_inc(a, x)
    return log(Q)
end

@inline function _gamma_logF(a, x)
    P, _ = SpecialFunctions.gamma_inc(a, x)
    return log(P)
end

# log of the Gamma(a, 1) density at x: (a-1) log x - x - log Γ(a).
@inline _gamma_logδ(a, lgamma_a, x) = (a - 1) * log(x) - x - lgamma_a

# Unitless ratios:
#   g(x) = x · δ(x) / Γ(a, x)        (RIGHT branch helper)
#   h(x) = x · δ(x) / γ(a, x)        (LEFT branch helper)
# Both computed via log-domain for tail stability.
@inline function _gamma_g(a, lgamma_a, x)
    _, Q = SpecialFunctions.gamma_inc(a, x)
    return exp(log(x) + _gamma_logδ(a, lgamma_a, x) - log(Q))
end

@inline function _gamma_h(a, lgamma_a, x)
    P, _ = SpecialFunctions.gamma_inc(a, x)
    return exp(log(x) + _gamma_logδ(a, lgamma_a, x) - log(P))
end

# --- log-link, all-uncensored fast path --------------------------------
# Reduces to the existing GammaLikelihood log-density and η-derivatives.
# log f = φ log φ - φ η + (φ-1) log y - φ y exp(-η) - log Γ(φ)
# ∂η = φ y exp(-η) - φ = φ (y/μ - 1)
# ∂²η = -φ y exp(-η)
# ∂³η = +φ y exp(-η)

function log_density(ℓ::GammaSurvLikelihood{LogLink, Nothing}, y, η, θ)
    φ = exp(θ[1])
    log_φ = θ[1]
    lgamma_φ = SpecialFunctions.loggamma(φ)
    s = zero(promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        y[i] > 0 || return -Inf
        s += φ * log_φ - φ * η[i] - lgamma_φ +
             (φ - 1) * log(y[i]) - φ * y[i] * exp(-η[i])
    end
    return s
end

function ∇_η_log_density(ℓ::GammaSurvLikelihood{LogLink, Nothing}, y, η, θ)
    φ = exp(θ[1])
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        out[i] = φ * (y[i] * exp(-η[i]) - 1)
    end
    return out
end

function ∇²_η_log_density(ℓ::GammaSurvLikelihood{LogLink, Nothing}, y, η, θ)
    φ = exp(θ[1])
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        out[i] = -φ * y[i] * exp(-η[i])
    end
    return out
end

function ∇³_η_log_density(ℓ::GammaSurvLikelihood{LogLink, Nothing}, y, η, θ)
    φ = exp(θ[1])
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        out[i] = φ * y[i] * exp(-η[i])
    end
    return out
end

# --- log-link, mixed censoring -----------------------------------------
# Per-row dispatch on `Censoring`. With a = φ and x = φ t exp(-η):
#   ∂η x = -x, ∂²η x = +x, ∂³η x = -x.
# RIGHT: log S = log Γ(a, x) - log Γ(a)
#        ∂η log S =  g(x)
#        ∂²η log S = -g(a - x + g)
#        ∂³η log S =  g((a - x + g)² + g(a - x + g) - x)
# LEFT:  log F = log γ(a, x) - log Γ(a)
#        ∂η log F = -h(x)
#        ∂²η log F =  h(a - x - h)
#        ∂³η log F =  h(x + h(a - x - h) - (a - x - h)²)

function log_density(ℓ::GammaSurvLikelihood{LogLink, <:AbstractVector{Censoring}},
        y, η, θ)
    φ = exp(θ[1])
    log_φ = θ[1]
    lgamma_φ = SpecialFunctions.loggamma(φ)
    s = zero(promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        y[i] > 0 || return -Inf
        c = ℓ.censoring[i]
        η_i = η[i]
        x = φ * y[i] * exp(-η_i)
        if c === NONE
            s += φ * log_φ - φ * η_i - lgamma_φ +
                 (φ - 1) * log(y[i]) - x
        elseif c === RIGHT
            _, Q = SpecialFunctions.gamma_inc(φ, x)
            s += log(Q)
        elseif c === LEFT
            P, _ = SpecialFunctions.gamma_inc(φ, x)
            s += log(P)
        else  # INTERVAL
            t_hi = ℓ.time_hi[i]
            x_hi = φ * t_hi * exp(-η_i)
            # P(a, x_hi) - P(a, x_lo) is the regularised mass between bounds.
            # Equivalent to Q(a, x_lo) - Q(a, x_hi); use the form that
            # avoids catastrophic cancellation at small x (both Q ≈ 1).
            P_lo, Q_lo = SpecialFunctions.gamma_inc(φ, x)
            P_hi, Q_hi = SpecialFunctions.gamma_inc(φ, x_hi)
            D = Q_lo > 0.5 ? P_hi - P_lo : Q_lo - Q_hi
            s += log(D)
        end
    end
    return s
end

function ∇_η_log_density(
        ℓ::GammaSurvLikelihood{LogLink, <:AbstractVector{Censoring}}, y, η, θ)
    φ = exp(θ[1])
    lgamma_φ = SpecialFunctions.loggamma(φ)
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        c = ℓ.censoring[i]
        η_i = η[i]
        x = φ * y[i] * exp(-η_i)
        if c === NONE
            out[i] = φ * (y[i] * exp(-η_i) - 1)            # = x - φ
        elseif c === RIGHT
            out[i] = _gamma_g(φ, lgamma_φ, x)
        elseif c === LEFT
            out[i] = -_gamma_h(φ, lgamma_φ, x)
        else  # INTERVAL
            t_hi = ℓ.time_hi[i]
            x_hi = φ * t_hi * exp(-η_i)
            # D̃ = Q(a, x_lo) - Q(a, x_hi), ∂η Q(a, x) = x δ_norm(x) =: Ñ(x).
            P_lo, Q_lo = SpecialFunctions.gamma_inc(φ, x)
            P_hi, Q_hi = SpecialFunctions.gamma_inc(φ, x_hi)
            D = Q_lo > 0.5 ? P_hi - P_lo : Q_lo - Q_hi
            N_lo = x    * exp(_gamma_logδ(φ, lgamma_φ, x))
            N_hi = x_hi * exp(_gamma_logδ(φ, lgamma_φ, x_hi))
            out[i] = (N_lo - N_hi) / D
        end
    end
    return out
end

function ∇²_η_log_density(
        ℓ::GammaSurvLikelihood{LogLink, <:AbstractVector{Censoring}}, y, η, θ)
    φ = exp(θ[1])
    lgamma_φ = SpecialFunctions.loggamma(φ)
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        c = ℓ.censoring[i]
        η_i = η[i]
        x = φ * y[i] * exp(-η_i)
        if c === NONE
            out[i] = -x                                    # = -φ y/μ
        elseif c === RIGHT
            g = _gamma_g(φ, lgamma_φ, x)
            A = φ - x + g
            out[i] = -g * A
        elseif c === LEFT
            h = _gamma_h(φ, lgamma_φ, x)
            A = φ - x - h
            out[i] = h * A
        else  # INTERVAL
            t_hi = ℓ.time_hi[i]
            x_hi = φ * t_hi * exp(-η_i)
            P_lo, Q_lo = SpecialFunctions.gamma_inc(φ, x)
            P_hi, Q_hi = SpecialFunctions.gamma_inc(φ, x_hi)
            D = Q_lo > 0.5 ? P_hi - P_lo : Q_lo - Q_hi
            N_lo = x    * exp(_gamma_logδ(φ, lgamma_φ, x))
            N_hi = x_hi * exp(_gamma_logδ(φ, lgamma_φ, x_hi))
            # ∂η Ñ(x) = x δ_norm (x - a)
            M_lo = N_lo * (x    - φ)
            M_hi = N_hi * (x_hi - φ)
            Dp  = N_lo - N_hi
            Dpp = M_lo - M_hi
            out[i] = Dpp / D - (Dp / D)^2
        end
    end
    return out
end

function ∇³_η_log_density(
        ℓ::GammaSurvLikelihood{LogLink, <:AbstractVector{Censoring}}, y, η, θ)
    φ = exp(θ[1])
    lgamma_φ = SpecialFunctions.loggamma(φ)
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        c = ℓ.censoring[i]
        η_i = η[i]
        x = φ * y[i] * exp(-η_i)
        if c === NONE
            out[i] = x                                     # = +φ y/μ
        elseif c === RIGHT
            g = _gamma_g(φ, lgamma_φ, x)
            A = φ - x + g
            out[i] = g * (A^2 + g * A - x)
        elseif c === LEFT
            h = _gamma_h(φ, lgamma_φ, x)
            A = φ - x - h
            out[i] = h * (x + h * A - A^2)
        else  # INTERVAL
            t_hi = ℓ.time_hi[i]
            x_hi = φ * t_hi * exp(-η_i)
            P_lo, Q_lo = SpecialFunctions.gamma_inc(φ, x)
            P_hi, Q_hi = SpecialFunctions.gamma_inc(φ, x_hi)
            D = Q_lo > 0.5 ? P_hi - P_lo : Q_lo - Q_hi
            N_lo = x    * exp(_gamma_logδ(φ, lgamma_φ, x))
            N_hi = x_hi * exp(_gamma_logδ(φ, lgamma_φ, x_hi))
            # ∂η[x δ_norm (x - a)] = x δ_norm ((x-a)² - x).
            M_lo = N_lo * (x    - φ)
            M_hi = N_hi * (x_hi - φ)
            K_lo = N_lo * ((x    - φ)^2 - x)
            K_hi = N_hi * ((x_hi - φ)^2 - x_hi)
            Dp   = N_lo - N_hi
            Dpp  = M_lo - M_hi
            Dppp = K_lo - K_hi
            out[i] = Dppp / D - 3 * Dp * Dpp / D^2 + 2 * (Dp / D)^3
        end
    end
    return out
end

# --- pointwise log-density ---------------------------------------------

function pointwise_log_density(ℓ::GammaSurvLikelihood{LogLink, Nothing},
        y, η, θ)
    φ = exp(θ[1])
    log_φ = θ[1]
    lgamma_φ = SpecialFunctions.loggamma(φ)
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        if y[i] > 0
            out[i] = φ * log_φ - φ * η[i] - lgamma_φ +
                     (φ - 1) * log(y[i]) - φ * y[i] * exp(-η[i])
        else
            out[i] = T(-Inf)
        end
    end
    return out
end

function pointwise_log_density(
        ℓ::GammaSurvLikelihood{LogLink, <:AbstractVector{Censoring}},
        y, η, θ)
    φ = exp(θ[1])
    log_φ = θ[1]
    lgamma_φ = SpecialFunctions.loggamma(φ)
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        c = ℓ.censoring[i]
        η_i = η[i]
        if !(y[i] > 0)
            out[i] = T(-Inf)
            continue
        end
        x = φ * y[i] * exp(-η_i)
        if c === NONE
            out[i] = φ * log_φ - φ * η_i - lgamma_φ +
                     (φ - 1) * log(y[i]) - x
        elseif c === RIGHT
            out[i] = _gamma_logS(φ, x)
        elseif c === LEFT
            out[i] = _gamma_logF(φ, x)
        else  # INTERVAL
            t_hi = ℓ.time_hi[i]
            x_hi = φ * t_hi * exp(-η_i)
            P_lo, Q_lo = SpecialFunctions.gamma_inc(φ, x)
            P_hi, Q_hi = SpecialFunctions.gamma_inc(φ, x_hi)
            D = Q_lo > 0.5 ? P_hi - P_lo : Q_lo - Q_hi
            out[i] = log(D)
        end
    end
    return out
end

# --- pointwise CDF (PIT) ----------------------------------------------
# F(t) = P(a, b t). All-uncensored only; censored PIT is a v0.2 item.

function pointwise_cdf(::GammaSurvLikelihood{LogLink, Nothing}, y, η, θ)
    φ = exp(θ[1])
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        out[i] = exp(_gamma_logF(φ, φ * y[i] * exp(-η[i])))
    end
    return out
end

function pointwise_cdf(
        ℓ::GammaSurvLikelihood{LogLink, <:AbstractVector{Censoring}}, y, η, θ)
    if any(!=(NONE), ℓ.censoring)
        throw(ArgumentError(
            "pointwise_cdf is undefined for censored observations; " *
            "PIT under censoring (Henderson-Crowther) is deferred to v0.2"))
    end
    φ = exp(θ[1])
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        out[i] = exp(_gamma_logF(φ, φ * y[i] * exp(-η[i])))
    end
    return out
end

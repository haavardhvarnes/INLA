"""
    LognormalSurvLikelihood(; link = IdentityLink(), censoring = nothing,
                              time_hi = nothing,
                              hyperprior = PCPrecision(1.0, 0.01))

Lognormal survival likelihood, matching R-INLA's `family = "lognormalsurv"`.
Under the canonical `IdentityLink`,

```
log T_i  ~ N(η_i, σ²),    σ² = 1/τ
f(t)     = (1 / (t σ √(2π))) exp(-(log t − η)² / (2σ²))
S(t)     = Φ̄((log t − η) / σ) = Φ((η − log t) / σ)
F(t)     = Φ((log t − η) / σ)
```

`η_i` is the mean of `log T_i` — equivalently, `exp(η_i)` is the median
survival time. The single hyperparameter is the precision `τ`, carried
internally as `θ_ℓ = [log τ]`. The default hyperprior is the PC prior on
standard deviation `σ = τ^(-1/2)` with `P(σ > 1) = 0.01`, matching
R-INLA-modern Gaussian conventions.

`censoring` is an optional `AbstractVector{Censoring}` of length
`length(y)`; when `nothing` (default), every observation is uncensored.
For `INTERVAL` rows, `time_hi[i]` is the upper bound (with `y[i]` the
lower bound).

Closed-form derivatives are provided for all four `Censoring` modes via
the inverse-Mills ratio `h(u) = φ(u)/Φ(u)` (and its first/second
derivatives). Stable evaluation in both tails uses
`Distributions.logpdf` / `logcdf` / `logccdf`.

# Example

```julia
ℓ = LognormalSurvLikelihood(censoring = [NONE, RIGHT, NONE])
```
"""
struct LognormalSurvLikelihood{
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

function LognormalSurvLikelihood(;
        link::AbstractLinkFunction = IdentityLink(),
        censoring = nothing,
        time_hi::Union{Nothing, AbstractVector{<:Real}} = nothing,
        hyperprior::AbstractHyperPrior = PCPrecision(1.0, 0.01))
    link isa IdentityLink ||
        throw(ArgumentError(
            "LognormalSurvLikelihood: only IdentityLink is supported, got $(typeof(link))"))
    return LognormalSurvLikelihood(
        link, _coerce_censoring(censoring), time_hi, hyperprior)
end

link(ℓ::LognormalSurvLikelihood) = ℓ.link
nhyperparameters(::LognormalSurvLikelihood) = 1
initial_hyperparameters(::LognormalSurvLikelihood) = [0.0]   # log τ = 0, τ = 1
log_hyperprior(ℓ::LognormalSurvLikelihood, θ) = log_prior_density(ℓ.hyperprior, θ[1])

# --- inverse-Mills helpers -------------------------------------------
# Numerically stable inverse Mills ratio h(u) = φ(u) / Φ(u), evaluated
# from `logpdf − logcdf` so it is accurate in both tails.
@inline function _imills(u::Real)
    N = Distributions.Normal()
    return exp(Distributions.logpdf(N, u) - Distributions.logcdf(N, u))
end

# h'(u) = -h(u) (u + h(u))
# h''(u) = h(u) (u² + 3 u h(u) + 2 h(u)² - 1)
@inline _imills_prime(u, h) = -h * (u + h)
@inline _imills_dprime(u, h) = h * (u^2 + 3 * u * h + 2 * h^2 - 1)

# --- log-link, all-uncensored fast path -------------------------------
# log f(t) = -log t - 0.5 log(2π) + 0.5 log τ - 0.5 τ (log t − η)²
# ∂η log f =  τ (log t − η)
# ∂²η     = -τ
# ∂³η     =  0

function log_density(ℓ::LognormalSurvLikelihood{IdentityLink, Nothing}, y, η, θ)
    τ = exp(θ[1])
    log_τ = θ[1]
    log2π = log(2π)
    s = zero(promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        y[i] > 0 || return -Inf
        log_y = log(y[i])
        z = log_y - η[i]
        s += -log_y - 0.5 * log2π + 0.5 * log_τ - 0.5 * τ * z^2
    end
    return s
end

function ∇_η_log_density(ℓ::LognormalSurvLikelihood{IdentityLink, Nothing}, y, η, θ)
    τ = exp(θ[1])
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        out[i] = τ * (log(y[i]) - η[i])
    end
    return out
end

function ∇²_η_log_density(ℓ::LognormalSurvLikelihood{IdentityLink, Nothing}, y, η, θ)
    τ = exp(θ[1])
    return fill(-τ, length(y))
end

function ∇³_η_log_density(ℓ::LognormalSurvLikelihood{IdentityLink, Nothing}, y, η, θ)
    T = promote_type(eltype(η), eltype(y), Float64)
    return zeros(T, length(y))
end

# --- log-link, mixed censoring ----------------------------------------

function log_density(ℓ::LognormalSurvLikelihood{IdentityLink, <:AbstractVector{Censoring}},
        y, η, θ)
    τ = exp(θ[1])
    log_τ = θ[1]
    σ = sqrt(1 / τ)
    log2π = log(2π)
    N = Distributions.Normal()
    s = zero(promote_type(eltype(η), eltype(y), Float64))
    @inbounds for i in eachindex(y)
        y[i] > 0 || return -Inf
        c = ℓ.censoring[i]
        log_y = log(y[i])
        η_i = η[i]
        if c === NONE
            z = log_y - η_i
            s += -log_y - 0.5 * log2π + 0.5 * log_τ - 0.5 * τ * z^2
        elseif c === RIGHT
            # log S(t) = log Φ((η − log t) / σ)
            w = (η_i - log_y) / σ
            s += Distributions.logcdf(N, w)
        elseif c === LEFT
            # log F(t) = log Φ((log t − η) / σ)
            v = (log_y - η_i) / σ
            s += Distributions.logcdf(N, v)
        else  # INTERVAL
            t_hi = ℓ.time_hi[i]
            log_thi = log(t_hi)
            w_hi = (log_thi - η_i) / σ
            w_lo = (log_y - η_i) / σ
            # log[Φ(w_hi) - Φ(w_lo)] via stable logsubexp on logcdf.
            s += logsubexp(
                Distributions.logcdf(N, w_hi), Distributions.logcdf(N, w_lo))
        end
    end
    return s
end

function ∇_η_log_density(
        ℓ::LognormalSurvLikelihood{IdentityLink, <:AbstractVector{Censoring}}, y, η, θ)
    τ = exp(θ[1])
    σ = sqrt(1 / τ)
    sqrt_τ = sqrt(τ)
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    N = Distributions.Normal()
    @inbounds for i in eachindex(y)
        c = ℓ.censoring[i]
        η_i = η[i]
        log_y = log(y[i])
        if c === NONE
            out[i] = τ * (log_y - η_i)
        elseif c === RIGHT
            w = (η_i - log_y) * sqrt_τ
            h = _imills(w)
            out[i] = h * sqrt_τ
        elseif c === LEFT
            v = (log_y - η_i) * sqrt_τ
            h = _imills(v)
            out[i] = -h * sqrt_τ
        else  # INTERVAL
            t_hi = ℓ.time_hi[i]
            w_hi = (log(t_hi) - η_i) * sqrt_τ
            w_lo = (log_y - η_i) * sqrt_τ
            φ_hi = exp(Distributions.logpdf(N, w_hi))
            φ_lo = exp(Distributions.logpdf(N, w_lo))
            log_D = logsubexp(
                Distributions.logcdf(N, w_hi), Distributions.logcdf(N, w_lo))
            # ∂η log D = -√τ (φ_hi - φ_lo) / D
            out[i] = -sqrt_τ * (φ_hi - φ_lo) / exp(log_D)
        end
    end
    return out
end

function ∇²_η_log_density(
        ℓ::LognormalSurvLikelihood{IdentityLink, <:AbstractVector{Censoring}}, y, η, θ)
    τ = exp(θ[1])
    sqrt_τ = sqrt(τ)
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    N = Distributions.Normal()
    @inbounds for i in eachindex(y)
        c = ℓ.censoring[i]
        η_i = η[i]
        log_y = log(y[i])
        if c === NONE
            out[i] = -τ
        elseif c === RIGHT
            w = (η_i - log_y) * sqrt_τ
            h = _imills(w)
            out[i] = τ * _imills_prime(w, h)
        elseif c === LEFT
            v = (log_y - η_i) * sqrt_τ
            h = _imills(v)
            out[i] = τ * _imills_prime(v, h)
        else  # INTERVAL
            t_hi = ℓ.time_hi[i]
            w_hi = (log(t_hi) - η_i) * sqrt_τ
            w_lo = (log_y - η_i) * sqrt_τ
            φ_hi = exp(Distributions.logpdf(N, w_hi))
            φ_lo = exp(Distributions.logpdf(N, w_lo))
            D = exp(logsubexp(
                Distributions.logcdf(N, w_hi), Distributions.logcdf(N, w_lo)))
            Δφ = φ_hi - φ_lo
            Δwφ = w_lo * φ_lo - w_hi * φ_hi
            out[i] = τ * (Δwφ * D - Δφ^2) / D^2
        end
    end
    return out
end

function ∇³_η_log_density(
        ℓ::LognormalSurvLikelihood{IdentityLink, <:AbstractVector{Censoring}}, y, η, θ)
    τ = exp(θ[1])
    sqrt_τ = sqrt(τ)
    τ_32 = τ * sqrt_τ                  # τ^(3/2)
    out = similar(η, promote_type(eltype(η), eltype(y), Float64))
    N = Distributions.Normal()
    @inbounds for i in eachindex(y)
        c = ℓ.censoring[i]
        η_i = η[i]
        log_y = log(y[i])
        if c === NONE
            out[i] = 0
        elseif c === RIGHT
            # ∂η w = +√τ → ∂³ = τ^(3/2) h''(w)
            w = (η_i - log_y) * sqrt_τ
            h = _imills(w)
            out[i] = τ_32 * _imills_dprime(w, h)
        elseif c === LEFT
            # ∂η v = -√τ → ∂³ = -τ^(3/2) h''(v)
            v = (log_y - η_i) * sqrt_τ
            h = _imills(v)
            out[i] = -τ_32 * _imills_dprime(v, h)
        else  # INTERVAL
            t_hi = ℓ.time_hi[i]
            w_hi = (log(t_hi) - η_i) * sqrt_τ
            w_lo = (log_y - η_i) * sqrt_τ
            φ_hi = exp(Distributions.logpdf(N, w_hi))
            φ_lo = exp(Distributions.logpdf(N, w_lo))
            D = exp(logsubexp(
                Distributions.logcdf(N, w_hi), Distributions.logcdf(N, w_lo)))
            Δφ = φ_hi - φ_lo
            Δwφ = w_lo * φ_lo - w_hi * φ_hi
            # D'   = -√τ Δφ                  (Δφ  = φ_hi - φ_lo)
            # D''  = -τ (w_hi φ_hi - w_lo φ_lo) = τ Δwφ   (Δwφ = w_lo φ_lo - w_hi φ_hi)
            # D''' = ∂η D''.  Using ∂η(w φ(w)) = √τ φ(w) (w² - 1):
            #        D''' = -τ · [√τ φ_hi (w_hi² - 1) - √τ φ_lo (w_lo² - 1)]
            #             = -τ^(3/2) [Δwwφ - Δφ] = τ^(3/2) (Δφ - Δwwφ)
            Δwwφ = w_hi^2 * φ_hi - w_lo^2 * φ_lo
            Dp  = -sqrt_τ * Δφ
            Dpp = τ * Δwφ
            Dppp = τ_32 * (Δφ - Δwwφ)
            # (log D)''' = D'''/D - 3 D' D'' / D² + 2 (D'/D)³
            out[i] = Dppp / D - 3 * Dp * Dpp / D^2 + 2 * (Dp / D)^3
        end
    end
    return out
end

# --- pointwise log-density --------------------------------------------

function pointwise_log_density(ℓ::LognormalSurvLikelihood{IdentityLink, Nothing},
        y, η, θ)
    τ = exp(θ[1])
    log_τ = θ[1]
    log2π = log(2π)
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        if y[i] > 0
            log_y = log(y[i])
            z = log_y - η[i]
            out[i] = -log_y - 0.5 * log2π + 0.5 * log_τ - 0.5 * τ * z^2
        else
            out[i] = T(-Inf)
        end
    end
    return out
end

function pointwise_log_density(
        ℓ::LognormalSurvLikelihood{IdentityLink, <:AbstractVector{Censoring}},
        y, η, θ)
    τ = exp(θ[1])
    log_τ = θ[1]
    σ = sqrt(1 / τ)
    log2π = log(2π)
    N = Distributions.Normal()
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        c = ℓ.censoring[i]
        η_i = η[i]
        if !(y[i] > 0)
            out[i] = T(-Inf)
            continue
        end
        log_y = log(y[i])
        if c === NONE
            z = log_y - η_i
            out[i] = -log_y - 0.5 * log2π + 0.5 * log_τ - 0.5 * τ * z^2
        elseif c === RIGHT
            w = (η_i - log_y) / σ
            out[i] = Distributions.logcdf(N, w)
        elseif c === LEFT
            v = (log_y - η_i) / σ
            out[i] = Distributions.logcdf(N, v)
        else  # INTERVAL
            t_hi = ℓ.time_hi[i]
            w_hi = (log(t_hi) - η_i) / σ
            w_lo = (log_y - η_i) / σ
            out[i] = logsubexp(
                Distributions.logcdf(N, w_hi), Distributions.logcdf(N, w_lo))
        end
    end
    return out
end

# --- pointwise CDF (PIT) ---------------------------------------------
# F(t) = Φ((log t − η) / σ). All-uncensored only; censored PIT is a v0.2 item.

function pointwise_cdf(::LognormalSurvLikelihood{IdentityLink, Nothing}, y, η, θ)
    τ = exp(θ[1])
    σ = sqrt(1 / τ)
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    N = Distributions.Normal()
    @inbounds for i in eachindex(y)
        out[i] = Distributions.cdf(N, (log(y[i]) - η[i]) / σ)
    end
    return out
end

function pointwise_cdf(
        ℓ::LognormalSurvLikelihood{IdentityLink, <:AbstractVector{Censoring}},
        y, η, θ)
    if any(!=(NONE), ℓ.censoring)
        throw(ArgumentError(
            "pointwise_cdf is undefined for censored observations; " *
            "PIT under censoring (Henderson-Crowther) is deferred to v0.2"))
    end
    τ = exp(θ[1])
    σ = sqrt(1 / τ)
    T = promote_type(eltype(η), eltype(y), Float64)
    out = Vector{T}(undef, length(y))
    N = Distributions.Normal()
    @inbounds for i in eachindex(y)
        out[i] = Distributions.cdf(N, (log(y[i]) - η[i]) / σ)
    end
    return out
end

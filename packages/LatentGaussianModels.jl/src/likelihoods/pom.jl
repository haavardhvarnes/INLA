"""
    POMLikelihood(n_classes; link = LogitLink(),
                  dirichlet_concentration = 3.0)

Proportional-odds ordinal regression in R-INLA's `family = "pom"`
parameterisation. With ordered response `y_i ∈ {1, …, K}` and linear
predictor `η_i` (without an intercept — the cut points absorb it),

    P(y_i ≤ k | η_i, α) = F(α_k − η_i),    k = 1, …, K − 1
    P(y_i = k | η_i, α) = F(α_k − η_i) − F(α_{k−1} − η_i)

with `α_0 = −∞`, `α_K = +∞`, and `F` the standard logistic CDF. The
`K − 1` ordered cut points `α_1 < α_2 < ⋯ < α_{K−1}` carry as
likelihood hyperparameters on the internal scale:

    θ[1]   = α_1                    (real-valued, no transform)
    θ[k]   = log(α_k − α_{k−1}),    k = 2, …, K − 1

so the increments `α_k − α_{k−1} = exp(θ[k])` are strictly positive,
guaranteeing the ordering for any `θ ∈ ℝ^{K−1}`. The struct stores
`n_classes = K`; `nhyperparameters(ℓ) = K − 1`.

# Prior — Dirichlet on the implied class probabilities

R-INLA's `pom` family hard-wires a single Dirichlet prior on the
class probabilities implied by the cut points at `η = 0`,

    π_k(α) = F(α_k) − F(α_{k−1}),    k = 1, …, K

with `α_0 = −∞`, `α_K = +∞`, so `(π_1, …, π_K) ∈ Δ^{K−1}`. The prior
density is

    p(π_1, …, π_K) = Γ(K γ) / Γ(γ)^K · ∏_k π_k^{γ−1}

for a single concentration `γ > 0` (`dirichlet_concentration`,
default 3 — matching R-INLA). Pushing this back to `θ` via the chain
`θ → α → π` adds the Jacobian

    log |det(dπ/dθ)| = Σ_{k=1}^{K−1} log f(α_k)  +  Σ_{k=2}^{K−1} θ_k

where `f(x) = F(x) (1 − F(x))` is the logistic pdf.

The R-INLA pom doc records that R-INLA's internal Dirichlet prior is
"correct only up to a multiplicative constant due to a missing
correction in the log-Jacobian for the sum-to-zero constraint." This
implementation includes the full Jacobian (i.e. is mathematically
exact); the consequence is that the marginal log-likelihood reported
by Julia and R-INLA on the same fit will differ by a fixed θ-
independent additive constant, while every posterior moment of `α`,
`β`, and `θ` matches.

Currently only `LogitLink` is supported (the cumulative link). The
linear predictor `η_i` is fed through identity — there is no separate
"link on η". Probit is the only alternative `R-INLA` supports for
`family = "pom"` and could be added later via dispatch on `link`; the
closed forms below are specific to logistic-CDF arithmetic.

Closed-form derivatives (logit case): with `g_k = F(α_k − η_i) ∈ (0,1)`
and `f_k = g_k (1 − g_k)`, `f'_k = f_k (1 − 2 g_k)` for `k ∈
{1, …, K − 1}`, and `g_0 = f_0 = f'_0 = 0`, `g_K = 1`, `f_K = f'_K = 0`,

    log p_i              = log(g_{y_i} − g_{y_i − 1})
    ∂ log p_i / ∂η_i     = (f_{y_i − 1} − f_{y_i}) / p_i
    ∂² log p_i / ∂η_i²   = (f'_{y_i} − f'_{y_i − 1}) / p_i
                           − (f_{y_i − 1} − f_{y_i})² / p_i²

For the boundary classes the formulae collapse to:

    y_i = 1:   ∂ log p / ∂η = −(1 − g_1),
               ∂² log p / ∂η² = −g_1 (1 − g_1)
    y_i = K:   ∂ log p / ∂η = g_{K−1},
               ∂² log p / ∂η² = −g_{K−1} (1 − g_{K−1})

(∂³ inherits the abstract finite-difference fallback.)

The cumulative-logit likelihood is globally log-concave in `η`, so the
inner Newton step is well-behaved without damping.
"""
struct POMLikelihood{L <: AbstractLinkFunction} <: AbstractLikelihood
    link::L
    n_classes::Int
    dirichlet_concentration::Float64
end

function POMLikelihood(n_classes::Integer;
        link::AbstractLinkFunction=LogitLink(),
        dirichlet_concentration::Real=3.0)
    n_classes >= 2 ||
        throw(ArgumentError("POMLikelihood: n_classes must be ≥ 2, got $n_classes"))
    link isa LogitLink ||
        throw(ArgumentError("POMLikelihood: only LogitLink is supported, got $(typeof(link))"))
    dirichlet_concentration > 0 ||
        throw(ArgumentError("POMLikelihood: dirichlet_concentration must be > 0, got $dirichlet_concentration"))
    return POMLikelihood(link, Int(n_classes), Float64(dirichlet_concentration))
end

link(ℓ::POMLikelihood) = ℓ.link
nhyperparameters(ℓ::POMLikelihood) = ℓ.n_classes - 1

# Initial cut points evenly spaced at α = 0, 1, 2, ..., K − 2 — i.e.
# θ[1] = 0 and θ[k] = 0 (so each increment exp(0) = 1) for k ≥ 2.
initial_hyperparameters(ℓ::POMLikelihood) = zeros(Float64, ℓ.n_classes - 1)

function log_hyperprior(ℓ::POMLikelihood, θ)
    K = ℓ.n_classes
    γ = ℓ.dirichlet_concentration
    α = _pom_cutpoints(θ, K)
    T = eltype(α)
    # F(α_k) at η = 0, k = 1..K−1 — the cumulative class probabilities.
    g = Vector{T}(undef, K - 1)
    @inbounds for k in 1:(K - 1)
        g[k] = _pom_sigmoid(α[k])
    end
    # Implied class probabilities π_k = F(α_k) − F(α_{k−1}), k = 1..K
    # with F(α_0) = 0 and F(α_K) = 1. Compute log π_k directly to keep
    # the tail probability stable: log π_K = log(1 − F(α_{K−1})) =
    # log F(−α_{K−1}) = −log1p(exp(α_{K−1})).
    log_π_sum = log(g[1])
    @inbounds for k in 2:(K - 1)
        log_π_sum += log(g[k] - g[k - 1])
    end
    log_π_sum += log1p(-g[K - 1])
    log_dir = SpecialFunctions.loggamma(K * γ) -
              K * SpecialFunctions.loggamma(γ) +
              (γ - 1) * log_π_sum
    # |det(dπ/dα)| = prod_{k=1..K−1} f(α_k) = prod g_k (1 − g_k)
    log_jac_α = zero(T)
    @inbounds for k in 1:(K - 1)
        log_jac_α += log(g[k] * (1 - g[k]))
    end
    # |det(dα/dθ)| = exp(Σ_{k=2..K−1} θ_k); the leading α_1 = θ_1
    # block contributes 1.
    log_jac_θ = zero(T)
    @inbounds for k in 2:(K - 1)
        log_jac_θ += θ[k]
    end
    return log_dir + log_jac_α + log_jac_θ
end

# Cumulative cut points α from internal coordinates θ.
# α[1] = θ[1], α[k] = α[k − 1] + exp(θ[k]) for k = 2, …, K − 1.
@inline function _pom_cutpoints(θ::AbstractVector{T}, n_classes::Integer) where {T}
    K = n_classes
    α = Vector{T}(undef, K - 1)
    α[1] = θ[1]
    @inbounds for k in 2:(K - 1)
        α[k] = α[k - 1] + exp(θ[k])
    end
    return α
end

# Logistic CDF — same as `inverse_link(LogitLink(), t)` but inlined
# locally to avoid the dispatch overhead inside the inner loops.
@inline function _pom_sigmoid(t::T) where {T <: Real}
    if t >= zero(T)
        return one(T) / (one(T) + exp(-t))
    else
        e = exp(t)
        return e / (one(T) + e)
    end
end

# --- logit-link closed forms ------------------------------------------

function log_density(ℓ::POMLikelihood{LogitLink}, y, η, θ)
    α = _pom_cutpoints(θ, ℓ.n_classes)
    K = ℓ.n_classes
    s = zero(promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        k = Int(y[i])
        (1 <= k <= K) ||
            throw(ArgumentError("POMLikelihood: y[$i] = $(y[i]) outside 1:$K"))
        if k == 1
            g_u = _pom_sigmoid(α[1] - η[i])
            s += log(g_u)
        elseif k == K
            g_l = _pom_sigmoid(α[K - 1] - η[i])
            s += log1p(-g_l)
        else
            g_u = _pom_sigmoid(α[k] - η[i])
            g_l = _pom_sigmoid(α[k - 1] - η[i])
            s += log(g_u - g_l)
        end
    end
    return s
end

function ∇_η_log_density(ℓ::POMLikelihood{LogitLink}, y, η, θ)
    α = _pom_cutpoints(θ, ℓ.n_classes)
    K = ℓ.n_classes
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        k = Int(y[i])
        if k == 1
            g_u = _pom_sigmoid(α[1] - η[i])
            out[i] = -(1 - g_u)
        elseif k == K
            g_l = _pom_sigmoid(α[K - 1] - η[i])
            out[i] = g_l
        else
            g_u = _pom_sigmoid(α[k] - η[i])
            g_l = _pom_sigmoid(α[k - 1] - η[i])
            f_u = g_u * (1 - g_u)
            f_l = g_l * (1 - g_l)
            p = g_u - g_l
            out[i] = (f_l - f_u) / p
        end
    end
    return out
end

function ∇²_η_log_density(ℓ::POMLikelihood{LogitLink}, y, η, θ)
    α = _pom_cutpoints(θ, ℓ.n_classes)
    K = ℓ.n_classes
    out = similar(η, promote_type(eltype(η), Float64))
    @inbounds for i in eachindex(y)
        k = Int(y[i])
        if k == 1
            g_u = _pom_sigmoid(α[1] - η[i])
            out[i] = -g_u * (1 - g_u)
        elseif k == K
            g_l = _pom_sigmoid(α[K - 1] - η[i])
            out[i] = -g_l * (1 - g_l)
        else
            g_u = _pom_sigmoid(α[k] - η[i])
            g_l = _pom_sigmoid(α[k - 1] - η[i])
            f_u = g_u * (1 - g_u)
            f_l = g_l * (1 - g_l)
            fp_u = f_u * (1 - 2 * g_u)
            fp_l = f_l * (1 - 2 * g_l)
            p = g_u - g_l
            out[i] = (fp_u - fp_l) / p - (f_l - f_u)^2 / p^2
        end
    end
    return out
end

# --- pointwise log-density (CPO/WAIC) ---------------------------------

function pointwise_log_density(ℓ::POMLikelihood{LogitLink}, y, η, θ)
    α = _pom_cutpoints(θ, ℓ.n_classes)
    K = ℓ.n_classes
    T = promote_type(eltype(η), Float64)
    out = Vector{T}(undef, length(y))
    @inbounds for i in eachindex(y)
        k = Int(y[i])
        if k == 1
            g_u = _pom_sigmoid(α[1] - η[i])
            out[i] = log(g_u)
        elseif k == K
            g_l = _pom_sigmoid(α[K - 1] - η[i])
            out[i] = log1p(-g_l)
        else
            g_u = _pom_sigmoid(α[k] - η[i])
            g_l = _pom_sigmoid(α[k - 1] - η[i])
            out[i] = log(g_u - g_l)
        end
    end
    return out
end

# Posterior-predictive diagnostics for INLAResult.
#
# All four diagnostics reduce to expectations under the INLA posterior
# `(x, θ) | y` restricted to the linear predictor `η = A x` and the
# likelihood hyperparameters `θ_ℓ`.
#
# - DIC uses a second-order Taylor expansion of `-2 log p(y | η, θ_ℓ)`
#   around the per-θ_k mode, with posterior variance of `η_i` supplied
#   by the Laplace-step precision (kriging-corrected).
# - WAIC, CPO, and PIT are Monte-Carlo estimates over joint posterior
#   draws of `(η, θ_ℓ)`.
#
# References:
#   - Spiegelhalter et al. (2002), DIC.
#   - Watanabe (2010), WAIC.
#   - Held, Schrödle & Rue (2010), CPO / PIT in INLA.

# ---------------------------------------------------------------------
# Posterior sampling of the linear predictor
# ---------------------------------------------------------------------

"""
    posterior_samples_η(rng, res, model; n_samples = 1000)
      -> @NamedTuple{η::Matrix{Float64}, θℓ::Matrix{Float64}}

Joint draws of the linear predictor `η = A x` and the likelihood
hyperparameters `θ_ℓ` from the INLA posterior. Returns an `n_obs ×
n_samples` matrix `η` and an `n_ℓ × n_samples` matrix `θℓ` (empty
rows when the likelihood has no hyperparameters).

The sampler picks `θ_k` from the discrete integration design with
probabilities `res.θ_weights`, then draws `x | θ_k ∼ N(mode_k,
H_k⁻¹)` using the cached Cholesky factor at `θ_k`. If the component
stack declared hard linear constraints, the kriging correction is
applied to each draw so that `C x = e` to working precision.
"""
function posterior_samples_η(rng::Random.AbstractRNG,
        res::INLAResult,
        model::LatentGaussianModel;
        n_samples::Integer=1000)
    n_samples ≥ 1 || throw(ArgumentError("n_samples must be ≥ 1"))
    A = as_matrix(model.mapping)
    n_obs = size(A, 1)
    n_ℓ = n_likelihood_hyperparameters(model)

    η_samples = Matrix{Float64}(undef, n_obs, n_samples)
    θℓ_samples = Matrix{Float64}(undef, n_ℓ, n_samples)

    cw = cumsum(res.θ_weights)
    cw[end] = 1.0

    for s in 1:n_samples
        k = searchsortedfirst(cw, rand(rng))
        k = min(k, length(cw))
        lp = res.laplaces[k]

        x = _sample_laplace(rng, lp)
        η_view = view(η_samples, :, s)
        η_view .= A * x
        joint_apply_copy_contributions!(η_view, model, x, lp.θ)
        if n_ℓ > 0
            @views θℓ_samples[:, s] .= res.θ_points[k][1:n_ℓ]
        end
    end
    return (η=η_samples, θℓ=θℓ_samples)
end

# Draw one sample from N(lp.mode, lp.precision^{-1}) with optional
# kriging correction to satisfy Cx = e.
function _sample_laplace(rng::Random.AbstractRNG, lp::LaplaceResult)
    F = GMRFs.factor(lp.factor)
    n_x = length(lp.mode)
    z = randn(rng, n_x)
    x0 = F.UP \ z                               # Cov(x0) = H^{-1}
    if lp.constraint !== nothing
        U = lp.constraint.U
        W_fact = lp.constraint.W_fact
        C = lp.constraint.C
        x0 .-= U * (W_fact \ (C * x0))          # project onto null(C)
    end
    return lp.mode .+ x0
end

# ---------------------------------------------------------------------
# Deviance Information Criterion (closed-form moment approximation)
# ---------------------------------------------------------------------

"""
    dic(res, model, y) -> @NamedTuple{DIC, pD, D_bar, D_mode}

Spiegelhalter DIC, computed without sampling:

- `D_bar  = E_{x,θ | y}[-2 log p(y | η, θ_ℓ)]` estimated by averaging
  the per-`θ_k` Laplace approximation with a second-order Taylor
  correction in the posterior variance of `η_i`.
- `D_mode = -2 log p(y | η̂, θ̂_ℓ)` evaluated at the posterior means.
- `pD    = D_bar - D_mode`  (effective number of parameters).
- `DIC   = D_bar + pD`.

The second-order correction is

    E[-2 log p(y_i|η_i, θ_ℓ)] ≈ -2 log p(y_i|η̂_i, θ_ℓ)
                               + (-∇²_η log p)_i · Var(η_i | θ_k)

which is exact for Gaussian identity-link and a good approximation
for Poisson/Binomial near the mode. For more complex likelihoods the
MC-based [`waic`](@ref) is a robust alternative.
"""
function dic(res::INLAResult, model::LatentGaussianModel, y)
    A = as_matrix(model.mapping)

    # D at the posterior mean of the linear predictor + full θ̄. With a
    # `Copy` contribution η depends on θ through β; use the effective
    # Jacobian at `res.θ_mean` so D_mode is evaluated at consistent
    # (x̄, θ̄).
    η_mean = A * res.x_mean
    joint_apply_copy_contributions!(η_mean, model, res.x_mean, res.θ_mean)
    D_mode = -2 * joint_log_density(model, y, η_mean, res.θ_mean)

    # D̄ via θ-averaged second-order Taylor.
    D_bar = 0.0
    for k in eachindex(res.laplaces)
        w = res.θ_weights[k]
        w == 0 && continue
        lp = res.laplaces[k]
        θ_k = res.θ_points[k]
        # Per-θ_k effective Jacobian. Equals A for non-Copy models.
        J_k = joint_effective_jacobian(model, θ_k)
        η_k = A * lp.mode
        joint_apply_copy_contributions!(η_k, model, lp.mode, θ_k)
        neg_∇²η = .-joint_∇²_η_log_density(model, y, η_k, θ_k)  # nonneg for canonical links
        var_η_k = _predictor_variance(J_k, lp)
        D_bar += w * (-2 * joint_log_density(model, y, η_k, θ_k) +
                      sum(neg_∇²η .* var_η_k))
    end

    pD = D_bar - D_mode
    return (DIC=D_bar + pD, pD=pD, D_bar=D_bar, D_mode=D_mode)
end

# Diagonal of `A H^{-1} A'` with the constraint correction.
# For small n_obs this is the cleanest path (one matrix solve on H).
function _predictor_variance(A::AbstractMatrix, lp::LaplaceResult)
    # H^{-1} A' as a dense n_x × n_obs matrix. Sparse factor + dense RHS
    # is handled by SuiteSparse's triangular solve.
    M = lp.factor \ Matrix(A')
    main = _diag_A_times(A, M)
    if lp.constraint === nothing
        return main
    end
    # Subtract kriging correction: diag(A U W^{-1} U' A') where
    # U = H^{-1} C'. With B = A U we want diag(B W^{-1} B').
    U = lp.constraint.U
    W_fact = lp.constraint.W_fact
    B = A * U
    corr = B * (W_fact \ Matrix(B'))
    corr_diag = diag(corr)
    return main .- corr_diag
end

# diag(A * M) for sparse A and dense M. Avoids materialising A*M.
function _diag_A_times(A::SparseArrays.AbstractSparseMatrix, M::AbstractMatrix)
    n_obs = size(A, 1)
    out = zeros(Float64, n_obs)
    rows = SparseArrays.rowvals(A)
    vals = SparseArrays.nonzeros(A)
    @inbounds for j in 1:size(A, 2)
        for p in SparseArrays.nzrange(A, j)
            i = rows[p]
            out[i] += vals[p] * M[j, i]
        end
    end
    return out
end

function _diag_A_times(A::AbstractMatrix, M::AbstractMatrix)
    n_obs = size(A, 1)
    return [dot(@view(A[i, :]), @view(M[:, i])) for i in 1:n_obs]
end

# ---------------------------------------------------------------------
# WAIC (Watanabe)
# ---------------------------------------------------------------------

"""
    waic(rng, res, model, y; n_samples = 1000)
      -> @NamedTuple{WAIC, lpd, pWAIC, elpd_WAIC}

Watanabe's widely-applicable information criterion (WAIC-2):

    elpd_WAIC = Σ_i (lpd_i - pWAIC_i)
    WAIC      = -2 · elpd_WAIC

with

    lpd_i   = log E[ p(y_i | η_i, θ_ℓ) ]
    pWAIC_i = Var[ log p(y_i | η_i, θ_ℓ) ]

Expectations are Monte-Carlo under the posterior samples.
"""
function waic(rng::Random.AbstractRNG,
        res::INLAResult,
        model::LatentGaussianModel,
        y;
        n_samples::Integer=1000)
    samples = posterior_samples_η(rng, res, model; n_samples)
    logp = _pointwise_log_density_matrix(model, y, samples)
    n_obs = size(logp, 1)

    lpd = Vector{Float64}(undef, n_obs)
    pw = Vector{Float64}(undef, n_obs)
    @inbounds for i in 1:n_obs
        row = @view logp[i, :]
        lpd[i] = _logsumexp(row) - log(n_samples)
        pw[i] = Statistics.var(row; corrected=true)
    end
    elpd = sum(lpd) - sum(pw)
    return (WAIC=-2 * elpd, lpd=lpd, pWAIC=pw, elpd_WAIC=elpd)
end

function waic(res::INLAResult, model::LatentGaussianModel, y; kwargs...)
    waic(Random.default_rng(), res, model, y; kwargs...)
end

# ---------------------------------------------------------------------
# CPO — conditional predictive ordinate
# ---------------------------------------------------------------------

"""
    cpo(rng, res, model, y; n_samples = 1000)
      -> @NamedTuple{CPO, log_CPO, log_pseudo_marginal}

Held-Schrödle-Rue CPO via the harmonic-mean identity

    CPO_i⁻¹ = E[ 1 / p(y_i | η_i, θ_ℓ) ]

where the expectation is over the posterior `(η_i, θ_ℓ) | y`. The
pseudo-marginal likelihood `Σ log CPO_i` is returned as a summary
log-predictive score.

The harmonic-mean estimator is known to be noisy when a few
observations have very small likelihood values; R-INLA flags these
as "failures" via a failure indicator. We do not replicate that flag
yet — outliers will produce large `log_CPO_i` variance across RNG
seeds. Increasing `n_samples` stabilises the estimator.
"""
function cpo(rng::Random.AbstractRNG,
        res::INLAResult,
        model::LatentGaussianModel,
        y;
        n_samples::Integer=1000)
    samples = posterior_samples_η(rng, res, model; n_samples)
    logp = _pointwise_log_density_matrix(model, y, samples)
    n_obs = size(logp, 1)

    log_cpo = Vector{Float64}(undef, n_obs)
    @inbounds for i in 1:n_obs
        # log CPO_i = -log E[exp(-log p_i)] = -(logsumexp(-log p_i) - log N)
        neg_row = -(@view logp[i, :])
        log_cpo[i] = log(n_samples) - _logsumexp(neg_row)
    end
    return (CPO=exp.(log_cpo), log_CPO=log_cpo,
        log_pseudo_marginal=sum(log_cpo))
end

function cpo(res::INLAResult, model::LatentGaussianModel, y; kwargs...)
    cpo(Random.default_rng(), res, model, y; kwargs...)
end

# ---------------------------------------------------------------------
# PIT — probability integral transform
# ---------------------------------------------------------------------

"""
    pit(rng, res, model, y; n_samples = 1000) -> Vector{Float64}

Posterior-predictive probability integral transform

    PIT_i = E[ F(y_i | η_i, θ_ℓ) ]

evaluated via Monte-Carlo. For a well-specified model PIT values are
approximately uniform on `[0, 1]`; deviations flag miscalibration.
For discrete likelihoods this is the naive (non-randomised) variant
— the CDF is evaluated at the observed integer, so ties at the
endpoints appear but summary statistics still diagnose gross
miscalibration.

Requires the likelihood to implement [`pointwise_cdf`](@ref).
"""
function pit(rng::Random.AbstractRNG,
        res::INLAResult,
        model::LatentGaussianModel,
        y;
        n_samples::Integer=1000)
    samples = posterior_samples_η(rng, res, model; n_samples)
    n_ℓ = n_likelihood_hyperparameters(model)
    n_obs = length(y)
    acc = zeros(Float64, n_obs)
    @inbounds for s in 1:n_samples
        η_s = @view samples.η[:, s]
        θℓ_s = n_ℓ > 0 ? collect(@view samples.θℓ[:, s]) : Float64[]
        acc .+= joint_pointwise_cdf(model, y, η_s, θℓ_s)
    end
    return acc ./ n_samples
end

function pit(res::INLAResult, model::LatentGaussianModel, y; kwargs...)
    pit(Random.default_rng(), res, model, y; kwargs...)
end

# ---------------------------------------------------------------------
# Shared helper: matrix of pointwise log-densities across MC samples.
# ---------------------------------------------------------------------

function _pointwise_log_density_matrix(model::LatentGaussianModel, y,
        samples::NamedTuple)
    n_ℓ = n_likelihood_hyperparameters(model)
    n_obs = length(y)
    n_samples = size(samples.η, 2)
    out = Matrix{Float64}(undef, n_obs, n_samples)
    @inbounds for s in 1:n_samples
        η_s = @view samples.η[:, s]
        θℓ_s = n_ℓ > 0 ? collect(@view samples.θℓ[:, s]) : Float64[]
        out[:, s] .= joint_pointwise_log_density(model, y, η_s, θℓ_s)
    end
    return out
end

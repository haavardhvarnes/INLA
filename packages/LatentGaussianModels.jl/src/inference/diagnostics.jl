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

"""
    posterior_sample(rng, res, model; n_samples = 1000)
      -> @NamedTuple{x::Matrix{Float64}, θ::Matrix{Float64}}

Joint draws `(x, θ) ∼ π̂(· | y)` from the INLA posterior. The output
matrices have one column per draw:

- `x` is `n_x × n_samples` — full latent vector (component blocks
  concatenated in joint order).
- `θ` is `n_θ × n_samples` — full hyperparameter vector (likelihood
  block first, then per-component blocks; same ordering as
  [`hyperparameters`](@ref) and `res.θ_points`).

The sampler picks `θ_k` from the discrete integration design with
probabilities `res.θ_weights`, then draws `x | θ_k ∼ N(mode_k,
H_k⁻¹)` using the cached Cholesky factor at `θ_k`. If the component
stack declared hard linear constraints, the kriging correction is
applied so each `x` sample satisfies `C x = e` to working precision.

Use [`posterior_samples_η`](@ref) instead when you want draws of
`η = A x` and the likelihood-only hyperparameters `θ_ℓ` (the form
that drives WAIC / CPO / PIT). `posterior_sample` is the right
building block for joint-`x` summaries (e.g. random-effect contrast
posteriors) and Stan/NUTS triangulation.

When `n_hyperparameters(model) == 0` (the dim(θ)=0 fast path used
by ADR-024's multinomial-via-Poisson), the returned `θ` matrix has
zero rows.
"""
function posterior_sample(rng::Random.AbstractRNG,
        res::INLAResult,
        model::LatentGaussianModel;
        n_samples::Integer=1000)
    n_samples ≥ 1 || throw(ArgumentError("n_samples must be ≥ 1"))
    n_x = model.n_x
    n_θ = n_hyperparameters(model)

    x_samples = Matrix{Float64}(undef, n_x, n_samples)
    θ_samples = Matrix{Float64}(undef, n_θ, n_samples)

    cw = cumsum(res.θ_weights)
    cw[end] = 1.0

    for s in 1:n_samples
        k = searchsortedfirst(cw, rand(rng))
        k = min(k, length(cw))
        lp = res.laplaces[k]

        x_view = view(x_samples, :, s)
        x_view .= _sample_laplace(rng, lp)
        if n_θ > 0
            @views θ_samples[:, s] .= res.θ_points[k]
        end
    end
    return (x=x_samples, θ=θ_samples)
end

# ---------------------------------------------------------------------
# Posterior predictive at a new observation mapping
# ---------------------------------------------------------------------

"""
    posterior_predictive(rng, res, model, mapping; n_samples = 1000)
      -> @NamedTuple{x::Matrix{Float64}, θ::Matrix{Float64}, η::Matrix{Float64}}

Posterior predictive samples of the linear predictor `η_new = A_new x`
at a new observation mapping. `mapping` may be either an
[`AbstractObservationMapping`](@ref) (e.g. [`LinearProjector`](@ref),
[`IdentityMapping`](@ref), [`StackedMapping`](@ref)) or an
`AbstractMatrix` `A_new` — matrices are wrapped in `LinearProjector`
automatically, mirroring `LatentGaussianModel`'s convenience
constructor.

The function returns the joint draws `(x, θ)` from
[`posterior_sample`](@ref) plus `η::Matrix{Float64}` of size
`nrows(mapping) × n_samples`. Each column is `η_s = mapping * x_s`.

`η` is the foundation for downstream predictive inference: applying
the likelihood's inverse link gives `μ` posterior samples, and
sampling `y_new ∼ p(y | η, θ)` per likelihood gives full posterior
predictive `y` samples. The latter is left to per-likelihood
sampling code (Phase K follow-up); this function ships the
likelihood-agnostic part.

`mapping` must satisfy `ncols(mapping) == n_latent(model)`.

# Example

```julia
res = inla(model, y)
# Predict at new covariate rows X_new (rows match the original design):
draws = posterior_predictive(rng, res, model, X_new; n_samples = 500)
μ_new = inverse_link.(link(model.likelihood), draws.η)
```
"""
function posterior_predictive(rng::Random.AbstractRNG,
        res::INLAResult,
        model::LatentGaussianModel,
        mapping::AbstractObservationMapping;
        n_samples::Integer=1000)
    n_samples ≥ 1 || throw(ArgumentError("n_samples must be ≥ 1"))
    ncols(mapping) == model.n_x ||
        throw(DimensionMismatch("mapping has ncols=$(ncols(mapping)); " *
                                "model has n_x=$(model.n_x)"))

    draws = posterior_sample(rng, res, model; n_samples=n_samples)
    n_new = nrows(mapping)
    η = Matrix{Float64}(undef, n_new, n_samples)

    x_buf = Vector{Float64}(undef, model.n_x)
    η_buf = Vector{Float64}(undef, n_new)
    for s in 1:n_samples
        @views copyto!(x_buf, draws.x[:, s])
        apply!(η_buf, mapping, x_buf)
        @views η[:, s] .= η_buf
    end
    return (x=draws.x, θ=draws.θ, η=η)
end

function posterior_predictive(rng::Random.AbstractRNG,
        res::INLAResult,
        model::LatentGaussianModel,
        A::AbstractMatrix;
        kwargs...)
    return posterior_predictive(rng, res, model, LinearProjector(A); kwargs...)
end

"""
    posterior_predictive_y(rng, res, model; n_samples = 1000)
      -> @NamedTuple{x::Matrix{Float64}, θ::Matrix{Float64},
                     η::Matrix{Float64}, y_rep::Matrix{Float64}}

Posterior-predictive samples on the *response scale*. Extends
[`posterior_predictive`](@ref) with a per-likelihood
[`sample_y`](@ref) draw, returning the simulated replicate observations
`y_rep` of size `n_obs × n_samples` alongside the underlying joint
draws `(x, θ, η)`.

For multi-likelihood models the response sampler dispatches per block:
each likelihood `m.likelihoods[k]` draws a replicate of length
`length(m.block_rows[k])` from `p(y | η_block, θ_ℓ_k)` using its own
metadata (Binomial `n_trials`, Poisson/NegBin offset `E`, …) read
directly from the likelihood instance.

`y_rep` is always returned as `Matrix{Float64}` regardless of the
likelihood family — counts arrive as integer-valued floats, continuous
likelihoods as real-valued floats — to keep multi-likelihood blocks
typed-uniformly.

Used by posterior-predictive checks (Gelman et al. 2014; bayesplot's
`pp_check`). To overlay densities of `y_rep` against the observed
response, draw `n_samples ≈ 200–1000` and plot a subset of the columns
of `y_rep` against `y_obs`.

# Example

```julia
res = inla(model, y)
draws = posterior_predictive_y(rng, res, model; n_samples = 400)
# draws.y_rep is n_obs × 400 — each column a posterior-predictive
# replicate of the full observation vector.
```

# Note

Likelihoods without `sample_y` defined (currently: censored survival
families and zero-inflated likelihoods) raise `ArgumentError`. Closed-form
samplers ship for `GaussianLikelihood`, `PoissonLikelihood`,
`BinomialLikelihood`, `NegativeBinomialLikelihood`, and `GammaLikelihood`.
"""
function posterior_predictive_y(rng::Random.AbstractRNG,
        res::INLAResult,
        model::LatentGaussianModel;
        n_samples::Integer=1000)
    n_samples ≥ 1 || throw(ArgumentError("n_samples must be ≥ 1"))

    draws = posterior_sample(rng, res, model; n_samples=n_samples)
    A = as_matrix(model.mapping)
    n_obs = size(A, 1)

    η = Matrix{Float64}(undef, n_obs, n_samples)
    y_rep = Matrix{Float64}(undef, n_obs, n_samples)

    x_buf = Vector{Float64}(undef, model.n_x)
    η_buf = Vector{Float64}(undef, n_obs)

    @inbounds for s in 1:n_samples
        @views copyto!(x_buf, draws.x[:, s])
        η_buf .= A * x_buf
        @views joint_apply_copy_contributions!(η_buf, model, x_buf, draws.θ[:, s])
        @views η[:, s] .= η_buf

        for (k, ℓ) in enumerate(model.likelihoods)
            rows = model.block_rows[k]
            θ_ℓ_k = collect(view(draws.θ, model.likelihood_θ_ranges[k], s))
            y_block = sample_y(rng, ℓ, view(η_buf, rows), θ_ℓ_k)
            @views y_rep[rows, s] .= y_block
        end
    end
    return (x=draws.x, θ=draws.θ, η=η, y_rep=y_rep)
end

function posterior_predictive_y(res::INLAResult, model::LatentGaussianModel;
        kwargs...)
    return posterior_predictive_y(Random.default_rng(), res, model; kwargs...)
end

"""
    pp_check(rng, res, model, y_obs; n_samples = 400)
      -> @NamedTuple{y::AbstractVector, y_rep::Matrix{Float64}}

Posterior-predictive-check data: the observed response `y_obs` paired
with `n_samples` replicate response vectors `y_rep` drawn under the
fitted model. Convenience wrapper over [`posterior_predictive_y`](@ref)
that drops the `(x, θ, η)` joint draws and keeps just the data needed
for pp-check overlays.

`n_samples` defaults to `400` — enough for stable density estimates,
small enough that overlaying every column of `y_rep` is reasonable.

# Plotting (Makie)

```julia
using GLMakie, LatentGaussianModels
ck = pp_check(rng, res, model, y_obs; n_samples = 400)

fig = Figure(); ax = Axis(fig[1, 1])
for j in axes(ck.y_rep, 2)[1:50]                         # 50 replicates
    density!(ax, ck.y_rep[:, j]; color = (:gray, 0.4))
end
density!(ax, ck.y; color = :black, linewidth = 3)
fig
```

The y-axis covers `min(y_rep)` to `max(y_rep)` automatically; a clear
visual gap between the observed and replicated densities indicates
mis-specification (location, scale, or shape).
"""
function pp_check(rng::Random.AbstractRNG,
        res::INLAResult,
        model::LatentGaussianModel,
        y_obs;
        n_samples::Integer=400)
    length(y_obs) == n_observations(model) ||
        throw(DimensionMismatch("y_obs has length $(length(y_obs)); " *
                                "model has $(n_observations(model)) observation rows"))
    draws = posterior_predictive_y(rng, res, model; n_samples=n_samples)
    return (y=y_obs, y_rep=draws.y_rep)
end

function pp_check(res::INLAResult, model::LatentGaussianModel, y_obs; kwargs...)
    return pp_check(Random.default_rng(), res, model, y_obs; kwargs...)
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
# PSIS-LOO — Pareto-smoothed importance sampling LOO-CV
# ---------------------------------------------------------------------

"""
    psis_loo(rng, res, model, y; n_samples = 1000)
      -> @NamedTuple{elpd_loo, looic, pointwise_elpd_loo, pointwise_p_loo,
                     p_loo, pareto_k}

Pareto-smoothed importance sampling estimate of the leave-one-out
expected log pointwise predictive density (Vehtari, Gelman & Gabry
2017; Vehtari, Simpson, Gelman, Yao & Gabry 2024).

Returns:

- `elpd_loo` — sum of pointwise LOO elpd (higher is better).
- `looic     = -2 · elpd_loo` — information-criterion form.
- `pointwise_elpd_loo` — per-observation elpd_loo, length `n_obs`.
- `pointwise_p_loo` — per-observation effective parameter contribution.
- `p_loo` — sum of pointwise effective parameters.
- `pareto_k` — vector of fitted Pareto shape parameters `k̂_i`. Values
  above 0.7 indicate the LOO importance ratios for that observation
  have a heavy tail and the PSIS estimate is unreliable; refit-based
  LOO is the gold-standard remedy. Values above 0.5 (but below 0.7)
  are a soft warning.

PSIS-LOO supplements [`cpo`](@ref): both target the per-observation
predictive distribution but PSIS-LOO is more robust to outliers and
provides the diagnostic `k̂` for failure detection.

This function requires the **PSIS.jl** weakdep:

```julia
using PSIS, LatentGaussianModels
res = inla(model, y)
loo = psis_loo(rng, res, model, y; n_samples = 2000)
```

Without PSIS loaded a `MethodError` is raised.

# References
- Vehtari, A., Gelman, A. & Gabry, J. (2017). Practical Bayesian
  model evaluation using leave-one-out cross-validation and WAIC.
  *Statistics and Computing*, 27(5), 1413-1432.
- Vehtari, A., Simpson, D., Gelman, A., Yao, Y. & Gabry, J. (2024).
  Pareto smoothed importance sampling. *JMLR*, 25(72), 1-58.
"""
function psis_loo end

function psis_loo(res::INLAResult, model::LatentGaussianModel, y; kwargs...)
    psis_loo(Random.default_rng(), res, model, y; kwargs...)
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

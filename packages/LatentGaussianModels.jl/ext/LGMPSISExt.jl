"""
    LGMPSISExt

Weakdep extension that activates [`LatentGaussianModels.psis_loo`](@ref)
when `PSIS.jl` is loaded. The PSIS algorithm itself (Pareto smoothing of
importance ratios, fitting of the generalised Pareto distribution to the
upper tail) lives in `PSIS.jl`; this extension is the thin glue that
wires INLA posterior samples into the PSIS API and turns the smoothed
weights into LOO-CV summaries.
"""
module LGMPSISExt

using LatentGaussianModels
using LatentGaussianModels: INLAResult, LatentGaussianModel,
                            posterior_samples_η,
                            _pointwise_log_density_matrix,
                            _logsumexp
using Random
using PSIS: psis

function LatentGaussianModels.psis_loo(rng::Random.AbstractRNG,
        res::INLAResult,
        model::LatentGaussianModel,
        y;
        n_samples::Integer=1000)
    n_samples ≥ 1 || throw(ArgumentError("n_samples must be ≥ 1"))

    samples = posterior_samples_η(rng, res, model; n_samples=n_samples)
    log_lik = _pointwise_log_density_matrix(model, y, samples)
    n_obs, S = size(log_lik)

    # PSIS expects (ndraws, nchains, nparams). Use nchains = 1 and
    # treat each observation as a separate `param`. log_ratios are
    # `-log_lik` (the LOO importance ratio in log space; the constant
    # offset cancels under self-normalisation).
    log_ratios = Array{Float64}(undef, S, 1, n_obs)
    @inbounds for i in 1:n_obs, s in 1:S
        log_ratios[s, 1, i] = -log_lik[i, s]
    end

    psis_result = psis(log_ratios)
    log_w = psis_result.log_weights              # normalised, sums to 1 per obs
    pareto_shape = psis_result.pareto_shape
    pareto_k = pareto_shape isa AbstractVector ? collect(pareto_shape) :
               fill(Float64(pareto_shape), n_obs)

    elpd_loo_pw = Vector{Float64}(undef, n_obs)
    lpd_pw = Vector{Float64}(undef, n_obs)
    log_S = log(S)
    @inbounds for i in 1:n_obs
        ll = @view log_lik[i, :]
        lw = @view log_w[:, 1, i]
        elpd_loo_pw[i] = _logsumexp(lw .+ ll)
        lpd_pw[i] = _logsumexp(ll) - log_S
    end

    p_loo_pw = lpd_pw .- elpd_loo_pw
    elpd_loo = sum(elpd_loo_pw)
    p_loo = sum(p_loo_pw)
    return (elpd_loo=elpd_loo, looic=-2 * elpd_loo,
            pointwise_elpd_loo=elpd_loo_pw,
            pointwise_p_loo=p_loo_pw,
            p_loo=p_loo,
            pareto_k=pareto_k)
end

end # module

"""
    EmpiricalBayes(; laplace = Laplace(), θ0 = nothing, optim_options = NamedTuple())

Empirical Bayes inference: plug-in estimate of the hyperparameters,

    θ̂ = argmax_θ log π(θ | y)  ≈  argmax_θ [ log p(y | θ) + log π(θ) ],

using the Laplace approximation for `log p(y | θ)`. The final result is
the Laplace fit at `θ̂` plus the mode and Hessian of the hyperparameter
log-posterior (for reporting only — no integration is performed, by
contrast with full `INLA`).

Optimisation is performed through the SciML Optimization.jl frontend;
the default algorithm is `OptimizationOptimJL.LBFGS()`. Custom options
may be supplied via `optim_options` (forwarded to `Optimization.solve`).
"""
Base.@kwdef struct EmpiricalBayes{S <: Laplace} <: AbstractInferenceStrategy
    laplace::S = Laplace()
    θ0::Union{Nothing, Vector{Float64}} = nothing
    optim_options::NamedTuple = NamedTuple()
end

"""
    EmpiricalBayesResult <: AbstractInferenceResult

- `θ̂::Vector{Float64}` — mode of `log π(θ | y)` on the internal scale.
- `laplace::LaplaceResult` — Laplace fit at `θ̂`.
- `log_marginal::Float64` — `log p(y | θ̂)`.
- `optim_result` — raw Optimization.jl solution (for diagnostics).
"""
struct EmpiricalBayesResult <: AbstractInferenceResult
    θ̂::Vector{Float64}
    laplace::LaplaceResult
    log_marginal::Float64
    optim_result::Any
end

"""
    fit(m::LatentGaussianModel, y, strategy::EmpiricalBayes) -> EmpiricalBayesResult

Empirical-Bayes fit. Convenience alias: `empirical_bayes(m, y; kwargs...)`.
"""
function fit(m::LatentGaussianModel, y, strategy::EmpiricalBayes)
    θ0 = strategy.θ0 === nothing ? initial_hyperparameters(m) : copy(strategy.θ0)

    neg_log_posterior = (θ, _p) -> begin
        local res
        try
            res = laplace_mode(m, y, θ; strategy = strategy.laplace)
        catch
            return Inf
        end
        !isfinite(res.log_marginal) && return Inf
        return -(res.log_marginal + log_hyperprior(m, θ))
    end

    optf = Optimization.OptimizationFunction(neg_log_posterior,
                                             Optimization.AutoFiniteDiff())
    prob = Optimization.OptimizationProblem(optf, θ0, nothing)
    opt_res = Optimization.solve(prob, OptimizationOptimJL.LBFGS();
                                 strategy.optim_options...)
    θ̂ = collect(opt_res.u)
    final = laplace_mode(m, y, θ̂; strategy = strategy.laplace)

    return EmpiricalBayesResult(θ̂, final, final.log_marginal, opt_res)
end

"""
    fit(m, y) -> EmpiricalBayesResult

Convenience: default strategy is `EmpiricalBayes()`.
"""
fit(m::LatentGaussianModel, y) = fit(m, y, EmpiricalBayes())

"""
    empirical_bayes(m, y; kwargs...)

Alias for `fit(m, y, EmpiricalBayes(; kwargs...))`.
"""
empirical_bayes(m::LatentGaussianModel, y; kwargs...) =
    fit(m, y, EmpiricalBayes(; kwargs...))

"""
    laplace(m, y, θ; kwargs...)

Laplace fit at fixed `θ`. Returns a `LaplaceResult`.
"""
laplace(m::LatentGaussianModel, y, θ::AbstractVector{<:Real}; kwargs...) =
    laplace_mode(m, y, θ; strategy = Laplace(; kwargs...))

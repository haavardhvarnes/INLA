"""
    nuts_sample(model::LatentGaussianModel, y, n_samples;
                n_adapts            = div(n_samples, 2),
                init_θ              = nothing,
                init_from_inla      = nothing,
                target_acceptance   = 0.8,
                rng                 = Random.default_rng(),
                laplace             = Laplace(),
                drop_warmup         = true,
                progress            = false)
        -> MCMCChains.Chains

Run AdvancedHMC NUTS on the INLA hyperparameter posterior
`log π(θ | y) = log p(y | θ) + log π(θ)`. The sampler walks `θ` only —
the latent vector `x` is integrated out at each leapfrog step by the
inner Laplace approximation. This is the "INLA on θ" arm of the tier-3
triangulation tests against R-INLA's grid integration.

### Initialisation

`init_θ` overrides everything. Otherwise:

- if `init_from_inla::INLAResult`, use `init_from_inla.θ̂`,
- if `init_from_inla === true` (default-style), error — there is no
  fit to read from; pass an `INLAResult` explicitly,
- otherwise fall back to `initial_hyperparameters(model)`.

The default (`init_from_inla === nothing`) is the cold-start fallback.
For triangulation tests, pass the actual `INLAResult` so HMC starts
near the mode and warm-up doesn't have to find it.

### Returned chain

`MCMCChains.Chains` with one column per hyperparameter. Column names
match `_hyperparameter_names(model)` (e.g. `"likelihood[1]"`,
`"IID[2]"`).

### Tolerances

NUTS step-size adaptation targets `target_acceptance` (Stan default
0.8). Each `logdensity` call costs a full Laplace fit; for typical
LGMs a few hundred adapts and a few hundred kept draws is enough for
the tier-3 mean / sd cross-check.
"""
function nuts_sample(model::LatentGaussianModel, y, n_samples::Integer;
        n_adapts::Integer=div(n_samples, 2),
        init_θ::Union{Nothing, AbstractVector{<:Real}}=nothing,
        init_from_inla::Union{Nothing, INLAResult}=nothing,
        target_acceptance::Real=0.8,
        rng::AbstractRNG=default_rng(),
        laplace::Laplace=Laplace(),
        drop_warmup::Bool=true,
        progress::Bool=false)
    n_samples ≥ 1 || throw(ArgumentError("n_samples must be ≥ 1"))
    n_adapts ≥ 0 || throw(ArgumentError("n_adapts must be ≥ 0"))
    0 < target_acceptance < 1 ||
        throw(ArgumentError("target_acceptance must be in (0, 1)"))

    ld = INLALogDensity(model, y; laplace=laplace)
    D = LogDensityProblems.dimension(ld)

    init = if init_θ !== nothing
        length(init_θ) == D ||
            throw(DimensionMismatch("init_θ has length $(length(init_θ)); " *
                                    "model has $D hyperparameters"))
        collect(Float64, init_θ)
    elseif init_from_inla !== nothing
        length(init_from_inla.θ̂) == D ||
            throw(DimensionMismatch("init_from_inla.θ̂ has length " *
                                    "$(length(init_from_inla.θ̂)); model has " *
                                    "$D hyperparameters"))
        collect(Float64, init_from_inla.θ̂)
    else
        collect(Float64, initial_hyperparameters(model))
    end

    metric = DiagEuclideanMetric(D)
    hamiltonian = Hamiltonian(metric, ld)
    ϵ0 = find_good_stepsize(rng, hamiltonian, init)
    integrator = Leapfrog(ϵ0)
    kernel = AdvancedHMC.HMCKernel(
        AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(
        integrator, GeneralisedNoUTurn()))
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric),
        StepSizeAdaptor(target_acceptance, integrator))

    # AdvancedHMC's `n_samples` is total iterations including warmup; we
    # want `n_samples` to mean kept-post-warmup samples (Stan convention).
    total = drop_warmup ? n_adapts + n_samples : n_samples
    samples, _stats = AdvancedHMC.sample(rng, hamiltonian, kernel, init,
        total, adaptor, n_adapts;
        drop_warmup=drop_warmup,
        verbose=false,
        progress=progress)

    n_kept = length(samples)
    arr = Array{Float64, 3}(undef, n_kept, D, 1)
    for i in 1:n_kept, j in 1:D
        arr[i, j, 1] = samples[i][j]
    end
    names = Symbol.(LatentGaussianModels._hyperparameter_names(model))
    return Chains(arr, names)
end

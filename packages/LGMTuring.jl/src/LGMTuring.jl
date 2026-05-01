module LGMTuring

using LatentGaussianModels: LatentGaussianModels, LatentGaussianModel,
                            INLAResult, INLALogDensity, Laplace, initial_hyperparameters,
                            n_hyperparameters, hyperparameters
using AdvancedHMC: AdvancedHMC, DiagEuclideanMetric, Hamiltonian, Leapfrog,
                   GeneralisedNoUTurn, StanHMCAdaptor, MassMatrixAdaptor, StepSizeAdaptor,
                   find_good_stepsize
using LogDensityProblems: LogDensityProblems
using MCMCChains: Chains
using Random: AbstractRNG, default_rng
using Statistics: mean, std

include("logdensity.jl")
include("sample_nuts.jl")
include("compare.jl")

export INLALogDensity, inla_log_density, nuts_sample, compare_posteriors

end # module LGMTuring

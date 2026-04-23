"""
    LatentGaussianModels

Julia-native implementation of latent Gaussian models and their
approximate-Bayesian inference (INLA-style). Consumes the sparse
precision numerics from `GMRFs.jl`; owns the likelihood, hyperprior,
component, and inference-strategy abstractions.

See `CLAUDE.md` and `plans/plan.md` for the design document.
"""
module LatentGaussianModels

using LinearAlgebra
using SparseArrays
using Random
using Statistics

using Distributions: Distributions
using GMRFs
using GMRFs: AbstractGMRFGraph, GMRFGraph, AbstractGMRF, NoConstraint,
             LinearConstraint, FactorCache
using Optimization: Optimization
using OptimizationOptimJL: OptimizationOptimJL
using LogDensityProblems: LogDensityProblems

# --- hyperpriors (loaded first; likelihoods reference them) -----------
include("priors/abstract.jl")
include("priors/pc.jl")

# --- link functions + likelihoods -------------------------------------
include("likelihoods/links.jl")
include("likelihoods/abstract.jl")
include("likelihoods/gaussian.jl")
include("likelihoods/poisson.jl")
include("likelihoods/binomial.jl")

# --- components -------------------------------------------------------
include("components/abstract.jl")
include("components/intercept.jl")
include("components/iid.jl")
include("components/rw.jl")
include("components/ar1.jl")
include("components/besag.jl")

# --- model + inference ------------------------------------------------
include("model.jl")
include("inference/abstract.jl")
include("inference/laplace.jl")
include("inference/empirical_bayes.jl")

# Link functions
export AbstractLinkFunction, IdentityLink, LogLink, LogitLink,
       ProbitLink, CloglogLink
export inverse_link, ∂inverse_link, ∂²inverse_link

# Likelihoods
export AbstractLikelihood, GaussianLikelihood, PoissonLikelihood,
       BinomialLikelihood
export log_density, ∇_η_log_density, ∇²_η_log_density, link

# Hyperpriors
export AbstractHyperPrior
export PCPrecision, GammaPrecision, LogNormalPrecision, WeakPrior
export log_prior_density, user_scale, prior_name

# Components
export AbstractLatentComponent
export Intercept, FixedEffects, IID, RW1, RW2, AR1, Besag
export precision_matrix, initial_hyperparameters, nhyperparameters,
       log_hyperprior, prior_mean

# Model + inference
export LatentGaussianModel, n_latent, n_hyperparameters
export joint_precision
export AbstractInferenceStrategy, AbstractInferenceResult
export Laplace, LaplaceResult, laplace_mode
export EmpiricalBayes, EmpiricalBayesResult
export fit, empirical_bayes, laplace

end # module

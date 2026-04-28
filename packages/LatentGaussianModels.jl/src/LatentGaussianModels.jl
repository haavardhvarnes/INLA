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
using Printf: @sprintf
using SparseArrays
using Random
using Statistics

using Distributions: Distributions
using FastGaussQuadrature: FastGaussQuadrature
using FiniteDiff: FiniteDiff
using GMRFs
using GMRFs: AbstractGMRFGraph, GMRFGraph, AbstractGMRF, NoConstraint,
             LinearConstraint, FactorCache
using Optimization: Optimization
using OptimizationOptimJL: OptimizationOptimJL
using LogDensityProblems: LogDensityProblems

# --- hyperpriors (loaded first; likelihoods reference them) -----------
include("priors/abstract.jl")
include("priors/pc.jl")
include("priors/logit_beta.jl")
include("priors/bym2_phi.jl")

# --- link functions + likelihoods -------------------------------------
include("likelihoods/links.jl")
include("likelihoods/abstract.jl")
include("likelihoods/gaussian.jl")
include("likelihoods/poisson.jl")
include("likelihoods/binomial.jl")
include("likelihoods/negbinomial.jl")
include("likelihoods/gamma.jl")

# --- components -------------------------------------------------------
include("components/abstract.jl")
include("components/intercept.jl")
include("components/iid.jl")
include("components/rw.jl")
include("components/ar1.jl")
include("components/seasonal.jl")
include("components/besag.jl")
include("components/bym.jl")
include("components/bym2.jl")
include("components/leroux.jl")
include("components/generic0.jl")
include("components/generic1.jl")

# --- model + inference ------------------------------------------------
include("model.jl")
include("inference/abstract.jl")
include("inference/constraints.jl")
include("inference/laplace.jl")
include("inference/empirical_bayes.jl")
include("inference/integration.jl")
include("inference/simplified_laplace_correction.jl")
include("inference/inla.jl")
include("inference/marginals.jl")
include("inference/accessors.jl")
include("inference/diagnostics.jl")
include("inference/log_density.jl")

# Link functions
export AbstractLinkFunction, IdentityLink, LogLink, LogitLink,
       ProbitLink, CloglogLink
export inverse_link, ∂inverse_link, ∂²inverse_link

# Likelihoods
export AbstractLikelihood, GaussianLikelihood, PoissonLikelihood,
       BinomialLikelihood, NegativeBinomialLikelihood, GammaLikelihood
export log_density, ∇_η_log_density, ∇²_η_log_density, ∇³_η_log_density, link
export pointwise_log_density, pointwise_cdf

# Hyperpriors
export AbstractHyperPrior
export PCPrecision, GammaPrecision, LogNormalPrecision, WeakPrior
export PCBYM2Phi, LogitBeta
export log_prior_density, user_scale, prior_name

# Components
export AbstractLatentComponent
export Intercept, FixedEffects, IID, RW1, RW2, AR1, Seasonal, Besag, BYM, BYM2,
       Leroux, Generic0, Generic1
export precision_matrix, initial_hyperparameters, nhyperparameters,
       log_hyperprior, prior_mean

# Model + inference
export LatentGaussianModel, n_latent, n_hyperparameters
export joint_precision
export AbstractInferenceStrategy, AbstractInferenceResult
export AbstractIntegrationScheme, Grid, GaussHermite, CCD
export Laplace, LaplaceResult, laplace_mode
export EmpiricalBayes, EmpiricalBayesResult
export INLA, INLAResult
export fit, empirical_bayes, laplace, inla
export posterior_marginal_x, posterior_marginal_θ
export fixed_effects, random_effects, hyperparameters
export log_marginal_likelihood, component_range
export posterior_samples_η, dic, waic, cpo, pit
export inla_summary
export INLALogDensity, sample_conditional

end # module

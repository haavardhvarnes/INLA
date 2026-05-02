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
using SpecialFunctions: SpecialFunctions
using GMRFs: AbstractGMRFGraph, GMRFGraph, AbstractGMRF, NoConstraint,
             LinearConstraint, FactorCache
using Optimization: Optimization
using OptimizationOptimJL: OptimizationOptimJL
using LogDensityProblems: LogDensityProblems

# --- hyperpriors (loaded first; likelihoods reference them) -----------
include("priors/abstract.jl")
include("priors/pc.jl")
include("priors/pc_alphaw.jl")
include("priors/pc_cor0.jl")
include("priors/logit_beta.jl")
include("priors/bym2_phi.jl")
include("priors/gaussian_internal.jl")

# --- link functions + likelihoods -------------------------------------
include("likelihoods/links.jl")
include("likelihoods/abstract.jl")
include("likelihoods/gaussian.jl")
include("likelihoods/poisson.jl")
include("likelihoods/binomial.jl")
include("likelihoods/negbinomial.jl")
include("likelihoods/gamma.jl")
include("likelihoods/survival/_censoring.jl")
include("likelihoods/survival/exponential.jl")
include("likelihoods/survival/coxph.jl")
include("likelihoods/survival/weibull.jl")
include("likelihoods/survival/lognormal.jl")
include("likelihoods/survival/gamma_surv.jl")
include("likelihoods/survival/weibull_cure.jl")
include("likelihoods/zero_inflated/_helpers.jl")
include("likelihoods/zero_inflated/poisson.jl")
include("likelihoods/zero_inflated/binomial.jl")
include("likelihoods/zero_inflated/negbinomial.jl")
include("likelihoods/copy.jl")

# --- components -------------------------------------------------------
include("components/abstract.jl")
include("components/intercept.jl")
include("components/iid.jl")
include("components/iidnd.jl")
include("components/rw.jl")
include("components/ar1.jl")
include("components/seasonal.jl")
include("components/besag.jl")
include("components/bym.jl")
include("components/bym2.jl")
include("components/leroux.jl")
include("components/generic0.jl")
include("components/generic1.jl")

# --- observation mapping (load before model.jl) -----------------------
include("observation_mapping.jl")

# --- model + inference ------------------------------------------------
include("model.jl")
include("joint_likelihood.jl")
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
       BinomialLikelihood, NegativeBinomialLikelihood, GammaLikelihood,
       ExponentialLikelihood, WeibullLikelihood, LognormalSurvLikelihood,
       GammaSurvLikelihood, WeibullCureLikelihood,
       ZeroInflatedPoissonLikelihood0, ZeroInflatedPoissonLikelihood1,
       ZeroInflatedPoissonLikelihood2,
       ZeroInflatedBinomialLikelihood0, ZeroInflatedBinomialLikelihood1,
       ZeroInflatedBinomialLikelihood2,
       ZeroInflatedNegativeBinomialLikelihood0,
       ZeroInflatedNegativeBinomialLikelihood1,
       ZeroInflatedNegativeBinomialLikelihood2
export CoxphAugmented, inla_coxph, coxph_design
export Copy, CopyTargetLikelihood
export log_density, ∇_η_log_density, ∇²_η_log_density, ∇³_η_log_density, link
export pointwise_log_density, pointwise_cdf
export add_copy_contributions!

# Censoring (survival likelihoods). Enum values are not exported by
# default to avoid namespace pollution; users can either pass symbols
# (`[:none, :right, :none]`) or qualified names (`Censoring.NONE`,
# or `using LatentGaussianModels: NONE, RIGHT, LEFT, INTERVAL`).
export Censoring

# Hyperpriors
export AbstractHyperPrior
export PCPrecision, GammaPrecision, LogNormalPrecision, WeakPrior
export PCBYM2Phi, LogitBeta, PCAlphaW, PCCor0, GaussianPrior
export log_prior_density, user_scale, prior_name

# Components
export AbstractLatentComponent
export Intercept, FixedEffects, IID, IIDND, IID2D, RW1, RW2, AR1, Seasonal, Besag, BYM, BYM2,
       Leroux, Generic0, Generic1
export AbstractIIDND, IIDND_Sep
export precision_matrix, initial_hyperparameters, nhyperparameters,
       log_hyperprior, prior_mean

# Observation mapping
export AbstractObservationMapping, IdentityMapping, LinearProjector,
       StackedMapping, KroneckerMapping
export apply!, apply_adjoint!, nrows, ncols, likelihood_for, as_matrix

# Model + inference
export LatentGaussianModel, n_latent, n_observations, n_hyperparameters,
       n_likelihoods, n_likelihood_hyperparameters
export joint_precision
export joint_log_density, joint_∇_η_log_density, joint_∇²_η_log_density,
       joint_∇³_η_log_density, joint_pointwise_log_density, joint_pointwise_cdf
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

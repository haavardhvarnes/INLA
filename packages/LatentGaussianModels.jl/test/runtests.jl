using Test
using Random
using SparseArrays
using LinearAlgebra
using LatentGaussianModels
using GMRFs

@testset verbose=true "LatentGaussianModels.jl" begin
    @testset "Links" begin
        include("regression/test_links.jl")
    end
    @testset "Priors" begin
        include("regression/test_priors.jl")
    end
    @testset "Likelihoods" begin
        include("regression/test_likelihoods.jl")
    end
    @testset "Zero-inflated likelihoods" begin
        include("regression/test_zero_inflated.jl")
    end
    @testset "ExponentialLikelihood — Censoring" begin
        include("regression/test_exponential_censoring.jl")
    end
    @testset "WeibullLikelihood — Censoring" begin
        include("regression/test_weibull_censoring.jl")
    end
    @testset "LognormalSurvLikelihood — Censoring" begin
        include("regression/test_lognormal_surv_censoring.jl")
    end
    @testset "GammaSurvLikelihood — Censoring" begin
        include("regression/test_gamma_surv_censoring.jl")
    end
    @testset "WeibullCureLikelihood — Censoring" begin
        include("regression/test_weibull_cure_censoring.jl")
    end
    @testset "Cox PH — augmentation invariants" begin
        include("regression/test_coxph_augmentation.jl")
    end
    @testset "Components" begin
        include("regression/test_components.jl")
    end
    @testset "IIDND — Phase I-A PR-1a/PR-1b" begin
        include("regression/test_iidnd.jl")
    end
    @testset "MEB — Phase I-B PR-2b" begin
        include("regression/test_meb.jl")
    end
    @testset "MEC — Phase I-B PR-2c" begin
        include("regression/test_mec.jl")
    end
    @testset "Replicate — Phase I-C PR-3a" begin
        include("regression/test_replicate.jl")
    end
    @testset "Group — Phase I-C PR-3b" begin
        include("regression/test_group.jl")
    end
    @testset "Multinomial helpers — Phase J PR-7" begin
        include("regression/test_multinomial.jl")
    end
    @testset "BYM" begin
        include("regression/test_bym.jl")
    end
    @testset "BYM2" begin
        include("regression/test_bym2.jl")
    end
    @testset "Leroux" begin
        include("regression/test_leroux.jl")
    end
    @testset "Generic0" begin
        include("regression/test_generic0.jl")
    end
    @testset "Generic1" begin
        include("regression/test_generic1.jl")
    end
    @testset "Generic2" begin
        include("regression/test_generic2.jl")
    end
    @testset "Seasonal" begin
        include("regression/test_seasonal.jl")
    end
    @testset "Laplace — Gaussian identity" begin
        include("regression/test_laplace_gaussian.jl")
    end
    @testset "Laplace — Hard constraint" begin
        include("regression/test_laplace_constrained.jl")
    end
    @testset "Empirical Bayes — Gaussian" begin
        include("regression/test_eb_gaussian.jl")
    end
    @testset "INLA — Gaussian" begin
        include("regression/test_inla_gaussian.jl")
    end
    @testset "INLA — Marginals + Accessors" begin
        include("regression/test_inla_marginals.jl")
    end
    @testset "Copy — fixed β=1.0 oracle + free β recovery" begin
        include("regression/test_copy.jl")
    end
    @testset "INLA — Joint longitudinal + Weibull survival (Copy)" begin
        include("regression/test_inla_joint_baghfalaki.jl")
    end
    @testset "INLA — Poisson + BYM2 (synthetic)" begin
        include("regression/test_inla_poisson_bym2.jl")
    end
    @testset "INLA — Cox PH (synthetic recovery)" begin
        include("regression/test_inla_coxph.jl")
    end
    @testset "INLA — WeibullCure (synthetic, no R-INLA family)" begin
        include("regression/test_inla_weibull_cure.jl")
    end
    @testset "Diagnostics — DIC / WAIC / CPO / PIT" begin
        include("regression/test_diagnostics.jl")
    end
    @testset "refine_hyperposterior — Phase K PR-3" begin
        include("regression/test_refine_hyperposterior.jl")
    end
    @testset "posterior_predictive — Phase K PR-4" begin
        include("regression/test_posterior_predictive.jl")
    end
    @testset "PSIS-LOO — Phase K PR-5" begin
        include("regression/test_psis_loo.jl")
    end
    @testset "posterior_predictive_y — Phase K PR-6" begin
        include("regression/test_posterior_predictive_y.jl")
    end
    @testset "Simplified Laplace — skew correction" begin
        include("regression/test_simplified_laplace.jl")
    end
    @testset "Simplified Laplace — mean-shift correction" begin
        include("regression/test_sla_mean_shift.jl")
    end
    @testset "AbstractMarginalStrategy — Phase L PR-1 dispatch" begin
        include("regression/test_marginal_strategy_dispatch.jl")
    end
    @testset "LogDensityProblems conformance" begin
        include("regression/test_log_density.jl")
    end
    @testset "Integration schemes" begin
        include("regression/test_integration_schemes.jl")
    end
    @testset "Summary layout vs R-INLA" begin
        include("regression/test_summary_layout.jl")
    end
    @testset "Oracle (R-INLA)" begin
        include("oracle/test_scotland_bym2.jl")
        include("oracle/test_scotland_bym.jl")
        include("oracle/test_pennsylvania_bym2.jl")
        include("oracle/test_synthetic_nbinomial.jl")
        include("oracle/test_synthetic_gamma.jl")
        include("oracle/test_synthetic_disconnected_besag.jl")
        include("oracle/test_synthetic_generic0.jl")
        include("oracle/test_synthetic_generic1.jl")
        include("oracle/test_synthetic_seasonal.jl")
        include("oracle/test_synthetic_leroux.jl")
        include("oracle/test_synthetic_exponential_survival.jl")
        include("oracle/test_synthetic_weibull_survival.jl")
        include("oracle/test_synthetic_lognormal_survival.jl")
        include("oracle/test_synthetic_gamma_survival.jl")
        include("oracle/test_synthetic_coxph.jl")
        include("oracle/test_synthetic_joint_gauss_pois.jl")
        include("oracle/test_synthetic_baghfalaki.jl")
        include("oracle/test_synthetic_zip1.jl")
        include("oracle/test_synthetic_iid2d.jl")
        include("oracle/test_synthetic_replicate_ar1.jl")
        include("oracle/test_synthetic_beta.jl")
        include("oracle/test_synthetic_betabinomial.jl")
        include("oracle/test_synthetic_studentt.jl")
        include("oracle/test_synthetic_skewnormal.jl")
        include("oracle/test_synthetic_gev.jl")
        include("oracle/test_synthetic_pom.jl")
        include("oracle/test_synthetic_multinomial.jl")
    end
    @testset "Quality" begin
        include("quality/test_aqua.jl")
        include("quality/test_jet.jl")
    end
end

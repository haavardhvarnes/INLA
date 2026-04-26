using Test
using Random
using SparseArrays
using LinearAlgebra
using LatentGaussianModels
using GMRFs

@testset "LatentGaussianModels.jl" begin
    @testset "Links" begin
        include("regression/test_links.jl")
    end
    @testset "Priors" begin
        include("regression/test_priors.jl")
    end
    @testset "Likelihoods" begin
        include("regression/test_likelihoods.jl")
    end
    @testset "Components" begin
        include("regression/test_components.jl")
    end
    @testset "BYM" begin
        include("regression/test_bym.jl")
    end
    @testset "BYM2" begin
        include("regression/test_bym2.jl")
    end
    @testset "Generic0" begin
        include("regression/test_generic0.jl")
    end
    @testset "Generic1" begin
        include("regression/test_generic1.jl")
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
    @testset "INLA — Poisson + BYM2 (synthetic)" begin
        include("regression/test_inla_poisson_bym2.jl")
    end
    @testset "Diagnostics — DIC / WAIC / CPO / PIT" begin
        include("regression/test_diagnostics.jl")
    end
    @testset "Simplified Laplace — skew correction" begin
        include("regression/test_simplified_laplace.jl")
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
    end
end

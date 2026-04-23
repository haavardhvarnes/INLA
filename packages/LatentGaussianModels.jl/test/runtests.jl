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
    @testset "Laplace — Gaussian identity" begin
        include("regression/test_laplace_gaussian.jl")
    end
    @testset "Empirical Bayes — Gaussian" begin
        include("regression/test_eb_gaussian.jl")
    end
end

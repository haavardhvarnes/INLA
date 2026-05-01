using Test
using GMRFs
using LinearAlgebra
using SparseArrays
using Random
using Statistics

@testset "IIDGMRF precision" begin
    g = IIDGMRF(5; τ=2.5)
    @test num_nodes(g) == 5
    @test rankdef(g) == 0
    Q = precision_matrix(g)
    @test Q == 2.5 * sparse(I, 5, 5)
end

@testset "IIDGMRF logpdf closed form" begin
    rng = MersenneTwister(42)
    τ = 2.0
    n = 5
    g = IIDGMRF(n; τ=τ)
    x = rand(rng, g)
    expected = -0.5 * n * log(2π) + 0.5 * n * log(τ) - 0.5 * τ * sum(abs2, x)
    @test logpdf(g, x)≈expected rtol=1e-12
end

@testset "IIDGMRF sampling variance" begin
    rng = MersenneTwister(123)
    τ = 4.0
    n = 3
    g = IIDGMRF(n; τ=τ)
    N = 20_000
    samples = [rand(rng, g) for _ in 1:N]
    S = reduce(hcat, samples)                # n × N
    # Sample covariance should approximate τ⁻¹ · I.
    C = cov(S; dims=2)
    @test tr(C) / n≈1 / τ rtol=0.05     # 5% MC tolerance
    offdiag = C - Diagonal(C)
    @test maximum(abs, offdiag) < 0.05 * (1 / τ)  # off-diag small
end

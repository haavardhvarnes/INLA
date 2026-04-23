using Test
using GMRFs
using LinearAlgebra
using SparseArrays
using Random
using Statistics: cov, mean

# Monte-Carlo covariance check: draw N samples, compare the sample
# covariance to Q⁻¹ (proper) or the pseudo-inverse Q⁺ (intrinsic).
# Tolerance is loose (5 %) — this is a sanity check for the sampling
# algorithm, not a tight agreement test.

@testset "MC covariance: AR1GMRF" begin
    rng = MersenneTwister(17)
    n = 4
    ρ = 0.5
    τ = 1.0
    g = AR1GMRF(n; ρ = ρ, τ = τ)
    N = 30_000
    S = reduce(hcat, rand(rng, g) for _ in 1:N)
    C = cov(S; dims = 2)
    Σ_expected = inv(Matrix(precision_matrix(g)))
    @test maximum(abs, C - Σ_expected) < 0.05
end

@testset "MC covariance: RW1GMRF (intrinsic, projected)" begin
    rng = MersenneTwister(31)
    n = 5
    τ = 1.0
    g = RW1GMRF(n; τ = τ)
    N = 30_000
    S = reduce(hcat, rand(rng, g) for _ in 1:N)
    @test maximum(abs, mean(S; dims = 2)) < 0.05  # mean ~ 0

    # Expected pseudo-inverse: project onto non-null subspace
    Q = Matrix(precision_matrix(g))
    V = null_space_basis(g)
    Σ_expected = inv(Q + V * V') - V * V'
    C = cov(S; dims = 2)
    @test maximum(abs, C - Σ_expected) < 0.1  # looser MC tolerance for intrinsic
end

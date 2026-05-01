using Test
using GMRFs
using LinearAlgebra
using SparseArrays
using Random
using Statistics

@testset "AR1GMRF precision structure" begin
    n = 6
    ρ = 0.7
    τ = 1.5
    g = AR1GMRF(n; ρ=ρ, τ=τ)
    @test rankdef(g) == 0
    Q = Matrix(precision_matrix(g))
    # Endpoints: τ/(1-ρ²); interior: τ(1+ρ²)/(1-ρ²); off-diag: -τρ/(1-ρ²).
    # These give marginal variance 1/τ.
    scale = τ / (1 - ρ^2)
    for i in 1:n
        if i == 1 || i == n
            @test Q[i, i] ≈ scale
        else
            @test Q[i, i] ≈ scale * (1 + ρ^2)
        end
    end
    for i in 1:(n - 1)
        @test Q[i, i + 1] ≈ -scale * ρ
        @test Q[i + 1, i] ≈ -scale * ρ
    end
end

@testset "AR1GMRF covariance matches Toeplitz(ρ^|i-j|)/τ" begin
    n = 8
    ρ = 0.6
    τ = 2.0
    g = AR1GMRF(n; ρ=ρ, τ=τ)
    Σ_expected = [ρ^abs(i - j) / τ for i in 1:n, j in 1:n]
    Σ_computed = inv(Matrix(precision_matrix(g)))
    @test Σ_computed≈Σ_expected rtol=1e-10
end

@testset "AR1GMRF invalid arguments" begin
    @test_throws ArgumentError AR1GMRF(5; ρ=1.1, τ=1.0)
    @test_throws ArgumentError AR1GMRF(5; ρ=-1.0, τ=1.0)
    @test_throws ArgumentError AR1GMRF(5; ρ=0.5, τ=-1.0)
    @test_throws ArgumentError AR1GMRF(1; ρ=0.5, τ=1.0)
end

@testset "AR1GMRF sampling covariance (MC)" begin
    rng = MersenneTwister(2024)
    n = 4
    ρ = 0.5
    τ = 1.0
    g = AR1GMRF(n; ρ=ρ, τ=τ)
    N = 50_000
    samples = reduce(hcat, rand(rng, g) for _ in 1:N)
    Σ_sample = cov(samples; dims=2)
    Σ_expected = [ρ^abs(i - j) / τ for i in 1:n, j in 1:n]
    @test maximum(abs, Σ_sample - Σ_expected) < 0.05   # MC 5% tolerance
end

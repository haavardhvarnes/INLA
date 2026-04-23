using Test
using GMRFs
using LinearAlgebra
using SparseArrays
using Random
using Statistics: mean

# For each proper GMRF, the logpdf value should agree with the dense
# closed-form Gaussian logpdf using Matrix(Q).

function dense_proper_logpdf(g::AbstractGMRF, x::AbstractVector)
    n = length(x)
    Q = Matrix(precision_matrix(g))
    μ = prior_mean(g)
    xc = x .- μ
    ld = logdet(Q)
    quad = dot(xc, Q * xc)
    return -0.5 * n * log(2π) + 0.5 * ld - 0.5 * quad
end

@testset "logpdf agrees with dense closed-form: IID" begin
    rng = MersenneTwister(1)
    g = IIDGMRF(4; τ = 1.7)
    for _ in 1:5
        x = rand(rng, g)
        @test logpdf(g, x) ≈ dense_proper_logpdf(g, x) rtol = 1e-10
    end
end

@testset "logpdf agrees with dense closed-form: AR1" begin
    rng = MersenneTwister(2)
    g = AR1GMRF(7; ρ = 0.4, τ = 1.2)
    for _ in 1:5
        x = rand(rng, g)
        @test logpdf(g, x) ≈ dense_proper_logpdf(g, x) rtol = 1e-10
    end
end

@testset "logpdf intrinsic: constraint check trips on off-subspace input" begin
    g = RW1GMRF(5; τ = 1.0)
    x = randn(5)                                # generic vector; sum ≠ 0
    @test sum(x) != 0                           # sanity
    @test_throws ArgumentError logpdf(g, x)     # default: check_constraint=true
    # Projected version should succeed
    xp = x .- mean(x)
    @test abs(sum(xp)) < 1e-12
    @test logpdf(g, xp) isa Float64
end

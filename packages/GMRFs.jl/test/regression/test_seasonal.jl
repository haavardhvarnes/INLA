using Test
using GMRFs
using LinearAlgebra
using SparseArrays
using Random

@testset "SeasonalGMRF precision structure" begin
    # Hand-build B and compare to Q = τ B' B at n=7, s=4, τ=2.
    n = 7
    s = 4
    τ = 2.0
    g = SeasonalGMRF(n; period = s, τ = τ)
    @test num_nodes(g) == n
    @test rankdef(g) == s - 1

    B = zeros(n - s + 1, n)
    for t in 1:(n - s + 1), j in t:(t + s - 1)
        B[t, j] = 1
    end
    Q_ref = τ .* (B' * B)

    Q = Matrix(precision_matrix(g))
    @test Q ≈ Q_ref
    @test issymmetric(Q)
end

@testset "SeasonalGMRF null space = period-s zero-sum repeats" begin
    # Any periodic-s sequence summing to zero within one period is in
    # the null space.
    n = 12
    s = 4
    g = SeasonalGMRF(n; period = s)
    Q = Matrix(precision_matrix(g))

    # Basis: ε_k = e_k - e_s (one period), repeated.
    for k in 1:(s - 1)
        v = zeros(n)
        for i in 1:n
            r = mod1(i, s)
            if r == k
                v[i] = 1.0
            elseif r == s
                v[i] = -1.0
            end
        end
        @test maximum(abs, Q * v) < 1.0e-12
    end

    # And the full null space is exactly (s-1)-dimensional.
    @test sum(abs.(eigvals(Symmetric(Q))) .< 1.0e-10) == s - 1
end

@testset "SeasonalGMRF default constraint matches null space" begin
    n = 10
    s = 3
    g = SeasonalGMRF(n; period = s)
    kc = GMRFs.constraints(g)
    @test kc isa LinearConstraint
    C = GMRFs.constraint_matrix(kc)
    @test size(C) == (s - 1, n)
    @test GMRFs.constraint_rhs(kc) == zeros(s - 1)

    # range(C^T) must equal null(Q), i.e. C has full row rank and its
    # row span coincides with the null-space basis.
    Q = Matrix(precision_matrix(g))
    null_basis = zeros(n, s - 1)
    for k in 1:(s - 1)
        for i in 1:n
            r = mod1(i, s)
            if r == k
                null_basis[i, k] = 1.0
            elseif r == s
                null_basis[i, k] = -1.0
            end
        end
    end
    # C * null_basis is (s-1) × (s-1) and must be invertible (i.e. C
    # separates every null direction).
    M = C * null_basis
    @test size(M) == (s - 1, s - 1)
    @test abs(det(M)) > 1.0e-10

    # And Q on the range(C') should be zero (constraint rows are in the
    # null space of Q).
    @test maximum(abs, Q * C') < 1.0e-10
end

@testset "SeasonalGMRF rejects n ≤ period" begin
    @test_throws ArgumentError SeasonalGMRF(4; period = 4)
    @test_throws ArgumentError SeasonalGMRF(10; period = 1)
end

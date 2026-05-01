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
    g = SeasonalGMRF(n; period=s, τ=τ)
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
    g = SeasonalGMRF(n; period=s)
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

@testset "SeasonalGMRF default constraint = single sum-to-zero" begin
    # R-INLA convention: one sum-to-zero row, leaving the remaining
    # (s-2) directions of the period-s zero-sum null space identified
    # by the data. See plans note in inference/constraints.jl for why
    # the Laplace pipeline supports null(Q) ⊋ range(C^T) here.
    #
    # Note: the all-ones vector is *not* in `null(Q)` — its rolling
    # period sums equal `s ≠ 0` — so `Q * C' ≠ 0`. The constraint
    # therefore consumes one PD direction of `Q` (not a null
    # direction); this is precisely why `Seasonal.log_normalizing_constant`
    # must use `(n - period)` rather than `(n - (period - 1))` for the
    # τ-coefficient.
    n = 10
    s = 3
    g = SeasonalGMRF(n; period=s)
    kc = GMRFs.constraints(g)
    @test kc isa LinearConstraint
    C = GMRFs.constraint_matrix(kc)
    @test size(C) == (1, n)
    @test GMRFs.constraint_rhs(kc) == zeros(1)
    @test C ≈ ones(1, n)

    # `range(C^T) ⊂ range(Q)` (rolling sums of `1` are non-zero), so
    # `Q * C^T` is *not* the zero vector — distinguishing this from the
    # F_GENERIC0/F_BYM2 case where `range(C^T) ⊂ null(Q)`.
    Q = Matrix(precision_matrix(g))
    @test maximum(abs, Q * vec(C')) > 1.0e-3
end

@testset "SeasonalGMRF null_space_basis" begin
    n = 12
    s = 4
    g = SeasonalGMRF(n; period=s)
    V = GMRFs.null_space_basis(g)
    @test size(V) == (n, s - 1)
    @test V' * V≈I(s - 1) atol=1.0e-12
    Q = Matrix(precision_matrix(g))
    @test maximum(abs, Q * V) < 1.0e-10
end

@testset "SeasonalGMRF rejects n ≤ period or period < 2" begin
    @test_throws ArgumentError SeasonalGMRF(4; period=4)
    @test_throws ArgumentError SeasonalGMRF(10; period=1)
end

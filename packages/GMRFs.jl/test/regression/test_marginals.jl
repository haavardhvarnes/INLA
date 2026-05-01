using Test
using GMRFs
using Graphs
using LinearAlgebra
using SparseArrays
using Statistics: mean

@testset "marginal_variances: IID" begin
    g = IIDGMRF(5; τ=2.0)
    v = marginal_variances(g)
    @test v≈fill(0.5, 5) rtol=1e-12
end

@testset "marginal_variances: AR1" begin
    n = 6
    ρ = 0.4
    τ = 1.0
    g = AR1GMRF(n; ρ=ρ, τ=τ)
    v = marginal_variances(g)
    # Stationary AR1 has constant marginal variance 1/τ
    @test v≈fill(1 / τ, n) rtol=1e-10
end

@testset "marginal_variances: RW1 (intrinsic)" begin
    n = 5
    g = RW1GMRF(n; τ=1.0)
    v = marginal_variances(g)
    # Must match the generalised-inverse diagonal on non-null subspace.
    Q = Matrix(precision_matrix(g))
    V = null_space_basis(g)
    Σ = inv(Q + V * V') - V * V'
    @test v≈diag(Σ) rtol=1e-10
    @test all(isfinite, v)
    @test all(v .> 0)
end

@testset "marginal_variances: Besag scaled geomean ≈ 1" begin
    # Sanity check of scaling: geomean of marginal variances ≈ 1 when
    # scale_model=true, τ=1.
    g = GMRFGraph(grid([3, 3]))
    b = BesagGMRF(g; τ=1.0, scale_model=true)
    v = marginal_variances(b)
    @test exp(mean(log.(v)))≈1.0 rtol=1e-8
end

@testset "marginal_variances: selinv scales past the old dense guard" begin
    # Post ADR-012: default :selinv path handles large n. On IIDGMRF, Q = τI,
    # so diag(Q⁻¹) = 1/τ constant regardless of dimension.
    g = IIDGMRF(1500; τ=2.5)
    v = marginal_variances(g)
    @test length(v) == 1500
    @test v≈fill(1 / 2.5, 1500) rtol=1e-10
end

@testset "marginal_variances: :selinv vs :dense agree on proper GMRF" begin
    n = 8
    ρ = 0.3
    τ = 1.5
    g = AR1GMRF(n; ρ=ρ, τ=τ)
    v_selinv = marginal_variances(g; method=:selinv)
    v_dense = marginal_variances(g; method=:dense)
    @test v_selinv≈v_dense rtol=1e-10
end

@testset "marginal_variances: :selinv errors on intrinsic GMRF" begin
    g = RW1GMRF(5; τ=1.0)
    @test_throws ArgumentError marginal_variances(g; method=:selinv)
end

@testset "marginal_variances(Q::AbstractSparseMatrix)" begin
    # Proper sparse PD matrix — the Laplace-posterior shape.
    n = 6
    Q = sparse(2.0I, n, n) + spdiagm(-1 => -0.3 * ones(n - 1),
        1 => -0.3 * ones(n - 1))
    v_selinv = marginal_variances(Q)
    v_dense = marginal_variances(Q; method=:dense)
    @test v_selinv≈v_dense rtol=1e-10
end

# Precision for α = 1: `Q = τ² (κ² C + G₁)` — Matérn smoothness ν = 0 in
# 2D. Tests cover structural properties (symmetry, positive-definiteness,
# sparsity) and the analytic (τ, κ) scaling law.

using LinearAlgebra
using SparseArrays

@testset "spde_precision(α=1) — closed-form on 2-triangle square" begin
    fem = FEMMatrices(POINTS_SQ, TRIS_SQ)
    τ, κ = 1.3, 2.5
    Q = spde_precision(fem, 1, τ, κ)

    Q_ref = τ^2 * (κ^2 * Matrix(fem.C) + Matrix(fem.G1))
    @test Matrix(Q) ≈ Q_ref rtol = 1.0e-12
    @test issymmetric(Q)

    # Positive-definite (α=1 has no nullspace — κ² C + G₁ is SPD).
    @test isposdef(Symmetric(Matrix(Q)))
end

@testset "spde_precision(α=1) — τ and κ scaling" begin
    fem = FEMMatrices(POINTS_SQ, TRIS_SQ)
    τ0, κ0 = 1.0, 1.0
    Q0 = spde_precision(fem, 1, τ0, κ0)

    # Scaling τ → c·τ multiplies Q by c² (at fixed κ).
    for c in (0.5, 2.0, 3.7)
        Qc = spde_precision(fem, 1, c * τ0, κ0)
        @test Matrix(Qc) ≈ c^2 * Matrix(Q0) rtol = 1.0e-12
    end

    # Scaling κ at fixed τ: τ²(κ² C + G₁) — non-linear in κ, check the
    # identity exactly.
    κ1 = 3.0
    Q1 = spde_precision(fem, 1, τ0, κ1)
    @test Matrix(Q1) ≈ τ0^2 * (κ1^2 * Matrix(fem.C) + Matrix(fem.G1)) rtol = 1.0e-12
end

@testset "spde_precision(α=1) — argument validation" begin
    fem = FEMMatrices(POINTS_SQ, TRIS_SQ)
    @test_throws ArgumentError spde_precision(fem, 1, -1.0, 1.0)
    @test_throws ArgumentError spde_precision(fem, 1, 1.0, -0.1)
    @test_throws ArgumentError spde_precision(fem, 0, 1.0, 1.0)
    @test_throws ArgumentError spde_precision(fem, 3, 1.0, 1.0)
end

@testset "spde_precision(α=1) — stateless form agrees" begin
    fem = FEMMatrices(POINTS_SQ, TRIS_SQ)
    τ, κ = 0.7, 1.9
    Q_struct = spde_precision(fem, 1, τ, κ)
    Q_stateless = spde_precision(1, τ, κ, fem.C, fem.G1)
    @test Matrix(Q_struct) ≈ Matrix(Q_stateless) rtol = 1.0e-12
end

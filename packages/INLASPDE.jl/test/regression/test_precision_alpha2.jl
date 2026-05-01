# Precision for α = 2: `Q = τ² (κ⁴ C̃ + 2κ² G₁ + G₂)` with
# `G₂ = G₁ · C̃⁻¹ · G₁`. This is the most common SPDE-Matérn case in
# practice (ν = 1 in 2D).

using LinearAlgebra
using SparseArrays

@testset "stiffness_squared — matches G₁ C̃⁻¹ G₁ densely" begin
    fem = FEMMatrices(POINTS_SQ, TRIS_SQ)
    G2_dense = Matrix(fem.G1) * inv(Matrix(fem.C_lumped)) * Matrix(fem.G1)
    @test Matrix(fem.G2)≈G2_dense rtol=1.0e-12
    @test issymmetric(fem.G2)

    # G₂ is PSD with constants in its null space (since G₁·1 = 0).
    @test maximum(abs, fem.G2 * ones(size(fem.G2, 1))) < 1.0e-12
end

@testset "spde_precision(α=2) — closed-form on 2-triangle square" begin
    fem = FEMMatrices(POINTS_SQ, TRIS_SQ)
    τ, κ = 1.1, 0.8
    Q = spde_precision(fem, 2, τ, κ)
    Q_ref = τ^2 * (
        κ^4 * Matrix(fem.C_lumped) +
        2 * κ^2 * Matrix(fem.G1) +
        Matrix(fem.G2)
    )
    @test Matrix(Q)≈Q_ref rtol=1.0e-12
    @test issymmetric(Q)
    @test isposdef(Symmetric(Matrix(Q)))
end

@testset "spde_precision(α=2) — τ scaling" begin
    fem = FEMMatrices(POINTS_SQ, TRIS_SQ)
    Q0 = spde_precision(fem, 2, 1.0, 1.5)
    for c in (0.25, 2.5)
        Qc = spde_precision(fem, 2, c, 1.5)
        @test Matrix(Qc)≈c^2 * Matrix(Q0) rtol=1.0e-12
    end
end

@testset "spde_precision(α=2) — sparsity pattern matches G₂" begin
    # With lumped C̃ the non-zero pattern of Q(α=2) is exactly the union
    # of patterns of G₁ and G₂. G₂'s pattern is the sparsity of `G₁ G₁`
    # (since C̃ is diagonal), i.e. vertices within two mesh hops.
    fem = FEMMatrices(POINTS_SQ, TRIS_SQ)
    Q = spde_precision(fem, 2, 1.0, 1.0)

    expected_pattern = (Matrix(fem.G2) .!= 0) .| (Matrix(fem.G1) .!= 0)
    @test all((Q .!= 0) .== expected_pattern)
end

@testset "spde_precision(α=2) — stateless form agrees" begin
    fem = FEMMatrices(POINTS_SQ, TRIS_SQ)
    τ, κ = 0.9, 1.4
    Q_struct = spde_precision(fem, 2, τ, κ)
    Q_stateless = spde_precision(2, τ, κ, fem.C, fem.G1)
    @test Matrix(Q_struct)≈Matrix(Q_stateless) rtol=1.0e-12
end

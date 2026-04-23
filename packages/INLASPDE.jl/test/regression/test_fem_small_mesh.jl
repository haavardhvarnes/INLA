# Hand-computed reference for the unit square split into two triangles.
#
# Vertices (1-based):
#   p1 = (0, 0)   p2 = (1, 0)   p3 = (1, 1)   p4 = (0, 1)
# Triangles: T₁ = (1, 2, 3), T₂ = (1, 3, 4). Each has area 1/2.
#
# For P1 FEM on a triangle of area A:
#   C^e[i,i] = A/6,  C^e[i,j] = A/12  (i ≠ j)
#   G^e[i,j] = (b_i b_j + c_i c_j) / (4A)
# where (b_k, c_k) are the rotated edge vectors opposite vertex k.
#
# Assembling globally yields closed-form entries — the values below were
# worked out by hand. They are the reference the implementation must hit
# to machine precision.

using SparseArrays

const POINTS_SQ = Float64[
    0.0 0.0
    1.0 0.0
    1.0 1.0
    0.0 1.0
]
const TRIS_SQ = [
    1 2 3
    1 3 4
]

@testset "assemble_fem_matrices — 2-triangle unit square" begin
    C, G1 = assemble_fem_matrices(POINTS_SQ, TRIS_SQ)

    @test size(C) == (4, 4)
    @test size(G1) == (4, 4)
    @test issymmetric(C)
    @test issymmetric(G1)

    # --- mass matrix reference ---
    C_ref = zeros(4, 4)
    C_ref[1, 1] = 1 / 6   # vertex 1 is in both triangles
    C_ref[2, 2] = 1 / 12
    C_ref[3, 3] = 1 / 6   # vertex 3 is in both triangles
    C_ref[4, 4] = 1 / 12
    C_ref[1, 2] = C_ref[2, 1] = 1 / 24
    C_ref[2, 3] = C_ref[3, 2] = 1 / 24
    C_ref[1, 3] = C_ref[3, 1] = 1 / 12    # shared edge, both triangles contribute
    C_ref[3, 4] = C_ref[4, 3] = 1 / 24
    C_ref[1, 4] = C_ref[4, 1] = 1 / 24
    # (2,4) are not adjacent (diagonal of the square on the other side)
    @test Matrix(C) ≈ C_ref rtol = 1.0e-12

    # Conservation: total integral of constant 1 over the unit square.
    @test sum(C) ≈ 1.0 rtol = 1.0e-12

    # --- stiffness matrix reference ---
    G1_ref = zeros(4, 4)
    G1_ref[1, 1] = 1.0
    G1_ref[2, 2] = 1.0
    G1_ref[3, 3] = 1.0
    G1_ref[4, 4] = 1.0
    G1_ref[1, 2] = G1_ref[2, 1] = -0.5
    G1_ref[2, 3] = G1_ref[3, 2] = -0.5
    G1_ref[3, 4] = G1_ref[4, 3] = -0.5
    G1_ref[1, 4] = G1_ref[4, 1] = -0.5
    # Off-diagonal (1,3) and (2,4): the two triangles contribute equal and
    # opposite values that would cancel at (1,3) — but actually (1,3) share
    # one triangle in each (T₁ contributes, T₂ contributes), summing to 0.
    @test Matrix(G1) ≈ G1_ref rtol = 1.0e-12

    # Constant-function in the kernel of G₁ (rows sum to zero).
    @test maximum(abs, G1 * ones(4)) < 1.0e-12
end

@testset "assemble_fem_matrices — orientation invariance" begin
    # Reversing triangle orientation (CW vs CCW) must not change the
    # assembled matrices, since assembly uses the unsigned area and
    # gradients are intrinsic to the shape.
    tris_reversed = similar(TRIS_SQ)
    tris_reversed[1, :] = [1, 3, 2]      # T₁ reversed
    tris_reversed[2, :] = [1, 4, 3]      # T₂ reversed

    C_a, G_a = assemble_fem_matrices(POINTS_SQ, TRIS_SQ)
    C_b, G_b = assemble_fem_matrices(POINTS_SQ, tris_reversed)

    @test Matrix(C_a) ≈ Matrix(C_b) rtol = 1.0e-12
    @test Matrix(G_a) ≈ Matrix(G_b) rtol = 1.0e-12
end

@testset "assemble_fem_matrices — argument validation" begin
    bad_points = zeros(4, 3)          # 3D points rejected in M1
    @test_throws ArgumentError assemble_fem_matrices(bad_points, TRIS_SQ)

    bad_tris = [1 2 3 4; 1 3 4 2]     # quads rejected
    @test_throws ArgumentError assemble_fem_matrices(POINTS_SQ, bad_tris)

    out_of_range = [1 2 99]
    @test_throws ArgumentError assemble_fem_matrices(POINTS_SQ, out_of_range)

    degenerate_pts = Float64[0 0; 1 0; 2 0; 0 1]  # collinear first three
    degenerate_tris = [1 2 3]
    @test_throws ArgumentError assemble_fem_matrices(degenerate_pts, degenerate_tris)
end

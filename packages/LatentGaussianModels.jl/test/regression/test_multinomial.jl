# Regression tests for the multinomial-to-Poisson reformulation
# helpers introduced in Phase J PR-7 (ADR-024).
#
# Pinned invariants:
#   - `multinomial_to_poisson(Y)` lays out long-format y/row_id/class_id
#     in row-major order so that y[(i-1)*K + k] == Y[i, k].
#   - `multinomial_design_matrix(helper, X)` returns a sparse
#     (n_long, (K-1)*p) matrix whose rows mirror that layout: for the
#     reference class, the row is all zeros; for the other classes, the
#     `((k′-1)*p+1):(k′*p)` columns carry x_i^⊤.
#   - With a `prec=list(initial=-10, fixed=TRUE)` IID(n) nuisance
#     intercept, the reformulation reproduces R-INLA's
#     Multinomial-INLA recipe verbatim.

using Test
using SparseArrays
using LatentGaussianModels: multinomial_to_poisson, multinomial_design_matrix

@testset "multinomial_to_poisson — round-trip layout" begin
    Y = [1 2 3; 4 5 6; 7 8 9; 10 11 12]
    h = multinomial_to_poisson(Y)
    @test h.n_rows == 4
    @test h.K == 3
    @test h.n_long == 12
    @test length(h.y) == 12
    for i in 1:4, k in 1:3
        idx = (i - 1) * 3 + k
        @test h.y[idx] == Y[i, k]
        @test h.row_id[idx] == i
        @test h.class_id[idx] == k
    end
    @test h.class_names == [1, 2, 3]
end

@testset "multinomial_to_poisson — class_names override" begin
    Y = [3 7; 1 4]
    h = multinomial_to_poisson(Y; class_names=["A", "B"])
    @test h.K == 2
    @test h.class_names == ["A", "B"]
end

@testset "multinomial_to_poisson — argument validation" begin
    @test_throws ArgumentError multinomial_to_poisson(zeros(Int, 0, 3))
    @test_throws ArgumentError multinomial_to_poisson(zeros(Int, 3, 1))
    @test_throws DimensionMismatch multinomial_to_poisson([1 2; 3 4];
        class_names=["A", "B", "C"])
end

@testset "multinomial_design_matrix — corner-point layout (K=3, p=2)" begin
    # Three rows, three classes; reference class = K = 3.
    Y = [2 5 1;
         0 3 4;
         1 1 6]
    X = [1.0 0.5;
         2.0 -0.5;
         -1.0 1.0]
    h = multinomial_to_poisson(Y)
    A = multinomial_design_matrix(h, X)
    @test size(A) == (h.n_long, (h.K - 1) * size(X, 2))
    @test size(A) == (9, 4)
    Ad = Matrix(A)

    # row (i=1, k=1): block 1 carries x_1 = (1.0, 0.5)
    @test Ad[1, 1] == 1.0
    @test Ad[1, 2] == 0.5
    @test Ad[1, 3] == 0.0
    @test Ad[1, 4] == 0.0

    # row (i=1, k=2): block 2 carries x_1
    @test Ad[2, 3] == 1.0
    @test Ad[2, 4] == 0.5
    @test Ad[2, 1] == 0.0
    @test Ad[2, 2] == 0.0

    # row (i=1, k=3) reference class: all zeros
    @test all(==(0.0), Ad[3, :])

    # row (i=3, k=2): block 2 carries x_3 = (-1.0, 1.0)
    @test Ad[8, 3] == -1.0
    @test Ad[8, 4] == 1.0
end

@testset "multinomial_design_matrix — non-default reference class" begin
    Y = [1 1 1; 2 2 2]
    X = reshape([1.0, 2.0], 2, 1)
    h = multinomial_to_poisson(Y)
    # Reference = class 1: only blocks for k = 2 and k = 3 carry x.
    A = Matrix(multinomial_design_matrix(h, X; reference_class=1))
    @test size(A) == (6, 2)
    # row (i=1, k=1) = reference -> all zeros
    @test all(==(0.0), A[1, :])
    # row (i=1, k=2): block 1 (k′=1) carries x_1 = 1.0
    @test A[2, 1] == 1.0
    @test A[2, 2] == 0.0
    # row (i=1, k=3): block 2 (k′=2) carries x_1
    @test A[3, 1] == 0.0
    @test A[3, 2] == 1.0
    # row (i=2, k=2): block 1 carries x_2 = 2.0
    @test A[5, 1] == 2.0
    # row (i=2, k=3): block 2 carries x_2
    @test A[6, 2] == 2.0
end

@testset "multinomial_design_matrix — argument validation" begin
    h = multinomial_to_poisson([1 2; 3 4])
    @test_throws DimensionMismatch multinomial_design_matrix(h, ones(3, 2))
    @test_throws ArgumentError multinomial_design_matrix(h, ones(2, 1);
        reference_class=0)
    @test_throws ArgumentError multinomial_design_matrix(h, ones(2, 1);
        reference_class=3)
end

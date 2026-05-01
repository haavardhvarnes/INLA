using LatentGaussianModels: Generic0, PCPrecision, GammaPrecision,
                            precision_matrix, log_hyperprior, nhyperparameters,
                            initial_hyperparameters, log_normalizing_constant
using GMRFs: Generic0GMRF, num_nodes, GMRFGraph, laplacian_matrix,
             NoConstraint, LinearConstraint, constraints, constraint_matrix,
             constraint_rhs

@testset "Generic0 — proper structure (full rank)" begin
    # 4×4 SPD: tridiagonal with 2 on diag, -0.5 on off-diag.
    R = sparse([1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
        [1, 2, 1, 2, 3, 2, 3, 4, 3, 4],
        [2.0, -0.5, -0.5, 2.0, -0.5, -0.5, 2.0, -0.5, -0.5, 2.0])
    c = Generic0(R; rankdef=0)
    @test length(c) == 4
    @test nhyperparameters(c) == 1
    @test initial_hyperparameters(c) == [0.0]

    Q = precision_matrix(c, [log(3.0)])
    @test Matrix(Q) ≈ 3.0 * Matrix(R)

    @test isfinite(log_hyperprior(c, [0.0]))
    @test constraints(c) isa NoConstraint

    # log NC: full rank → -½ n log(2π) + ½ n log τ (R-INLA F_GENERIC0).
    n = 4
    @test log_normalizing_constant(c, [0.5]) ≈ -0.5 * n * log(2π) + 0.5 * n * 0.5
end

@testset "Generic0 — rank-deficient with constraint" begin
    # Use the RW1 Laplacian on n=5 (rank n-1, sum-to-zero null space).
    n = 5
    R = sparse([1.0 -1.0 0.0 0.0 0.0;
                -1.0 2.0 -1.0 0.0 0.0;
                0.0 -1.0 2.0 -1.0 0.0;
                0.0 0.0 -1.0 2.0 -1.0;
                0.0 0.0 0.0 -1.0 1.0])
    Aeq = ones(1, n)
    e = zeros(1)
    c = Generic0(R; rankdef=1, constraint=LinearConstraint(Aeq, e))
    @test length(c) == n

    Q = precision_matrix(c, [log(2.0)])
    @test Matrix(Q) ≈ 2.0 * Matrix(R)

    con = constraints(c)
    @test con isa LinearConstraint
    @test constraint_matrix(con) ≈ Aeq
    @test constraint_rhs(con) ≈ e

    # log NC: -½ (n - rd) log(2π) + ½ (n - rd) log τ (R-INLA F_GENERIC0).
    @test log_normalizing_constant(c, [0.7]) ≈
          -0.5 * (n - 1) * log(2π) + 0.5 * (n - 1) * 0.7
end

@testset "Generic0 — scale_model = true rescales R" begin
    n = 5
    R = sparse([1.0 -1.0 0.0 0.0 0.0;
                -1.0 2.0 -1.0 0.0 0.0;
                0.0 -1.0 2.0 -1.0 0.0;
                0.0 0.0 -1.0 2.0 -1.0;
                0.0 0.0 0.0 -1.0 1.0])
    c_un = Generic0(R; rankdef=1, scale_model=false)
    c_sc = Generic0(R; rankdef=1, scale_model=true)
    Q_un = Matrix(precision_matrix(c_un, [0.0]))
    Q_sc = Matrix(precision_matrix(c_sc, [0.0]))
    # The scaled version should differ by a positive scalar factor.
    @test Q_un ≈ Matrix(R)
    ratio = Q_sc[1, 1] / Q_un[1, 1]
    @test ratio > 0
    @test Q_sc ≈ ratio .* Q_un
end

@testset "Generic0 — invalid input rejection" begin
    Rsq_nsym = sparse([1.0 0.0; 1.0 1.0])
    @test_throws ArgumentError Generic0(Rsq_nsym)

    Rrec = sparse([1.0 0.0 0.0; 0.0 1.0 0.0])
    @test_throws DimensionMismatch Generic0(Rrec)

    R = sparse([1.0 0.0; 0.0 1.0])
    @test_throws ArgumentError Generic0(R; rankdef=-1)
end

@testset "Generic0 — custom hyperprior" begin
    R = sparse([2.0 -0.5; -0.5 2.0])
    c = Generic0(R; hyperprior=GammaPrecision(1.0, 5.0e-5))
    θ = [0.3]
    expected = log_prior_density(GammaPrecision(1.0, 5.0e-5), θ[1])
    @test log_hyperprior(c, θ) ≈ expected
end

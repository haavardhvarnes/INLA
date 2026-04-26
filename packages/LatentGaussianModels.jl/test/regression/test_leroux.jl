using LatentGaussianModels: Leroux, LogitBeta, PCPrecision,
    precision_matrix, log_hyperprior, log_prior_density,
    nhyperparameters, initial_hyperparameters
using GMRFs: GMRFs, GMRFGraph, num_nodes, NoConstraint, constraints
using LinearAlgebra: I, Symmetric, eigvals
using SparseArrays
using Distributions: Distributions

@testset "Leroux — basic precision structure" begin
    # 4-node path graph.
    W = [0 1 0 0;
         1 0 1 0;
         0 1 0 1;
         0 0 1 0]
    g = GMRFGraph(W)
    n = num_nodes(g)
    R_ref = Matrix(GMRFs.laplacian_matrix(g))

    c = Leroux(g)
    @test length(c) == n
    @test nhyperparameters(c) == 2
    @test initial_hyperparameters(c) == [0.0, 0.0]

    # At (log τ, logit ρ) = (log 2, 0) ⟹ τ = 2, ρ = 0.5:
    #   Q = 2 · (0.5·I + 0.5·R) = I + R.
    Q = Matrix(precision_matrix(c, [log(2.0), 0.0]))
    @test Q ≈ I(n) + R_ref

    # ρ = 0.8 explicit value via logit.
    θ_ρ08 = [0.0, log(0.8 / 0.2)]
    Q08 = Matrix(precision_matrix(c, θ_ρ08))
    @test Q08 ≈ 0.2 .* Matrix(I, n, n) .+ 0.8 .* R_ref

    # ρ → 0 (logit → -∞): Q → τ I_n. Check approximately at ρ ≈ 10^-4.
    θ_iid = [log(4.0), -10.0]
    ρ_iid = inv(1 + exp(10.0))
    Q_iid = Matrix(precision_matrix(c, θ_iid))
    @test Q_iid ≈ 4.0 * ((1 - ρ_iid) .* Matrix(I, n, n) .+ ρ_iid .* R_ref)
end

@testset "Leroux — positive definite for ρ ∈ (0, 1)" begin
    W = [0 1 0 0;
         1 0 1 0;
         0 1 0 1;
         0 0 1 0]
    g = GMRFGraph(W)
    c = Leroux(g)

    for logit_ρ in (-3.0, -1.0, 0.0, 1.0, 3.0)
        Q = Matrix(precision_matrix(c, [0.0, logit_ρ]))
        λs = eigvals(Symmetric(Q))
        @test all(>(0), λs)
    end
end

@testset "Leroux — no default constraint" begin
    W = [0 1 0 0;
         1 0 1 0;
         0 1 0 1;
         0 0 1 0]
    c = Leroux(GMRFGraph(W))
    # Proper for ρ strictly < 1, so no sum-to-zero needed.
    @test constraints(c) isa NoConstraint
end

@testset "Leroux — adjacency-matrix constructor" begin
    W = sparse([0.0 1.0 0.0;
                1.0 0.0 1.0;
                0.0 1.0 0.0])
    c = Leroux(W)
    @test length(c) == 3
    @test nhyperparameters(c) == 2
end

@testset "Leroux — log_hyperprior factorises" begin
    W = [0 1; 1 0]
    c = Leroux(GMRFGraph(W);
               hyperprior_tau = PCPrecision(),
               hyperprior_rho = LogitBeta(1.0, 1.0))
    θ = [0.3, -0.5]
    expected = log_prior_density(PCPrecision(), θ[1]) +
               log_prior_density(LogitBeta(1.0, 1.0), θ[2])
    @test log_hyperprior(c, θ) ≈ expected
end

@testset "LogitBeta prior — Beta(1, 1) is uniform on ρ" begin
    p = LogitBeta(1.0, 1.0)
    for θ in (-2.0, -0.5, 0.0, 0.7, 3.0)
        ρ = inv(1 + exp(-θ))
        @test log_prior_density(p, θ) ≈ log(ρ) + log1p(-ρ)
    end
end

@testset "LogitBeta prior — Beta(2, 5) hand-computed" begin
    p = LogitBeta(2.0, 5.0)
    θ = 0.3
    ρ = inv(1 + exp(-θ))
    log_B = Distributions.loggamma(2.0) + Distributions.loggamma(5.0) -
            Distributions.loggamma(7.0)
    @test log_prior_density(p, θ) ≈
          2.0 * log(ρ) + 5.0 * log1p(-ρ) - log_B
end

@testset "LogitBeta — invalid input rejection" begin
    @test_throws ArgumentError LogitBeta(0.0, 1.0)
    @test_throws ArgumentError LogitBeta(1.0, -0.5)
end

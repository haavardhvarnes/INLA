using LatentGaussianModels: Intercept, FixedEffects, IID, RW1, RW2, AR1, Besag,
    BYM, Leroux, Generic0, Generic1,
    PCPrecision, LogitBeta,
    precision_matrix, log_hyperprior, nhyperparameters,
    initial_hyperparameters
using GMRFs: GMRFs, GMRFGraph, RW1GMRF, RW2GMRF, IIDGMRF, AR1GMRF, BesagGMRF,
    Generic0GMRF, num_nodes, LinearConstraint, constraint_matrix,
    constraint_rhs, nconnected_components
using Distributions: Distributions

@testset "Intercept" begin
    c = Intercept()
    @test length(c) == 1
    @test nhyperparameters(c) == 0
    Q = precision_matrix(c, Float64[])
    @test size(Q) == (1, 1)
    @test Q[1, 1] ≈ 1.0e-3
    @test log_hyperprior(c, Float64[]) == 0.0
end

@testset "FixedEffects" begin
    c = FixedEffects(3)
    @test length(c) == 3
    @test nhyperparameters(c) == 0
    Q = precision_matrix(c, Float64[])
    @test size(Q) == (3, 3)
    @test Matrix(Q) ≈ 1.0e-3 * I(3)
end

@testset "IID component" begin
    c = IID(5)
    @test length(c) == 5
    @test nhyperparameters(c) == 1
    @test initial_hyperparameters(c) == [0.0]
    Q = precision_matrix(c, [log(4.0)])
    @test Matrix(Q) ≈ 4.0 * I(5)
    @test isfinite(log_hyperprior(c, [0.0]))
end

@testset "RW1 component" begin
    c = RW1(6)
    @test length(c) == 6
    @test nhyperparameters(c) == 1
    Q = precision_matrix(c, [log(2.0)])
    # Compare against direct RW1GMRF
    Qref = GMRFs.precision_matrix(RW1GMRF(6; τ = 2.0))
    @test Matrix(Q) ≈ Matrix(Qref)
end

@testset "RW2 component" begin
    c = RW2(7)
    Q = precision_matrix(c, [0.0])
    Qref = GMRFs.precision_matrix(RW2GMRF(7; τ = 1.0))
    @test Matrix(Q) ≈ Matrix(Qref)
end

@testset "AR1 component" begin
    c = AR1(4)
    @test nhyperparameters(c) == 2
    @test initial_hyperparameters(c) == [0.0, 0.0]
    θ = [log(2.0), atanh(0.3)]
    Q = precision_matrix(c, θ)
    Qref = GMRFs.precision_matrix(AR1GMRF(4; ρ = 0.3, τ = 2.0))
    @test Matrix(Q) ≈ Matrix(Qref)
end

@testset "Besag component" begin
    W = [0 1 0 0;
         1 0 1 0;
         0 1 0 1;
         0 0 1 0]
    g = GMRFGraph(W)
    c = Besag(g; scale_model = false)
    @test length(c) == 4
    Q = precision_matrix(c, [log(3.0)])
    Qref = GMRFs.precision_matrix(BesagGMRF(g; τ = 3.0, scale_model = false))
    @test Matrix(Q) ≈ Matrix(Qref)
end

@testset "Generic0 component" begin
    # Symmetric positive-definite structure matrix.
    n = 6
    rng = Random.Xoshiro(4242)
    M = randn(rng, n, n)
    R = sparse(Symmetric(M'M + n * I(n)))

    c = Generic0(R)
    @test length(c) == n
    @test nhyperparameters(c) == 1
    @test initial_hyperparameters(c) == [0.0]

    Q = precision_matrix(c, [log(2.5)])
    @test Matrix(Q) ≈ 2.5 .* Matrix(R)

    # Agreement with direct Generic0GMRF.
    Qref = GMRFs.precision_matrix(Generic0GMRF(R; τ = 2.5))
    @test Matrix(Q) ≈ Matrix(Qref)

    @test isfinite(log_hyperprior(c, [0.0]))
end

@testset "Generic0 component — scale_model=true" begin
    # Use a Besag (Laplacian) structure matrix so scale_model has
    # something to actually rescale.
    W = [0 1 0 0;
         1 0 1 0;
         0 1 0 1;
         0 0 1 0]
    L = sparse(Diagonal(vec(sum(W; dims = 2))) - W)   # graph Laplacian; rankdef = 1
    c = Generic0(L; rankdef = 1, scale_model = true)
    Q = precision_matrix(c, [0.0])

    # Compare against a direct Generic0GMRF with the same options.
    Qref = GMRFs.precision_matrix(Generic0GMRF(L; τ = 1.0, rankdef = 1,
                                               scale_model = true))
    @test Matrix(Q) ≈ Matrix(Qref)
end

@testset "Generic1 component — eigenvalue rescaling" begin
    # Build a structure matrix whose top eigenvalue is clearly > 1.
    n = 5
    rng = Random.Xoshiro(7)
    M = randn(rng, n, n)
    R = sparse(Symmetric(M'M + 2 * I(n)))    # λ_max > 1 almost surely

    c = Generic1(R)
    @test length(c) == n
    @test nhyperparameters(c) == 1

    # The component's internal R should have λ_max == 1.
    λ_max_internal = maximum(eigvals(Symmetric(Matrix(c.R))))
    @test λ_max_internal ≈ 1.0 atol = 1.0e-10

    # And the precision at τ = 1 should equal (original R) / λ_max_original.
    Q = precision_matrix(c, [0.0])
    @test Matrix(Q) ≈ Matrix(R) ./ c.λ_max_original

    @test isfinite(log_hyperprior(c, [0.0]))
end

@testset "Generic1 — rejects negative λ_max" begin
    R_neg = sparse([-1.0 0.0; 0.0 -2.0])
    @test_throws ArgumentError Generic1(R_neg)
end

@testset "BYM component" begin
    # Classical BYM = iid + besag. Latent layout [v; u] of length 2n.
    W = [0 1 0 0;
         1 0 1 0;
         0 1 0 1;
         0 0 1 0]
    g = GMRFGraph(W)
    n = num_nodes(g)

    c = BYM(g; scale_model = false)
    @test length(c) == 2n
    @test nhyperparameters(c) == 2
    @test initial_hyperparameters(c) == [0.0, 0.0]

    # Q([log 2, log 3]) = blockdiag(2·I_n, 3·Q_besag).
    Q = precision_matrix(c, [log(2.0), log(3.0)])
    @test size(Q) == (2n, 2n)
    Qv = Matrix(Q[1:n, 1:n])
    Qu = Matrix(Q[(n + 1):(2n), (n + 1):(2n)])
    Q_off = Matrix(Q[1:n, (n + 1):(2n)])
    @test Qv ≈ 2.0 * I(n)
    Q_besag_ref = Matrix(GMRFs.precision_matrix(BesagGMRF(g; τ = 3.0,
                                                          scale_model = false)))
    @test Qu ≈ Q_besag_ref
    @test all(iszero, Q_off)                        # blocks are independent

    # Per-component sum-to-zero only on the spatial block.
    kc = GMRFs.constraints(c)
    @test kc isa LinearConstraint
    A = constraint_matrix(kc)
    @test size(A) == (nconnected_components(g), 2n)
    @test all(iszero, A[:, 1:n])                    # v block unconstrained
    @test Matrix(A[:, (n + 1):(2n)]) ==
          Matrix(constraint_matrix(GMRFs.sum_to_zero_constraints(g)))
    @test constraint_rhs(kc) == zeros(size(A, 1))

    @test isfinite(log_hyperprior(c, [0.0, 0.0]))
end

@testset "Leroux component" begin
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

    # ρ → 0 (logit → -∞ asymptotically): Q ≈ τ I. Use a finite logit
    # that puts ρ ≈ 10⁻⁴ so numerically we can compare components.
    θ_iid = [log(4.0), -10.0]
    ρ_iid = inv(1 + exp(10.0))
    Q_iid = Matrix(precision_matrix(c, θ_iid))
    @test Q_iid ≈ 4.0 * ((1 - ρ_iid) .* Matrix(I, n, n) .+ ρ_iid .* R_ref)

    # ρ = 0.8 explicit value.
    θ2 = [0.0, log(0.8 / 0.2)]                      # logit(0.8)
    Q2 = Matrix(precision_matrix(c, θ2))
    @test Q2 ≈ 0.2 .* Matrix(I, n, n) .+ 0.8 .* R_ref

    # Default Leroux has no sum-to-zero constraint (ρ strictly < 1 on
    # the internal logit scale keeps Q positive-definite).
    @test GMRFs.constraints(c) isa GMRFs.NoConstraint

    @test isfinite(log_hyperprior(c, [0.0, 0.0]))
end

@testset "LogitBeta prior" begin
    # Beta(1, 1) = Uniform on ρ; log density on θ is log ρ + log(1-ρ).
    p = LogitBeta(1.0, 1.0)
    for θ in (-2.0, -0.5, 0.0, 0.7, 3.0)
        ρ = inv(1 + exp(-θ))
        @test LatentGaussianModels.log_prior_density(p, θ) ≈
              log(ρ) + log1p(-ρ)
    end

    # Beta(2, 5): check against manual computation.
    p2 = LogitBeta(2.0, 5.0)
    θ = 0.3
    ρ = inv(1 + exp(-θ))
    log_B = Distributions.loggamma(2.0) + Distributions.loggamma(5.0) -
            Distributions.loggamma(7.0)
    @test LatentGaussianModels.log_prior_density(p2, θ) ≈
          2.0 * log(ρ) + 5.0 * log1p(-ρ) - log_B
end

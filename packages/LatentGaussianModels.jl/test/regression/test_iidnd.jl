using LatentGaussianModels: IIDND, IID2D, IIDND_Sep, AbstractIIDND,
                            PCPrecision, PCCor0, GaussianPrior,
                            precision_matrix, log_hyperprior, nhyperparameters,
                            initial_hyperparameters, gmrf, log_prior_density
using LatentGaussianModels: log_normalizing_constant
using LinearAlgebra: I, Diagonal, det, kron, issymmetric, logdet

@testset "IID2D — basic shape" begin
    c = IID2D(5)
    @test c isa IIDND_Sep{2}
    @test c isa AbstractIIDND
    @test length(c) == 10                  # 2 × 5 latent slots
    @test nhyperparameters(c) == 3         # log τ_1, log τ_2, atanh ρ
    @test initial_hyperparameters(c) == zeros(3)
    @test c.precpriors[1] isa PCPrecision
    @test c.precpriors[2] isa PCPrecision
    @test c.corrpriors[1] isa PCCor0
end

@testset "IIDND — argument validation" begin
    @test_throws ArgumentError IIDND(0, 2)
    @test_throws ArgumentError IIDND(-1, 2)
    @test_throws ArgumentError IIDND(5, 1)
    @test_throws ArgumentError IIDND(5, 3)         # PR-1b territory
    @test_throws ArgumentError IIDND(5, 2;
        hyperprior_corr=PCCor0(),
        hyperprior_corrs=(PCCor0(),))
    @test_throws ArgumentError IIDND(5, 2;
        hyperprior_precs=(PCPrecision(),))         # wrong length
end

@testset "IID2D — precision matrix vs Λ ⊗ I_n" begin
    n = 4
    c = IID2D(n)
    τ1 = 2.0
    τ2 = 3.5
    ρ = 0.4
    θ = [log(τ1), log(τ2), atanh(ρ)]

    Q = precision_matrix(c, θ)
    @test size(Q) == (2n, 2n)
    @test issymmetric(Q)

    # Reference Λ from the closed-form inverse of the 2×2 covariance.
    σ2 = [1/τ1            ρ / sqrt(τ1*τ2);
          ρ/sqrt(τ1*τ2)   1/τ2]
    Λ = inv(σ2)
    Qref = kron(Λ, Matrix(1.0I, n, n))
    @test Matrix(Q) ≈ Qref

    # det(Q) = det(Λ)^n = (τ_1 τ_2 / (1 - ρ²))^n
    expected_logdet = n * (log(τ1) + log(τ2) - log1p(-ρ^2))
    @test logdet(Matrix(Q)) ≈ expected_logdet rtol=1.0e-10
end

@testset "IID2D — independence limit (ρ = 0)" begin
    n = 3
    c = IID2D(n)
    τ1 = 1.5
    τ2 = 2.0
    θ = [log(τ1), log(τ2), 0.0]
    Q = precision_matrix(c, θ)
    Qref = Diagonal([fill(τ1, n); fill(τ2, n)])
    @test Matrix(Q) ≈ Matrix(Qref)
end

@testset "IID2D — log_normalizing_constant matches ½ log det Q − ½d log 2π" begin
    n = 6
    c = IID2D(n)
    for (τ1, τ2, ρ) in ((1.0, 1.0, 0.0),
                       (2.0, 0.5, 0.3),
                       (4.0, 4.0, -0.7))
        θ = [log(τ1), log(τ2), atanh(ρ)]
        Q = precision_matrix(c, θ)
        d = 2n
        expected = -0.5 * d * log(2π) + 0.5 * logdet(Matrix(Q))
        @test log_normalizing_constant(c, θ) ≈ expected rtol=1.0e-10
    end
end

@testset "IID2D — log_hyperprior is the sum of three priors" begin
    n = 4
    c = IID2D(n;
        hyperprior_precs=(PCPrecision(1.0, 0.01), PCPrecision(2.0, 0.05)),
        hyperprior_corr=PCCor0(0.5, 0.5))
    θ = [0.3, -0.2, 0.5]

    # Manual sum
    expected =
        log_prior_density(c.precpriors[1], θ[1]) +
        log_prior_density(c.precpriors[2], θ[2]) +
        log_prior_density(c.corrpriors[1], θ[3])
    @test log_hyperprior(c, θ) ≈ expected
end

@testset "IID2D — gmrf wrapper exposes the same Q" begin
    n = 5
    c = IID2D(n)
    θ = [log(1.5), log(2.5), atanh(0.2)]
    g = gmrf(c, θ)
    @test Matrix(GMRFs.precision_matrix(g)) ≈ Matrix(precision_matrix(c, θ))
end

@testset "IID2D — alternate priors (Gaussian on Fisher z, à la R-INLA 2diid)" begin
    # R-INLA's `2diid` defaults use a Gaussian-on-atanh-ρ prior; verify
    # the public kwarg path accepts it without ceremony.
    c = IID2D(3;
        hyperprior_precs=(PCPrecision(), PCPrecision()),
        hyperprior_corr=GaussianPrior(0.0, sqrt(1 / 0.2)))
    @test c.corrpriors[1] isa GaussianPrior
    @test isfinite(log_hyperprior(c, [0.0, 0.0, 0.5]))
end

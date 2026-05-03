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
    @test_throws ArgumentError IIDND(5, 4)         # ADR-022 caps separable form at N ≤ 3
    @test_throws ArgumentError IIDND(5, 2;
        hyperprior_corr=PCCor0(),
        hyperprior_corrs=(PCCor0(),))
    @test_throws ArgumentError IIDND(5, 2;
        hyperprior_precs=(PCPrecision(),))         # wrong length
    @test_throws ArgumentError IIDND(5, 3;
        hyperprior_corrs=(PCCor0(), PCCor0()))     # wrong length (need 3)
    @test_throws ArgumentError IIDND(5, 3;
        hyperprior_precs=(PCPrecision(), PCPrecision()))   # wrong length (need 3)
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

# ---------------------------------------------------------------------
# Phase I-A PR-1b — IID3D (N = 3, LKJ stick-breaking).
# ---------------------------------------------------------------------

# Build the dense 3×3 correlation matrix R from LKJ canonical partial
# correlations `(z21, z31, z32)`. Used as the closed-form reference
# against which `precision_matrix(c::IIDND_Sep{3}, …)` is verified.
function _lkj_R_from_z(z21, z31, z32)
    L = zeros(3, 3)
    L[1, 1] = 1.0
    L[2, 1] = z21
    L[2, 2] = sqrt(1 - z21^2)
    L[3, 1] = z31
    L[3, 2] = z32 * sqrt(1 - z31^2)
    L[3, 3] = sqrt((1 - z31^2) * (1 - z32^2))
    return L * L'
end

@testset "IID3D — basic shape" begin
    c = IID3D(4)
    @test c isa IIDND_Sep{3}
    @test c isa AbstractIIDND
    @test length(c) == 12                 # 3 × 4 latent slots
    @test nhyperparameters(c) == 6        # log τ_1, log τ_2, log τ_3, atanh z_{2,1}, _{3,1}, _{3,2}
    @test initial_hyperparameters(c) == zeros(6)
    @test all(p -> p isa PCPrecision, c.precpriors)
    @test all(p -> p isa PCCor0, c.corrpriors)
end

@testset "IID3D — precision matrix vs closed-form Λ ⊗ I_n" begin
    n = 3
    c = IID3D(n)
    τ = (1.5, 2.5, 0.8)
    z = (0.3, -0.4, 0.2)        # canonical partial correlations
    θ = [log(τ[1]), log(τ[2]), log(τ[3]),
        atanh(z[1]), atanh(z[2]), atanh(z[3])]

    Q = precision_matrix(c, θ)
    @test size(Q) == (3n, 3n)
    @test issymmetric(Q)

    # Reference: build R via LKJ stick-breaking, then Σ = D_τ^{-1/2} R D_τ^{-1/2}.
    R = _lkj_R_from_z(z...)
    D_inv_half = Diagonal([1 / sqrt(τ[1]), 1 / sqrt(τ[2]), 1 / sqrt(τ[3])])
    Σ = D_inv_half * R * D_inv_half
    Λ = inv(Σ)
    Qref = kron(Λ, Matrix(1.0I, n, n))
    @test Matrix(Q) ≈ Qref rtol=1.0e-12

    # det Λ = τ_1 τ_2 τ_3 / det R, and det R = (1 - z21²)(1 - z31²)(1 - z32²)
    # under LKJ stick-breaking. log det Q = n · log det Λ.
    expected_logdet = n * (log(τ[1]) + log(τ[2]) + log(τ[3]) -
                           log1p(-z[1]^2) - log1p(-z[2]^2) - log1p(-z[3]^2))
    @test logdet(Matrix(Q)) ≈ expected_logdet rtol=1.0e-10
end

@testset "IID3D — independence limit (all CPCs = 0)" begin
    n = 4
    c = IID3D(n)
    τ = (1.0, 2.0, 3.0)
    θ = [log(τ[1]), log(τ[2]), log(τ[3]), 0.0, 0.0, 0.0]
    Q = precision_matrix(c, θ)
    Qref = Diagonal(vcat(fill(τ[1], n), fill(τ[2], n), fill(τ[3], n)))
    @test Matrix(Q) ≈ Matrix(Qref)
end

@testset "IID3D — bivariate-block reduction (z31 = z32 = 0)" begin
    # With z_{3,1} = z_{3,2} = 0, variable 3 is independent of (1, 2);
    # the upper-left 2 × 2 block of Λ must coincide with `IID2D`.
    n = 3
    τ = (1.5, 2.0, 0.7)
    a = atanh(0.3)              # z_{2,1} = 0.3
    θ_3d = [log(τ[1]), log(τ[2]), log(τ[3]), a, 0.0, 0.0]
    θ_2d = [log(τ[1]), log(τ[2]), a]
    Q3 = precision_matrix(IID3D(n), θ_3d)
    Q2 = precision_matrix(IID2D(n), θ_2d)
    @test Matrix(Q3)[1:(2n), 1:(2n)] ≈ Matrix(Q2)
    # Variable-3 block is independent τ_3 · I_n.
    @test Matrix(Q3)[(2n + 1):(3n), (2n + 1):(3n)] ≈ Diagonal(fill(τ[3], n))
    @test all(iszero, Matrix(Q3)[1:(2n), (2n + 1):(3n)])
end

@testset "IID3D — log_normalizing_constant matches ½ log det Q − ½d log 2π" begin
    n = 5
    c = IID3D(n)
    for (τ1, τ2, τ3, z21, z31, z32) in (
            (1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
            (2.0, 0.5, 1.5, 0.3, -0.2, 0.1),
            (4.0, 0.25, 9.0, -0.6, 0.5, -0.3))
        θ = [log(τ1), log(τ2), log(τ3),
            atanh(z21), atanh(z31), atanh(z32)]
        Q = precision_matrix(c, θ)
        d = 3n
        expected = -0.5 * d * log(2π) + 0.5 * logdet(Matrix(Q))
        @test log_normalizing_constant(c, θ) ≈ expected rtol=1.0e-10
    end
end

@testset "IID3D — log_hyperprior is the sum of six priors" begin
    c = IID3D(4;
        hyperprior_precs=(PCPrecision(1.0, 0.01),
                          PCPrecision(2.0, 0.05),
                          PCPrecision(0.5, 0.10)),
        hyperprior_corrs=(PCCor0(0.3, 0.5),
                          PCCor0(0.5, 0.5),
                          PCCor0(0.7, 0.3)))
    θ = [0.3, -0.2, 0.1, 0.4, -0.5, 0.2]
    expected = sum(log_prior_density(c.precpriors[k], θ[k]) for k in 1:3) +
               sum(log_prior_density(c.corrpriors[k], θ[3 + k]) for k in 1:3)
    @test log_hyperprior(c, θ) ≈ expected
end

@testset "IID3D — gmrf wrapper exposes the same Q" begin
    n = 4
    c = IID3D(n)
    θ = [log(1.5), log(2.5), log(0.8),
        atanh(0.3), atanh(-0.4), atanh(0.2)]
    g = gmrf(c, θ)
    @test Matrix(GMRFs.precision_matrix(g)) ≈ Matrix(precision_matrix(c, θ))
end

@testset "IID3D — saturation stability (large |θ|)" begin
    # Outer-θ LBFGS line searches can probe |θ| ≫ 19, where
    # `1 - tanh(θ)²` underflows to 0. The cosh/sinh form must keep
    # `precision_matrix` and `log_normalizing_constant` finite.
    n = 3
    c = IID3D(n)
    θ = [0.0, 0.0, 0.0, 25.0, -25.0, 22.0]
    Q = precision_matrix(c, θ)
    @test all(isfinite, Matrix(Q))
    @test isfinite(log_normalizing_constant(c, θ))
end

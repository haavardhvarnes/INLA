using LatentGaussianModels: ZeroInflatedPoissonLikelihood0,
                            ZeroInflatedPoissonLikelihood1,
                            ZeroInflatedPoissonLikelihood2,
                            ZeroInflatedBinomialLikelihood0,
                            ZeroInflatedBinomialLikelihood1,
                            ZeroInflatedBinomialLikelihood2,
                            ZeroInflatedNegativeBinomialLikelihood0,
                            ZeroInflatedNegativeBinomialLikelihood1,
                            ZeroInflatedNegativeBinomialLikelihood2,
                            log_density, ∇_η_log_density, ∇²_η_log_density,
                            ∇³_η_log_density, log_hyperprior, nhyperparameters,
                            initial_hyperparameters, pointwise_log_density

# Reuse the FD helper from test_likelihoods.jl style.
function _fd_grad(f, η, h = 1.0e-6)
    g = similar(η)
    for i in eachindex(η)
        ep = copy(η)
        ep[i] += h
        em = copy(η)
        em[i] -= h
        g[i] = (f(ep) - f(em)) / (2h)
    end
    return g
end

const ZI_TOL_GRAD = 1.0e-4
const ZI_TOL_HESS = 1.0e-3
const ZI_TOL_TRIPLE = 1.0e-2

# Common test vectors. Mix zeros and positive counts to exercise both
# branches.
const _ZI_Y_COUNT = Float64[0, 2, 0, 5, 1, 0, 7]
const _ZI_Y_BIN = [0, 3, 0, 8, 1, 0, 10]
const _ZI_N_TRIALS = [10, 10, 12, 10, 8, 10, 10]
const _ZI_η = [0.2, -0.4, 0.7, 0.0, 1.0, -1.5, 0.5]
const _ZI_E = [1.0, 1.5, 0.8, 2.0, 1.0, 1.2, 0.9]

@testset "ZeroInflatedPoisson — types 0/1/2" begin
    for (T, name, θ) in (
            (ZeroInflatedPoissonLikelihood0, "ZIP0", [-0.5]),
            (ZeroInflatedPoissonLikelihood1, "ZIP1", [-0.5]),
            (ZeroInflatedPoissonLikelihood2, "ZIP2", [0.4]),
        )
        @testset "$name" begin
            ℓ = T(; E = _ZI_E)
            @test nhyperparameters(ℓ) == 1
            @test length(initial_hyperparameters(ℓ)) == 1
            lp = log_density(ℓ, _ZI_Y_COUNT, _ZI_η, θ)
            @test isfinite(lp)

            g = ∇_η_log_density(ℓ, _ZI_Y_COUNT, _ZI_η, θ)
            g_fd = _fd_grad(h -> log_density(ℓ, _ZI_Y_COUNT, h, θ), _ZI_η)
            @test g ≈ g_fd atol = ZI_TOL_GRAD

            H = ∇²_η_log_density(ℓ, _ZI_Y_COUNT, _ZI_η, θ)
            H_fd = _fd_grad(h -> sum(∇_η_log_density(ℓ, _ZI_Y_COUNT, h, θ)), _ZI_η)
            @test H ≈ H_fd atol = ZI_TOL_HESS

            pw = pointwise_log_density(ℓ, _ZI_Y_COUNT, _ZI_η, θ)
            @test sum(pw) ≈ lp
            @test isfinite(log_hyperprior(ℓ, θ))
        end
    end

    # Type 1 has a closed-form ∇³.
    @testset "ZIP1 ∇³" begin
        ℓ = ZeroInflatedPoissonLikelihood1(; E = _ZI_E)
        θ = [-0.5]
        H3 = ∇³_η_log_density(ℓ, _ZI_Y_COUNT, _ZI_η, θ)
        H3_fd = _fd_grad(h -> sum(∇²_η_log_density(ℓ, _ZI_Y_COUNT, h, θ)), _ZI_η)
        @test H3 ≈ H3_fd atol = ZI_TOL_TRIPLE
    end
end

@testset "ZeroInflatedBinomial — types 0/1/2" begin
    for (T, name, θ) in (
            (ZeroInflatedBinomialLikelihood0, "ZIB0", [-0.5]),
            (ZeroInflatedBinomialLikelihood1, "ZIB1", [-0.5]),
            (ZeroInflatedBinomialLikelihood2, "ZIB2", [0.4]),
        )
        @testset "$name" begin
            ℓ = T(_ZI_N_TRIALS)
            @test nhyperparameters(ℓ) == 1
            lp = log_density(ℓ, _ZI_Y_BIN, _ZI_η, θ)
            @test isfinite(lp)

            g = ∇_η_log_density(ℓ, _ZI_Y_BIN, _ZI_η, θ)
            g_fd = _fd_grad(h -> log_density(ℓ, _ZI_Y_BIN, h, θ), _ZI_η)
            @test g ≈ g_fd atol = ZI_TOL_GRAD

            H = ∇²_η_log_density(ℓ, _ZI_Y_BIN, _ZI_η, θ)
            H_fd = _fd_grad(h -> sum(∇_η_log_density(ℓ, _ZI_Y_BIN, h, θ)), _ZI_η)
            @test H ≈ H_fd atol = ZI_TOL_HESS

            pw = pointwise_log_density(ℓ, _ZI_Y_BIN, _ZI_η, θ)
            @test sum(pw) ≈ lp
        end
    end

    @testset "ZIB1 ∇³" begin
        ℓ = ZeroInflatedBinomialLikelihood1(_ZI_N_TRIALS)
        θ = [-0.5]
        H3 = ∇³_η_log_density(ℓ, _ZI_Y_BIN, _ZI_η, θ)
        H3_fd = _fd_grad(h -> sum(∇²_η_log_density(ℓ, _ZI_Y_BIN, h, θ)), _ZI_η)
        @test H3 ≈ H3_fd atol = ZI_TOL_TRIPLE
    end
end

@testset "ZeroInflatedNegativeBinomial — types 0/1/2" begin
    for (T, name, θ) in (
            (ZeroInflatedNegativeBinomialLikelihood0, "ZINB0", [0.5, -0.5]),
            (ZeroInflatedNegativeBinomialLikelihood1, "ZINB1", [0.5, -0.5]),
            (ZeroInflatedNegativeBinomialLikelihood2, "ZINB2", [0.5, 0.4]),
        )
        @testset "$name" begin
            ℓ = T(; E = _ZI_E)
            @test nhyperparameters(ℓ) == 2
            @test length(initial_hyperparameters(ℓ)) == 2
            lp = log_density(ℓ, _ZI_Y_COUNT, _ZI_η, θ)
            @test isfinite(lp)

            g = ∇_η_log_density(ℓ, _ZI_Y_COUNT, _ZI_η, θ)
            g_fd = _fd_grad(h -> log_density(ℓ, _ZI_Y_COUNT, h, θ), _ZI_η)
            @test g ≈ g_fd atol = ZI_TOL_GRAD

            H = ∇²_η_log_density(ℓ, _ZI_Y_COUNT, _ZI_η, θ)
            H_fd = _fd_grad(h -> sum(∇_η_log_density(ℓ, _ZI_Y_COUNT, h, θ)), _ZI_η)
            @test H ≈ H_fd atol = ZI_TOL_HESS

            pw = pointwise_log_density(ℓ, _ZI_Y_COUNT, _ZI_η, θ)
            @test sum(pw) ≈ lp
            @test isfinite(log_hyperprior(ℓ, θ))
        end
    end
end

# Numerical-stability smoke checks at extreme hyperparameter values.
# Hot-path code paths use logsumexp / -expm1 / log1p reductions to stay
# finite when π → 0 or π → 1; this guards regressions in those forms.
@testset "ZeroInflated — extreme hyperparameter smoke" begin
    y_zero = zeros(4)
    η = [0.2, -0.4, 0.7, 0.0]

    for ℓ in (
            ZeroInflatedPoissonLikelihood0(),
            ZeroInflatedPoissonLikelihood1(),
            ZeroInflatedPoissonLikelihood2(),
            ZeroInflatedNegativeBinomialLikelihood0(),
            ZeroInflatedNegativeBinomialLikelihood1(),
            ZeroInflatedNegativeBinomialLikelihood2(),
        )
        θ_extreme = nhyperparameters(ℓ) == 1 ? [10.0] : [0.5, 10.0]
        @test isfinite(log_density(ℓ, y_zero, η, θ_extreme))
        θ_low = nhyperparameters(ℓ) == 1 ? [-10.0] : [0.5, -10.0]
        @test isfinite(log_density(ℓ, y_zero, η, θ_low))
    end

    n_trials = [10, 12, 8, 10]
    for ℓ in (
            ZeroInflatedBinomialLikelihood0(n_trials),
            ZeroInflatedBinomialLikelihood1(n_trials),
            ZeroInflatedBinomialLikelihood2(n_trials),
        )
        @test isfinite(log_density(ℓ, [0, 3, 0, 5], η, [10.0]))
        @test isfinite(log_density(ℓ, [0, 3, 0, 5], η, [-10.0]))
    end
end

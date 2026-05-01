using LatentGaussianModels: GaussianLikelihood, IdentityLink, Intercept,
                            LatentGaussianModel, laplace_mode, Laplace
using LinearAlgebra
using SparseArrays
using Random

# Closed-form Laplace for y_i = α + ε_i, ε_i ~ N(0, τ⁻¹):
#   prior: α ~ N(0, prec⁻¹), posterior is conjugate.
#   Posterior mean: α̂ = (τ sum(y)) / (prec + n τ)
#   Posterior precision: prec + n τ

@testset "Gaussian+Identity+Intercept — closed form" begin
    rng = Random.Xoshiro(42)
    n = 30
    α_true = 2.0
    σ = 0.5
    τ_true = 1 / σ^2
    y = α_true .+ σ .* randn(rng, n)

    # One intercept, projector A = ones(n, 1).
    c = Intercept()
    A = sparse(ones(n, 1))
    ℓ = GaussianLikelihood()
    model = LatentGaussianModel(ℓ, (c,), A)

    θ = [log(τ_true)]   # only likelihood has hyperparameters (τ); Intercept has none
    res = laplace_mode(model, y, θ; strategy=Laplace())

    prec = 1.0e-3    # default Intercept prior precision
    α̂_expected = (τ_true * sum(y)) / (prec + n * τ_true)
    H_expected = prec + n * τ_true

    @test res.converged
    @test length(res.mode) == 1
    @test res.mode[1]≈α̂_expected rtol=1.0e-6
    @test res.precision[1, 1]≈H_expected rtol=1.0e-8
end

@testset "Gaussian+Identity+IID — recover field under tight prior" begin
    rng = Random.Xoshiro(7)
    n = 10
    σ = 0.2
    τ_obs = 1 / σ^2
    x_true = randn(rng, n)
    y = x_true .+ σ .* randn(rng, n)

    c = LatentGaussianModels.IID(n)
    A = sparse(I, n, n)
    ℓ = GaussianLikelihood()
    model = LatentGaussianModel(ℓ, (c,), A)

    θ = [log(τ_obs), log(1.0)]   # τ_obs, IID prior precision
    res = laplace_mode(model, y, θ)

    # Closed form: posterior precision = (τ_x + τ_obs) I; mean = (τ_obs / (τ_x+τ_obs)) y.
    τ_x = 1.0
    shrink = τ_obs / (τ_x + τ_obs)
    @test res.mode≈shrink .* y rtol=1.0e-6
    @test isapprox(diag(res.precision), fill(τ_x + τ_obs, n); rtol=1.0e-8)
end

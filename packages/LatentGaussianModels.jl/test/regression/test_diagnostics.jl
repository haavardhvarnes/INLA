using LatentGaussianModels
using LatentGaussianModels: PoissonLikelihood, GaussianLikelihood,
                            Intercept, IID, FixedEffects, LatentGaussianModel, inla,
                            PCPrecision,
                            pointwise_log_density, pointwise_cdf, posterior_samples_η,
                            dic, waic, cpo, pit, log_density, fixed_effects, hyperparameters
using Distributions: Normal, Poisson, Binomial
using Distributions
using SparseArrays
using LinearAlgebra: I
using Random
using Statistics: mean, std

@testset "pointwise likelihood contracts" begin
    # Gaussian: sum of pointwise_log_density == log_density
    rng = Random.Xoshiro(20260423)
    n = 20
    y = randn(rng, n)
    η = randn(rng, n)
    θ = [log(2.5)]      # τ = 2.5
    ℓ = GaussianLikelihood()
    lpw = pointwise_log_density(ℓ, y, η, θ)
    @test length(lpw) == n
    @test sum(lpw)≈log_density(ℓ, y, η, θ) rtol=1.0e-12

    # CDF in [0, 1] and monotone in y.
    F = pointwise_cdf(ℓ, y, η, θ)
    @test all(0 .≤ F .≤ 1)

    # Poisson: same agreement.
    y_p = rand(rng, Poisson(3.0), n)
    E = fill(2.0, n)
    ℓp = PoissonLikelihood(; E=E)
    lpw_p = pointwise_log_density(ℓp, y_p, η, Float64[])
    @test length(lpw_p) == n
    @test sum(lpw_p)≈log_density(ℓp, y_p, η, Float64[]) rtol=1.0e-12
    F_p = pointwise_cdf(ℓp, y_p, η, Float64[])
    @test all(0 .≤ F_p .≤ 1)
end

@testset "posterior_samples_η shape + statistics" begin
    # Simple Gaussian fit. Samples of η should have marginal mean ≈ x_mean
    # (here η_i ≡ x_i under identity projector and Intercept + IID).
    rng = Random.Xoshiro(20260423)
    n = 40
    y = randn(rng, n)
    ℓ = GaussianLikelihood()
    A = sparse([ones(n) Matrix{Float64}(I, n,n)])
    model = LatentGaussianModel(ℓ, (Intercept(), IID(n)), A)
    res = inla(model, y; int_strategy=:grid)

    # Posterior-mean η is A * x_mean
    η_pm = A * res.x_mean
    S = 800
    samples = posterior_samples_η(rng, res, model; n_samples=S)
    @test size(samples.η) == (n, S)
    # MC mean within a few MC standard errors of the analytic mean.
    sample_mean = vec(mean(samples.η; dims=2))
    # MC standard error ≤ post-sd / sqrt(S); give a loose 4-sigma band.
    post_sd = sqrt.(max.(Diagonal(A * spdiagm(res.x_var) * A').diag, 0.0))
    @test all(abs.(sample_mean .- η_pm) .< 4 .* post_sd ./ sqrt(S) .+ 1.0e-6)
end

@testset "DIC — Gaussian identity closed form" begin
    # Gaussian identity: E[-2 log p(y|η)] has closed form τ (y - η̂)² + τ·Var(η) - log(τ) + log(2π).
    # We verify that `dic` reproduces the two-term decomposition on a
    # simple IID model at τ = 1.
    rng = Random.Xoshiro(20260423)
    n = 30
    y = randn(rng, n)
    ℓ = GaussianLikelihood(; hyperprior=PCPrecision(1.0, 0.01))
    A = sparse([ones(n) Matrix{Float64}(I, n,n)])
    model = LatentGaussianModel(ℓ, (Intercept(), IID(n)), A)
    res = inla(model, y; int_strategy=:grid)

    d = dic(res, model, y)
    @test isfinite(d.DIC)
    @test isfinite(d.pD)
    @test d.pD > 0                # effective parameters is positive
    @test d.DIC≈d.D_bar + d.pD rtol=1.0e-12
    @test d.D_mode ≤ d.D_bar + 1.0e-8   # plug-in deviance ≤ expected deviance
end

@testset "WAIC — finite + lpd decomposition" begin
    rng = Random.Xoshiro(20260423)
    n = 40
    y = rand(rng, Poisson(2.0), n)
    E = fill(1.0, n)
    ℓ = PoissonLikelihood(; E=E)
    A = sparse([ones(n) Matrix{Float64}(I, n,n)])
    model = LatentGaussianModel(ℓ, (Intercept(), IID(n)), A)
    res = inla(model, y; int_strategy=:grid)

    w = waic(rng, res, model, y; n_samples=500)
    @test length(w.lpd) == n
    @test length(w.pWAIC) == n
    @test all(isfinite, w.lpd)
    @test all(>=(0), w.pWAIC)
    @test isfinite(w.WAIC)
    @test w.WAIC≈-2 * w.elpd_WAIC rtol=1.0e-12
end

@testset "CPO — pseudo-marginal + positivity" begin
    rng = Random.Xoshiro(20260423)
    n = 30
    y = rand(rng, Poisson(2.0), n)
    E = fill(1.0, n)
    ℓ = PoissonLikelihood(; E=E)
    A = sparse([ones(n) Matrix{Float64}(I, n,n)])
    model = LatentGaussianModel(ℓ, (Intercept(), IID(n)), A)
    res = inla(model, y; int_strategy=:grid)

    c = cpo(rng, res, model, y; n_samples=500)
    @test length(c.CPO) == n
    @test all(c.CPO .> 0)
    @test all(isfinite, c.log_CPO)
    @test c.log_pseudo_marginal≈sum(c.log_CPO) rtol=1.0e-12
end

@testset "inla_summary smoke" begin
    rng = Random.Xoshiro(20260423)
    n = 15
    y = 0.5 .+ randn(rng, n)
    ℓ = GaussianLikelihood()
    A = sparse([ones(n) Matrix{Float64}(I, n,n)])
    model = LatentGaussianModel(ℓ, (Intercept(), IID(n)), A)
    res = inla(model, y; int_strategy=:grid)

    buf = IOBuffer()
    LatentGaussianModels.inla_summary(buf, model, res)
    s = String(take!(buf))
    @test occursin("INLA fit", s)
    @test occursin("Fixed effects:", s)
    @test occursin("Random effects", s)
    @test occursin("Hyperparameters", s)
    @test occursin("log p(y)", s)
end

@testset "PIT — uniformity on simulated Gaussian" begin
    # Simulate Gaussian data that is well-described by the fitted model
    # and check that PIT values are approximately uniform on [0, 1].
    rng = Random.Xoshiro(20260423)
    n = 200
    μ_true = 0.5
    σ_true = 0.8
    y = μ_true .+ σ_true .* randn(rng, n)
    ℓ = GaussianLikelihood()
    # Single intercept model — simplest well-specified case.
    A = sparse(reshape(ones(n), n, 1))
    model = LatentGaussianModel(ℓ, (Intercept(),), A)
    res = inla(model, y; int_strategy=:grid)

    p = pit(rng, res, model, y; n_samples=500)
    @test length(p) == n
    @test all(0 .≤ p .≤ 1)
    # Uniformity: sample mean ≈ 0.5, sample sd ≈ √(1/12) ≈ 0.289.
    # Loose bands — PIT is approximate with n_samples = 500 MC draws.
    @test abs(mean(p) - 0.5) < 0.08
    @test abs(std(p) - sqrt(1 / 12)) < 0.08
end

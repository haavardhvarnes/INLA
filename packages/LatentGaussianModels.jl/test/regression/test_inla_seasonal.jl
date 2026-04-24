using LatentGaussianModels: Intercept, Seasonal, GaussianLikelihood,
    LatentGaussianModel, PCPrecision, inla, laplace_mode,
    fixed_effects, random_effects, hyperparameters
using GMRFs: GMRFs, SeasonalGMRF, constraint_matrix
using SparseArrays
using LinearAlgebra: I
using Random

@testset "Seasonal — Laplace enforces null-space constraints" begin
    # Gaussian likelihood with a known seasonal latent field. The
    # Laplace MAP must satisfy C x̂ ≈ 0 for the SeasonalGMRF's default
    # (s-1)-row constraint.
    rng = Random.Xoshiro(20260424)
    n = 24
    s = 4

    # True seasonal pattern: one period of length s with zero sum,
    # repeated n/s times.
    base = [1.5, -0.5, -1.5, 0.5]
    x_true = repeat(base, n ÷ s)
    σ_obs = 0.3
    y = x_true .+ σ_obs .* randn(rng, n)

    c_seas = Seasonal(n; period = s, hyperprior = PCPrecision(1.0, 0.01))
    A_proj = sparse(1.0I, n, n)
    ℓ = GaussianLikelihood()
    model = LatentGaussianModel(ℓ, (c_seas,), A_proj)

    # Evaluate Laplace at a fixed θ = [log τ_ε, log τ_x] ≈ truth.
    θ = [log(1 / σ_obs^2), log(1.0)]
    res = laplace_mode(model, y, θ)
    @test res.converged
    @test res.constraint !== nothing

    C = constraint_matrix(GMRFs.constraints(SeasonalGMRF(n; period = s)))
    @test maximum(abs, C * res.mode) < 1.0e-8
end

@testset "INLA — Gaussian + Intercept + Seasonal (synthetic)" begin
    rng = Random.Xoshiro(20260424)
    n = 36
    s = 6

    α_true = 0.2
    base = randn(rng, s)
    base .-= sum(base) / s                       # enforce ground-truth zero-sum
    x_true = repeat(base, n ÷ s)
    σ_obs = 0.4
    y = α_true .+ x_true .+ σ_obs .* randn(rng, n)

    c_int = Intercept()
    c_seas = Seasonal(n; period = s, hyperprior = PCPrecision(1.0, 0.01))
    A_proj = sparse([ones(n) Matrix{Float64}(I, n, n)])
    ℓ = GaussianLikelihood()
    model = LatentGaussianModel(ℓ, (c_int, c_seas), A_proj)

    res = inla(model, y; int_strategy = :grid)

    @test isfinite(res.log_marginal)

    fe = fixed_effects(model, res)
    @test length(fe) == 1
    @test abs(fe[1].mean - α_true) < 3 * fe[1].sd

    re = random_effects(model, res)
    v = first(values(re))
    @test length(v.mean) == n
    @test all(v.sd .> 0)

    hp = hyperparameters(model, res)
    @test length(hp) == 2                         # Gaussian τ_ε + Seasonal τ_x
    @test all(isfinite(r.mean) for r in hp)
    @test all(r.sd > 0 for r in hp)
end

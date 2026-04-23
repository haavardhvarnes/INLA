using LatentGaussianModels: GaussianLikelihood, Intercept, IID, LatentGaussianModel,
    inla, posterior_marginal_x, posterior_marginal_θ,
    fixed_effects, random_effects, hyperparameters, log_marginal_likelihood
using SparseArrays
using Random
using Statistics: mean

@testset "posterior_marginal_x — densities integrate to 1" begin
    rng = Random.Xoshiro(20260423)
    n = 150
    y = 1.0 .+ 0.5 .* randn(rng, n)

    c = Intercept()
    A = sparse(ones(n, 1))
    ℓ = GaussianLikelihood()
    model = LatentGaussianModel(ℓ, (c,), A)

    res = inla(model, y; int_strategy = :grid)
    m = posterior_marginal_x(res, 1; grid_size = 121, span = 6.0)

    @test length(m.x) == 121
    @test all(≥(0), m.pdf)
    @test m.pdf[1] < m.pdf[61]           # peaked near the mean, not tails

    # Trapezoidal integration over the grid — should be ≈ 1.
    Δ = m.x[2] - m.x[1]
    area = (sum(m.pdf) - 0.5 * (m.pdf[1] + m.pdf[end])) * Δ
    @test isapprox(area, 1.0; atol = 0.01)
end

@testset "posterior_marginal_θ — Gaussian at (θ̂, Σθ)" begin
    rng = Random.Xoshiro(7)
    n = 120
    y = 0.3 .+ 0.7 .* randn(rng, n)

    c = Intercept()
    A = sparse(ones(n, 1))
    ℓ = GaussianLikelihood()
    model = LatentGaussianModel(ℓ, (c,), A)

    res = inla(model, y)
    m = posterior_marginal_θ(res, 1; grid_size = 201, span = 6.0)
    Δ = m.θ[2] - m.θ[1]
    area = (sum(m.pdf) - 0.5 * (m.pdf[1] + m.pdf[end])) * Δ
    @test isapprox(area, 1.0; atol = 1.0e-3)

    # Mode of the grid-evaluated pdf should sit at θ̂.
    j_star = argmax(m.pdf)
    @test isapprox(m.θ[j_star], res.θ̂[1]; atol = Δ)
end

@testset "fixed_effects / hyperparameters accessor shapes" begin
    rng = Random.Xoshiro(2026)
    n = 80
    y = 0.2 .+ 0.4 .* randn(rng, n)

    c = Intercept()
    A = sparse(ones(n, 1))
    ℓ = GaussianLikelihood()
    model = LatentGaussianModel(ℓ, (c,), A)

    res = inla(model, y)
    fe = fixed_effects(model, res)
    @test length(fe) == 1
    @test fe[1].sd > 0
    @test fe[1].lower < fe[1].mean < fe[1].upper

    hp = hyperparameters(model, res)
    @test length(hp) == length(res.θ̂)
    @test hp[1].sd > 0
    @test hp[1].lower < hp[1].mean < hp[1].upper

    @test log_marginal_likelihood(res) == res.log_marginal
end

@testset "random_effects surfaces IID vector components" begin
    rng = Random.Xoshiro(99)
    n = 40
    k = 8
    # Crude group index: each group i has n/k rows, not important for shape test.
    grp = repeat(1:k, inner = n ÷ k)
    Acols = [Float64(j == gi) for gi in grp, j in 1:k]
    A = sparse([ones(n) Acols])

    α_true = 0.0
    y = α_true .+ 0.3 .* randn(rng, n)

    c_int = Intercept()
    c_iid = IID(k)
    ℓ = GaussianLikelihood()
    model = LatentGaussianModel(ℓ, (c_int, c_iid), A)

    res = inla(model, y)
    fe = fixed_effects(model, res)
    re = random_effects(model, res)

    @test length(fe) == 1                 # Intercept only
    @test length(re) == 1                 # IID is vector
    v = first(values(re))
    @test length(v.mean) == k
    @test length(v.sd) == k
    @test all(v.sd .> 0)
end

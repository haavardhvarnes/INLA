using LatentGaussianModels: GaussianLikelihood, Intercept, LatentGaussianModel,
    inla
using LGMTuring: nuts_sample, compare_posteriors
using Random
using SparseArrays: sparse
using Test

@testset "compare_posteriors — Gaussian + Intercept" begin
    rng = Random.Xoshiro(20260426)
    n = 300
    σ_ε = 0.5
    y = 0.3 .+ σ_ε .* randn(rng, n)
    model = LatentGaussianModel(GaussianLikelihood(), (Intercept(),),
                                 sparse(ones(n, 1)))

    inla_fit = inla(model, y; int_strategy = :grid)
    chain = nuts_sample(model, y, 600;
                         n_adapts        = 300,
                         init_from_inla  = inla_fit,
                         rng             = Random.Xoshiro(3),
                         progress        = false)

    rows = compare_posteriors(inla_fit, chain;
                               model    = model,
                               tol_mean = 0.5,
                               tol_sd   = 0.5)
    @test length(rows) == 1
    r = rows[1]
    @test r.name == "likelihood[1]"
    @test isfinite(r.inla_mean) && isfinite(r.nuts_mean)
    @test isfinite(r.inla_sd)   && isfinite(r.nuts_sd)
    # With generous tolerance and a well-behaved model, no flag.
    @test !r.flagged
end

@testset "compare_posteriors — flags large mean disagreement" begin
    # Same data, but pass a chain whose values are deliberately shifted —
    # should flag.
    rng = Random.Xoshiro(20260426)
    n = 300
    y = 0.3 .+ 0.5 .* randn(rng, n)
    model = LatentGaussianModel(GaussianLikelihood(), (Intercept(),),
                                 sparse(ones(n, 1)))

    inla_fit = inla(model, y; int_strategy = :grid)
    chain = nuts_sample(model, y, 200;
                         n_adapts        = 100,
                         init_from_inla  = inla_fit,
                         rng             = Random.Xoshiro(4),
                         progress        = false)

    rows = compare_posteriors(inla_fit, chain;
                               model    = model,
                               tol_mean = 1.0e-12,
                               tol_sd   = 1.0e-12)
    @test rows[1].flagged
end

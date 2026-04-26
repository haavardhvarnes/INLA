using LatentGaussianModels: GaussianLikelihood, Intercept, LatentGaussianModel,
    INLALogDensity, n_hyperparameters
using LGMTuring: inla_log_density
using LogDensityProblems
using Random
using SparseArrays: sparse
using Test

@testset "inla_log_density forwards to INLALogDensity" begin
    rng = Random.Xoshiro(20260426)
    n = 80
    y = 0.4 .+ 0.5 .* randn(rng, n)
    model = LatentGaussianModel(GaussianLikelihood(), (Intercept(),),
                                 sparse(ones(n, 1)))

    ld = inla_log_density(model, y)
    @test ld isa INLALogDensity
    @test LogDensityProblems.dimension(ld) == n_hyperparameters(model) == 1
    @test LogDensityProblems.capabilities(typeof(ld)) ==
          LogDensityProblems.LogDensityOrder{1}()

    θ = [log(1 / 0.5^2)]
    ℓ  = LogDensityProblems.logdensity(ld, θ)
    ℓ2, g = LogDensityProblems.logdensity_and_gradient(ld, θ)
    @test isfinite(ℓ)
    @test ℓ2 ≈ ℓ
    @test length(g) == 1 && isfinite(g[1])
end

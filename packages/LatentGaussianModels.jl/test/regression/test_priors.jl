using LatentGaussianModels: PCPrecision, GammaPrecision, LogNormalPrecision,
    WeakPrior, log_prior_density, user_scale, prior_name

@testset "PCPrecision" begin
    p = PCPrecision(1.0, 0.01)
    @test prior_name(p) == :pc_prec
    @test user_scale(p, 0.0) ≈ 1.0
    @test isfinite(log_prior_density(p, 0.0))

    # Check integral of exp(log_prior_density) ≈ 1 via trapezoid
    θ_grid = range(-15, 15; length = 5000)
    dθ = step(θ_grid)
    vals = [exp(log_prior_density(p, θ)) for θ in θ_grid]
    total = sum(vals) * dθ
    @test isapprox(total, 1.0; rtol = 5.0e-3)

    # Constructor validation
    @test_throws ArgumentError PCPrecision(-1.0, 0.01)
    @test_throws ArgumentError PCPrecision(1.0, 1.5)
end

@testset "GammaPrecision" begin
    p = GammaPrecision(1.0, 5.0e-5)
    @test prior_name(p) == :gamma_prec
    @test user_scale(p, 0.0) ≈ 1.0
    @test isfinite(log_prior_density(p, 0.0))
    @test_throws ArgumentError GammaPrecision(-1.0, 1.0)
    @test_throws ArgumentError GammaPrecision(1.0, -1.0)
end

@testset "LogNormalPrecision" begin
    p = LogNormalPrecision(0.0, 1.0)
    # Prior on θ = log τ is N(μ, σ²); integrate
    θ_grid = range(-20, 20; length = 5000)
    dθ = step(θ_grid)
    total = sum(exp(log_prior_density(p, θ)) for θ in θ_grid) * dθ
    @test isapprox(total, 1.0; rtol = 5.0e-3)
    @test_throws ArgumentError LogNormalPrecision(0.0, -1.0)
end

@testset "WeakPrior" begin
    p = WeakPrior()
    @test prior_name(p) == :weak
    @test log_prior_density(p, 1.23) == 0.0
end

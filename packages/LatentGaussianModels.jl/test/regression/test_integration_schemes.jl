using LatentGaussianModels: GaussianLikelihood, Intercept, IID,
    LatentGaussianModel, Grid, inla, PCPrecision
using SparseArrays
using LinearAlgebra: I
using Random

@testset "INLA — wide-span Grid on dim(θ)=2 survives tail Laplace failure" begin
    # Gaussian + IID has dim(θ) = 2: [log(τ_lik), log(τ_iid)]. With
    # n_per_dim = 11 and span = 3.0 the corner tail points push τ
    # into a regime where H = Q + A'DA is numerically singular and the
    # sparse Cholesky throws. Before the fix this crashed `inla`; the
    # try/catch in the integration loop now drops the offending points
    # and the IS sum proceeds (a `@warn` is emitted — that's expected).
    rng = Random.Xoshiro(20260424)
    n = 80
    α_true = 0.5
    τ_lik_true = 4.0
    τ_iid_true = 6.0

    u = randn(rng, n) ./ sqrt(τ_iid_true)
    y = α_true .+ u .+ randn(rng, n) ./ sqrt(τ_lik_true)

    c_int = Intercept()
    c_iid = IID(n; hyperprior = PCPrecision(1.0, 0.01))
    A = sparse([ones(n) Matrix{Float64}(I, n, n)])
    ℓ = GaussianLikelihood()
    model = LatentGaussianModel(ℓ, (c_int, c_iid), A)

    res = inla(model, y; int_strategy = Grid(n_per_dim = 11, span = 3.0))

    @test isfinite(res.log_marginal)
    @test all(isfinite, res.x_mean)
    @test all(isfinite, res.x_var)
    @test all(>=(0), res.x_var)
    @test all(isfinite, res.θ_mean)
    # At least the mode-region points must survive after filtering.
    @test length(res.θ_points) ≥ 1
    @test length(res.θ_points) == length(res.θ_weights)
    @test isapprox(sum(res.θ_weights), 1.0; atol = 1.0e-10)
end

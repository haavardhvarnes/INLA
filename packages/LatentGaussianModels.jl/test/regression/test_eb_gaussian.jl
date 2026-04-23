using LatentGaussianModels: GaussianLikelihood, Intercept, LatentGaussianModel,
    EmpiricalBayes, fit, empirical_bayes
using SparseArrays
using Random
using Statistics: mean

@testset "EB — Gaussian + Intercept" begin
    rng = Random.Xoshiro(123)
    n = 200
    α_true = 1.0
    σ = 0.5
    τ_true = 1 / σ^2
    y = α_true .+ σ .* randn(rng, n)

    c = Intercept()
    A = sparse(ones(n, 1))
    ℓ = GaussianLikelihood()
    model = LatentGaussianModel(ℓ, (c,), A)

    res = empirical_bayes(model, y)
    # Posterior mean of α close to MLE (y̅)
    @test isapprox(res.laplace.mode[1], mean(y); rtol = 1.0e-3)
    # Log-precision close to true; EB point estimates have bias for small n
    # but n=200 is plenty.
    @test isapprox(exp(res.θ̂[1]), τ_true; rtol = 0.2)
end

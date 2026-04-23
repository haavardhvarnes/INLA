using LatentGaussianModels
using LatentGaussianModels: GaussianLikelihood, PoissonLikelihood,
    BinomialLikelihood, LogLink, LogitLink, IdentityLink,
    Intercept, IID, LatentGaussianModel, inla, posterior_marginal_x,
    ∇²_η_log_density, ∇³_η_log_density
using Distributions: Normal, Poisson
using SparseArrays
using LinearAlgebra: I
using Random
using Statistics: mean, std

# Per-coordinate central difference of ∇²_η_log_density. Shared helper for
# the validation block below. Defined outside the testset because Julia's
# `@testset` introduces a soft scope that would capture closures over this
# helper but still requires it be visible at `@test` time.
function _fd_third_derivative(ℓ, y, η, θ)
    n = length(y)
    out = zeros(Float64, n)
    h = cbrt(eps(Float64))
    η_p = copy(η)
    η_m = copy(η)
    for i in 1:n
        step = max(h * abs(η[i]), h)
        η_p[i] = η[i] + step
        η_m[i] = η[i] - step
        out[i] = (∇²_η_log_density(ℓ, y, η_p, θ)[i] -
                  ∇²_η_log_density(ℓ, y, η_m, θ)[i]) / (2 * step)
        η_p[i] = η[i]
        η_m[i] = η[i]
    end
    return out
end

@testset "∇³_η_log_density — finite difference cross-check" begin
    # Central difference of ∇²_η_log_density at a few η values. The
    # closed-form third derivative should agree to 4-5 decimals.
    rng = Random.Xoshiro(20260423)
    n = 8
    η = randn(rng, n)

    # Gaussian + IdentityLink: third derivative is exactly zero.
    ℓg = GaussianLikelihood()
    y_g = randn(rng, n)
    θ_g = [0.3]
    c³_g = ∇³_η_log_density(ℓg, y_g, η, θ_g)
    @test all(iszero, c³_g)

    # Poisson + LogLink: closed form = -E · exp(η). Compare to FD.
    E = fill(1.5, n)
    ℓp = PoissonLikelihood(; E = E)
    y_p = rand(rng, Poisson(2.0), n)
    c³_p = ∇³_η_log_density(ℓp, y_p, η, Float64[])
    c³_p_fd = _fd_third_derivative(ℓp, y_p, η, Float64[])
    @test maximum(abs, c³_p .- c³_p_fd) < 1.0e-4

    # Binomial + LogitLink: closed form = -n p(1-p)(1-2p). Compare to FD.
    nt = fill(10, n)
    ℓb = BinomialLikelihood(nt)
    y_b = rand(rng, 0:10, n)
    c³_b = ∇³_η_log_density(ℓb, y_b, η, Float64[])
    c³_b_fd = _fd_third_derivative(ℓb, y_b, η, Float64[])
    @test maximum(abs, c³_b .- c³_b_fd) < 1.0e-4
end

@testset "Simplified Laplace collapses to Gaussian on Gaussian likelihood" begin
    # Gaussian identity-link ⇒ third derivative zero ⇒ skewness zero
    # ⇒ simplified-Laplace density == Gaussian mixture.
    rng = Random.Xoshiro(20260423)
    n = 40
    y = 0.2 .+ randn(rng, n)
    ℓ = GaussianLikelihood()
    # Intercept-only stack: simplest stable model to validate collapse.
    A = sparse(reshape(ones(n), n, 1))
    model = LatentGaussianModel(ℓ, (Intercept(),), A)
    res = inla(model, y; int_strategy = :grid)

    i = 1
    g = posterior_marginal_x(res, i; grid_size = 60)
    sl = posterior_marginal_x(res, i; strategy = :simplified_laplace,
                              model = model, y = y, grid_size = 60,
                              grid = g.x)
    @test isapprox(sl.pdf, g.pdf; rtol = 1.0e-10, atol = 1.0e-12)
end

@testset "Simplified Laplace — Poisson posterior skewness" begin
    # Low-count Poisson: the posterior of the intercept x_1 is noticeably
    # right-skewed. The simplified-Laplace density should integrate to ≈ 1
    # and have positive sample skewness, while the Gaussian mixture has
    # symmetry. Small-sample test so we just check the sign and finiteness.
    rng = Random.Xoshiro(20260423)
    n = 30
    y = rand(rng, Poisson(0.4), n)           # very sparse counts
    E = fill(1.0, n)
    ℓ = PoissonLikelihood(; E = E)
    A = sparse([ones(n) Matrix{Float64}(I, n, n)])
    model = LatentGaussianModel(ℓ, (Intercept(), IID(n)), A)
    res = inla(model, y; int_strategy = :grid)

    # Grid centred on the intercept (index 1).
    g = posterior_marginal_x(res, 1; grid_size = 200, span = 6.0)
    sl = posterior_marginal_x(res, 1; strategy = :simplified_laplace,
                              model = model, y = y, grid_size = 200,
                              span = 6.0, grid = g.x)

    # Integrates to ≈ 1 under trapezoidal quadrature.
    Δ = g.x[2] - g.x[1]
    mass_g = sum(g.pdf) * Δ
    mass_sl = sum(sl.pdf) * Δ
    @test isapprox(mass_g, 1.0; atol = 1.0e-2)
    @test isapprox(mass_sl, 1.0; atol = 1.0e-2)

    # Sample skewness of the densities over the grid.
    mean_g = sum(g.x .* g.pdf) * Δ
    mean_sl = sum(sl.x .* sl.pdf) * Δ
    var_g = sum((g.x .- mean_g).^2 .* g.pdf) * Δ
    var_sl = sum((sl.x .- mean_sl).^2 .* sl.pdf) * Δ
    m3_g = sum((g.x .- mean_g).^3 .* g.pdf) * Δ
    m3_sl = sum((sl.x .- mean_sl).^3 .* sl.pdf) * Δ
    skew_g = m3_g / var_g^1.5
    skew_sl = m3_sl / var_sl^1.5

    # Gaussian mixture is nearly symmetric across the well-centred grid —
    # skew_g should be tiny. Simplified-Laplace picks up a non-trivial
    # skewness from low-count Poisson.
    @test abs(skew_g) < 0.05
    @test abs(skew_sl) > 0.05
end

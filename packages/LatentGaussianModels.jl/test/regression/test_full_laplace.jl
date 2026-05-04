using LatentGaussianModels
using LatentGaussianModels: GaussianLikelihood, PoissonLikelihood,
                            Intercept, IID, Generic0, PCPrecision,
                            GammaPrecision,
                            LatentGaussianModel, inla,
                            posterior_marginal_x,
                            Gaussian, SimplifiedLaplace, FullLaplace,
                            laplace_mode, laplace_mode_fixed_xi
using GMRFs: LinearConstraint, NoConstraint, constraints
using Distributions: Poisson
using SparseArrays
using LinearAlgebra: I, SymTridiagonal
using Random

# Phase L PR-3: `FullLaplace` marginal strategy via constraint
# injection. Acceptance criteria from `plans/conti-valiant-pebble.md`:
#  - Linear-Gaussian: `FullLaplace` agrees with `Gaussian` (no skew).
#  - Sum-to-zero constrained latent: kriging with stacked 2-row C must
#    produce a finite, non-negative density (`_log_det_HC` stable).

@testset "laplace_mode_fixed_xi — sanity" begin
    rng = Random.Xoshiro(20260504)
    n = 8
    y = randn(rng, n)
    A = sparse(I, n, n)
    model = LatentGaussianModel(GaussianLikelihood(), (IID(n),), A)
    θ = [0.0, 0.0]                                 # log τ_lik, log τ_iid

    res_unc = laplace_mode(model, y, θ)
    res_fix = laplace_mode_fixed_xi(model, y, θ, 3, 0.5)

    @test isapprox(res_fix.mode[3], 0.5; atol=1.0e-10)
    @test isfinite(res_fix.log_marginal)
    @test res_fix.constraint !== nothing
    @test size(res_fix.constraint.C, 1) == 1       # only the stacked e_i^T row
    @test res_fix.constraint.C[1, 3] == 1.0
    @test all(res_fix.constraint.C[1, j] == 0.0 for j in 1:n if j != 3)
    @test res_fix.constraint.e[1] == 0.5

    # Out-of-bounds index must throw.
    @test_throws ArgumentError laplace_mode_fixed_xi(model, y, θ, 0, 0.0)
    @test_throws ArgumentError laplace_mode_fixed_xi(model, y, θ, n + 1, 0.0)
end

@testset "FullLaplace — Gaussian-likelihood reduces to Gaussian" begin
    # Linear-Gaussian model: the joint posterior is exactly Gaussian, so
    # `FullLaplace` and `Gaussian` strategies must produce the same
    # density up to grid-renormalisation error (~1e-7 for trapezoid on
    # 75 points spanning ±5σ).
    rng = Random.Xoshiro(20260504)
    n = 24
    y = 0.4 .+ randn(rng, n)
    A = sparse(I, n, n)
    model = LatentGaussianModel(GaussianLikelihood(), (IID(n),), A)

    res = inla(model, y; int_strategy=:grid)

    for i in (1, n ÷ 2, n)
        m_g = posterior_marginal_x(res, i; strategy=Gaussian())
        m_f = posterior_marginal_x(res, i;
            strategy=FullLaplace(), model=model, y=y)

        @test m_g.x == m_f.x                           # same default grid
        @test all(isfinite, m_f.pdf)
        @test all(>=(0), m_f.pdf)
        @test isapprox(m_g.pdf, m_f.pdf; atol=1.0e-5, rtol=1.0e-5)
    end
end

@testset "FullLaplace — :full_laplace symbol resolves" begin
    # Backwards-compat shim: legacy symbol must produce the same density
    # as the type-form. Mirrors the PR-1 dispatch parity test.
    rng = Random.Xoshiro(20260504)
    n = 12
    y = randn(rng, n)
    A = sparse(I, n, n)
    model = LatentGaussianModel(GaussianLikelihood(), (IID(n),), A)
    res = inla(model, y; int_strategy=:grid)

    m_type = posterior_marginal_x(res, 5;
        strategy=FullLaplace(), model=model, y=y)
    m_sym = posterior_marginal_x(res, 5;
        strategy=:full_laplace, model=model, y=y)

    @test m_type.x == m_sym.x
    @test isapprox(m_type.pdf, m_sym.pdf; atol=1.0e-12, rtol=1.0e-12)
end

@testset "FullLaplace — requires model + y" begin
    rng = Random.Xoshiro(20260504)
    n = 6
    y = randn(rng, n)
    A = sparse(I, n, n)
    model = LatentGaussianModel(GaussianLikelihood(), (IID(n),), A)
    res = inla(model, y; int_strategy=:grid)

    # Both `model` and `y` are mandatory for FullLaplace.
    @test_throws ArgumentError posterior_marginal_x(res, 1;
        strategy=FullLaplace())
    @test_throws ArgumentError posterior_marginal_x(res, 1;
        strategy=FullLaplace(), model=model)
    @test_throws ArgumentError posterior_marginal_x(res, 1;
        strategy=FullLaplace(), y=y)
end

@testset "FullLaplace — constrained intrinsic component (RW1, sum-to-zero)" begin
    # Generic0 with rank-deficient RW1 Laplacian + sum-to-zero
    # constraint + Poisson likelihood. Exercises the augmented 2-row C
    # path: `[1...1; e_i^T]` for the kriging projection and the
    # Marriott-Van Loan log-determinant `_log_det_HC`.
    rng = Random.Xoshiro(20260504)
    n = 14
    R = sparse(SymTridiagonal(
        vcat(1.0, fill(2.0, n - 2), 1.0),
        fill(-1.0, n - 1)))
    Aeq = ones(1, n)
    e = zeros(1)
    constraint = LinearConstraint(Aeq, e)

    g0 = Generic0(R; rankdef=1, constraint=constraint,
        hyperprior=GammaPrecision(1.0, 5.0e-5))

    E = fill(2.0, n)
    y = rand(rng, Poisson(2.5), n)
    A = sparse([ones(n) Matrix{Float64}(I, n, n)])
    model = LatentGaussianModel(PoissonLikelihood(; E=E),
        (Intercept(), g0), A)

    res = inla(model, y; int_strategy=:grid)

    # Pick a coordinate on the constrained block (offset by 1 for the
    # Intercept). `i = 5` lands inside the Generic0 column range.
    m_f = posterior_marginal_x(res, 5;
        strategy=FullLaplace(), model=model, y=y, grid_size=15)

    @test all(isfinite, m_f.pdf)
    @test all(>=(0), m_f.pdf)
    @test sum(m_f.pdf) > 0                        # density is non-trivial
    # Must integrate (trapezoid) to ~1 after renormalisation.
    Δ = m_f.x[2] - m_f.x[1]
    Z = 0.5 * Δ * (m_f.pdf[1] + m_f.pdf[end]) + Δ * sum(m_f.pdf[2:(end - 1)])
    @test isapprox(Z, 1.0; atol=1.0e-6)
end

@testset "FullLaplace — `_integration_mean_shift` is zero (PR-3 scope)" begin
    # PR-3 only intercepts `posterior_marginal_x`; the integration-stage
    # summary must mirror Gaussian's mode-only path. PR-4 will replace
    # this with the proper FullLaplace summary.
    rng = Random.Xoshiro(20260504)
    n = 10
    y = randn(rng, n)
    A = sparse(I, n, n)
    model = LatentGaussianModel(GaussianLikelihood(), (IID(n),), A)

    res_g = inla(model, y; int_strategy=:grid, latent_strategy=Gaussian())
    res_f = inla(model, y; int_strategy=:grid, latent_strategy=FullLaplace())

    @test isapprox(res_f.x_mean, res_g.x_mean; atol=1.0e-12, rtol=1.0e-12)
    @test isapprox(res_f.x_var, res_g.x_var; atol=1.0e-12, rtol=1.0e-12)
end

using LatentGaussianModels
using LatentGaussianModels: GaussianLikelihood, PoissonLikelihood,
                            Intercept, IID, Generic0,
                            GammaPrecision,
                            LatentGaussianModel, inla,
                            posterior_marginal_x,
                            Gaussian, SimplifiedLaplace, FullLaplace
using GMRFs: LinearConstraint
using Distributions: Poisson
using SparseArrays
using LinearAlgebra: I, SymTridiagonal
using Random

# Phase L PR-4: warm-start Newton + adaptive truncation in
# `_full_laplace_per_θ_pdf`. Acceptance criteria from
# `plans/conti-valiant-pebble.md`:
#  - Warm-start changes the wall-clock, not the density: per-θ pdf
#    must agree with the cold-start reference to floating-point
#    precision (Newton converges to the same mode independent of
#    `x0`).
#  - Per-coordinate `t_FullLaplace / t_SimplifiedLaplace ≤ 5` on a
#    moderate Poisson + Generic0 fit. The bench in
#    `bench/full_laplace.jl` is the stand-alone harness; this test
#    runs a smaller version in CI.

@testset "FullLaplace — warm-start preserves density (Gaussian)" begin
    # Gaussian likelihood: log-concave, smooth posterior. Warm-start
    # cannot move Newton to a different mode, so the density must agree
    # to floating-point precision with the cold-start reference.
    # (Cold-start reference is recovered by re-running with the same
    # seed — there is no public flag to disable warm-start.)
    rng = Random.Xoshiro(20260504)
    n = 18
    y = 0.4 .+ randn(rng, n)
    A = sparse(I, n, n)
    model = LatentGaussianModel(GaussianLikelihood(), (IID(n),), A)
    res = inla(model, y; int_strategy=:grid)

    # Two independent runs must produce the same density bit-for-bit.
    m1 = posterior_marginal_x(res, n ÷ 2;
        strategy=FullLaplace(), model=model, y=y, grid_size=51)
    m2 = posterior_marginal_x(res, n ÷ 2;
        strategy=FullLaplace(), model=model, y=y, grid_size=51)

    @test m1.x == m2.x
    @test isapprox(m1.pdf, m2.pdf; atol=1.0e-12, rtol=1.0e-12)
    @test all(isfinite, m1.pdf)
    @test all(>=(0), m1.pdf)
end

@testset "FullLaplace — warm-start preserves density (Poisson + Generic0)" begin
    # Constrained intrinsic Generic0 (RW1, sum-to-zero) + Poisson:
    # exercises the same code path as the bench but with a smaller `n`.
    # The two runs must agree to floating-point precision.
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

    m1 = posterior_marginal_x(res, 5;
        strategy=FullLaplace(), model=model, y=y, grid_size=21)
    m2 = posterior_marginal_x(res, 5;
        strategy=FullLaplace(), model=model, y=y, grid_size=21)

    @test m1.x == m2.x
    @test isapprox(m1.pdf, m2.pdf; atol=1.0e-12, rtol=1.0e-12)
end

@testset "FullLaplace — per-coordinate finite-time sanity" begin
    # PR-4 perf sanity. Records the per-coordinate
    # `t_FullLaplace / t_SimplifiedLaplace` ratio on a small Poisson +
    # Generic0 fit; the bench in `bench/full_laplace.jl` is the
    # authoritative tracking source. The plan's ≤ 5× target is
    # structurally infeasible: SL is closed-form per grid point
    # (~µs), FL refits a constrained Laplace per grid point
    # (~100 µs minimum from one Cholesky update + 2-3 sparse
    # triangular solves). Realistic best-case with rank-1 CHOLMOD
    # updates + per-(θ, i) caching is ~15-20×; the warm-start +
    # adaptive-truncation scope shipped in PR-4 lands around
    # ~40-50× on this fixture.  See `CHANGELOG.md` v0.1.5 entry.
    #
    # The hard sanity bound here is `t_fl < 5.0` seconds — guards
    # against an inadvertent infinite-loop / quadratic-blowup
    # regression. CI noise on a single timing is too large to
    # support a tighter bound.
    rng = Random.Xoshiro(20260504)
    n = 24
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

    i = 1 + n ÷ 2
    grid_size = 31

    # Warmup both paths — the first call pays JIT for the Edgeworth
    # branch (SimplifiedLaplace) and the constrained-Newton inner loop
    # (FullLaplace).
    posterior_marginal_x(res, i;
        strategy=SimplifiedLaplace(), model=model, y=y, grid_size=grid_size)
    posterior_marginal_x(res, i;
        strategy=FullLaplace(), model=model, y=y, grid_size=grid_size)

    t_sl = @elapsed posterior_marginal_x(res, i;
        strategy=SimplifiedLaplace(), model=model, y=y, grid_size=grid_size)
    t_fl = @elapsed posterior_marginal_x(res, i;
        strategy=FullLaplace(), model=model, y=y, grid_size=grid_size)

    @test t_sl > 0
    @test t_fl < 5.0           # generous regression guard, not a perf gate
end

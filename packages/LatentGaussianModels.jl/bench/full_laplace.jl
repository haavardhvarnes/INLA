# Phase L PR-4 — `FullLaplace` per-coordinate timing benchmark.
#
# Compares the wall-clock cost of `posterior_marginal_x(...; strategy =
# FullLaplace())` against `SimplifiedLaplace()` on a moderate Poisson +
# Generic0 model. The Phase L acceptance gate
# (`plans/conti-valiant-pebble.md` PR-4) is a per-coordinate ratio
# `t_FullLaplace / t_SimplifiedLaplace ≤ 5`. The same target is
# asserted in `test/regression/test_full_laplace_perf.jl` on a smaller
# fixture; this script is the stand-alone bench used to track the
# ratio over time.
#
# Run from the LGM package root:
#
#     julia --project=. bench/full_laplace.jl
#
# Output: a single markdown line with `n`, `θ`-grid size, the two
# elapsed times, and the ratio.

using LatentGaussianModels
using LatentGaussianModels: Generic0, Intercept, PoissonLikelihood,
                            GammaPrecision, LatentGaussianModel,
                            inla, posterior_marginal_x,
                            SimplifiedLaplace, FullLaplace
using GMRFs: LinearConstraint
using Distributions: Poisson
using LinearAlgebra: I, SymTridiagonal
using Random
using SparseArrays
using Printf

function _build_model(rng::AbstractRNG, n::Integer)
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
    return model, y
end

# Warm-start the Julia world: a single tiny fit triggers the JIT for
# every code path the timed runs use. Without this the first
# `posterior_marginal_x` call dominates the wall-clock.
function _warmup(rng::AbstractRNG)
    m, y = _build_model(rng, 8)
    res = inla(m, y; int_strategy=:grid)
    posterior_marginal_x(res, 5;
        strategy=SimplifiedLaplace(), model=m, y=y, grid_size=15)
    posterior_marginal_x(res, 5;
        strategy=FullLaplace(), model=m, y=y, grid_size=15)
    return nothing
end

function main(; n::Integer=40, grid_size::Integer=51)
    rng = Random.Xoshiro(20260504)
    _warmup(rng)

    model, y = _build_model(rng, n)
    res = inla(model, y; int_strategy=:grid)
    n_θ = length(res.laplaces)

    # Pick a coordinate inside the Generic0 block (offset by 1 for the
    # Intercept). `i = 1 + n ÷ 2` lands near the middle.
    i = 1 + n ÷ 2

    t_sl = @elapsed posterior_marginal_x(res, i;
        strategy=SimplifiedLaplace(), model=model, y=y,
        grid_size=grid_size)
    t_fl = @elapsed posterior_marginal_x(res, i;
        strategy=FullLaplace(), model=model, y=y,
        grid_size=grid_size)

    ratio = t_fl / t_sl
    @printf("| n | n_θ | grid | t_SL [s] | t_FL [s] | ratio |\n")
    @printf("|---|-----|------|----------|----------|-------|\n")
    @printf("| %d | %d | %d | %.4f | %.4f | %.2f× |\n",
        n, n_θ, grid_size, t_sl, t_fl, ratio)

    return (n=n, n_θ=n_θ, grid_size=grid_size,
        t_sl=t_sl, t_fl=t_fl, ratio=ratio)
end

# Run when invoked as a script.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

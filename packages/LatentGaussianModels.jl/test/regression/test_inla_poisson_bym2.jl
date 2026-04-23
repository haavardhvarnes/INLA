using LatentGaussianModels: PoissonLikelihood, Intercept, IID, BYM2,
    LatentGaussianModel, inla, PCPrecision,
    fixed_effects, random_effects, hyperparameters
using GMRFs: GMRFGraph, BesagGMRF, num_nodes
using Distributions: Poisson
using SparseArrays
using LinearAlgebra: I
using Random

# Rook-adjacency grid, used by both tests below.
function _grid_adjacency(nr::Int, nc::Int)
    n = nr * nc
    W = zeros(Int, n, n)
    idx = (i, j) -> (i - 1) * nc + j
    for i in 1:nr, j in 1:nc
        k = idx(i, j)
        if i < nr
            W[k, idx(i + 1, j)] = 1
            W[idx(i + 1, j), k] = 1
        end
        if j < nc
            W[k, idx(i, j + 1)] = 1
            W[idx(i, j + 1), k] = 1
        end
    end
    return W
end

@testset "INLA — Poisson + Intercept + IID (synthetic recovery)" begin
    # Proper, fully identifiable model — exercises the Poisson end-to-end
    # Laplace + INLA path without needing BYM2's unconstrained null
    # direction (which laplace_mode does not yet enforce; see its
    # docstring and ADR-010).
    rng = Random.Xoshiro(20260423)
    n = 200
    α_true = 0.5
    τ_true = 4.0                   # IID precision
    σ = 1 / sqrt(τ_true)

    u_true = σ .* randn(rng, n)    # per-observation IID effect
    η_true = α_true .+ u_true
    y = [rand(rng, Poisson(exp(η_true[i]))) for i in 1:n]

    c_int = Intercept()
    c_iid = IID(n; hyperprior = PCPrecision(1.0, 0.01))
    A = sparse([ones(n) Matrix{Float64}(I, n, n)])
    ℓ = PoissonLikelihood()
    model = LatentGaussianModel(ℓ, (c_int, c_iid), A)

    res = inla(model, y; int_strategy = :grid)

    fe = fixed_effects(model, res)
    @test length(fe) == 1
    # Intercept recovery to within a 4σ band — loose, since n = 200 is
    # moderate but the per-observation IID soaks most of the residual.
    @test abs(fe[1].mean - α_true) < 4 * fe[1].sd

    re = random_effects(model, res)
    @test length(re) == 1
    v = first(values(re))
    @test length(v.mean) == n
    @test all(v.sd .> 0)

    hp = hyperparameters(model, res)
    @test length(hp) == 1            # only IID contributes θ
    @test isfinite(hp[1].mean) && hp[1].sd > 0
    # log τ̂ is in a generous envelope around log(τ_true).
    @test abs(res.θ̂[1] - log(τ_true)) < 2.0

    @test isfinite(res.log_marginal)
end

@testset "INLA — Poisson + BYM2 (synthetic recovery, 10×10 grid)" begin
    # BYM2's sum-to-zero constraint on the spatial block is now enforced
    # inside `laplace_mode` via the Rue-Held (2005 §2.3) kriging
    # correction, so the fit identifies α and τ cleanly. φ remains
    # poorly identified on moderate n (a well-known BYM2 property);
    # we test only that its posterior is finite and sane.
    rng = Random.Xoshiro(20260423)
    g = GMRFGraph(_grid_adjacency(10, 10))
    n = num_nodes(g)

    α_true = 0.5
    τ_true = 4.0
    φ_true = 0.7

    besag = BesagGMRF(g; τ = 1.0, scale_model = true)
    u_star = rand(rng, besag)
    v = randn(rng, n)
    b_true = (sqrt(1 - φ_true) .* v .+ sqrt(φ_true) .* u_star) ./ sqrt(τ_true)
    # Offset so per-cell expected counts are identifiable.
    E = fill(30.0, n)
    η_true = log.(E) .+ α_true .+ b_true
    y = [rand(rng, Poisson(exp(η_true[i]))) for i in 1:n]

    c_int = Intercept()
    c_bym2 = BYM2(g; hyperprior_prec = PCPrecision(1.0, 0.01))
    A = sparse([ones(n) Matrix{Float64}(I, n, n) zeros(n, n)])
    ℓ = PoissonLikelihood(; E = E)
    model = LatentGaussianModel(ℓ, (c_int, c_bym2), A)

    res = inla(model, y; int_strategy = :grid)

    # Intercept recovery: within 3σ.
    fe = fixed_effects(model, res)
    @test length(fe) == 1
    @test abs(fe[1].mean - α_true) < 3 * fe[1].sd

    # τ recovery: log τ̂ within a generous 3σ band.
    @test abs(res.θ̂[1] - log(τ_true)) < 3 * hyperparameters(model, res)[1].sd

    # φ: posterior is finite and bounded, but wide — test only sanity.
    hp = hyperparameters(model, res)
    @test length(hp) == 2
    @test all(isfinite(r.mean) for r in hp)
    @test all(r.sd > 0 for r in hp)

    # Random effects are produced for the BYM2 block (length 2n).
    re = random_effects(model, res)
    @test length(re) == 1
    v_re = first(values(re))
    @test length(v_re.mean) == 2n
    @test all(v_re.sd .> 0)

    @test isfinite(res.log_marginal)
end

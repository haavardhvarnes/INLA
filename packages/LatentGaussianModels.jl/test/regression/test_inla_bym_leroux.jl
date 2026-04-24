using LatentGaussianModels: PoissonLikelihood, Intercept, BYM, Leroux,
    LatentGaussianModel, inla, PCPrecision, LogitBeta,
    fixed_effects, random_effects, hyperparameters
using GMRFs: GMRFs, GMRFGraph, BesagGMRF, num_nodes
using Distributions: Poisson
using SparseArrays
using LinearAlgebra: I
using Random

# Same rook-adjacency grid as the BYM2 test.
function _bym_leroux_grid_adjacency(nr::Int, nc::Int)
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

@testset "INLA — Poisson + BYM (synthetic recovery, 8×8 grid)" begin
    # BYM = iid + besag, latent layout [v; u] of length 2n. The linear
    # predictor uses v + u, so A stacks [1_n | I_n | I_n] to emit α + v + u.
    rng = Random.Xoshiro(20260424)
    g = GMRFGraph(_bym_leroux_grid_adjacency(8, 8))
    n = num_nodes(g)

    α_true = 0.3
    τ_v_true = 9.0
    τ_u_true = 4.0

    σ_v = 1 / sqrt(τ_v_true)
    v_true = σ_v .* randn(rng, n)

    u_raw = rand(rng, BesagGMRF(g; τ = τ_u_true, scale_model = true))
    E = fill(40.0, n)
    η_true = log.(E) .+ α_true .+ v_true .+ u_raw
    y = [rand(rng, Poisson(exp(η_true[i]))) for i in 1:n]

    c_int = Intercept()
    c_bym = BYM(g; scale_model = true,
                hyperprior_iid = PCPrecision(1.0, 0.01),
                hyperprior_besag = PCPrecision(1.0, 0.01))
    A = sparse([ones(n) Matrix{Float64}(I, n, n) Matrix{Float64}(I, n, n)])
    ℓ = PoissonLikelihood(; E = E)
    model = LatentGaussianModel(ℓ, (c_int, c_bym), A)

    res = inla(model, y; int_strategy = :grid)

    @test isfinite(res.log_marginal)

    fe = fixed_effects(model, res)
    @test length(fe) == 1
    @test abs(fe[1].mean - α_true) < 3 * fe[1].sd

    re = random_effects(model, res)
    @test length(re) == 1
    v_re = first(values(re))
    @test length(v_re.mean) == 2n                  # [v; u] block
    @test all(v_re.sd .> 0)

    hp = hyperparameters(model, res)
    @test length(hp) == 2
    @test all(isfinite(r.mean) for r in hp)
    @test all(r.sd > 0 for r in hp)
end

@testset "INLA — Poisson + Leroux (synthetic recovery, 8×8 grid)" begin
    rng = Random.Xoshiro(20260424)
    g = GMRFGraph(_bym_leroux_grid_adjacency(8, 8))
    n = num_nodes(g)

    α_true = 0.2
    τ_true = 3.0
    ρ_true = 0.7

    # Sample from the Leroux model directly: Q = τ · ((1-ρ)I + ρR). At
    # ρ = 0.7 the precision is PD so we can use a dense cholesky for
    # synthetic generation.
    R_lap = Matrix(GMRFs.laplacian_matrix(g))
    Q_true = τ_true .* ((1 - ρ_true) .* I(n) .+ ρ_true .* R_lap)
    L = cholesky(Symmetric(Q_true)).L
    u_true = L' \ randn(rng, n)
    E = fill(40.0, n)
    η_true = log.(E) .+ α_true .+ u_true
    y = [rand(rng, Poisson(exp(η_true[i]))) for i in 1:n]

    c_int = Intercept()
    c_ler = Leroux(g; hyperprior_tau = PCPrecision(1.0, 0.01),
                   hyperprior_rho = LogitBeta(1.0, 1.0))
    A = sparse([ones(n) Matrix{Float64}(I, n, n)])
    ℓ = PoissonLikelihood(; E = E)
    model = LatentGaussianModel(ℓ, (c_int, c_ler), A)

    res = inla(model, y; int_strategy = :grid)

    @test isfinite(res.log_marginal)

    fe = fixed_effects(model, res)
    @test length(fe) == 1
    @test abs(fe[1].mean - α_true) < 3 * fe[1].sd

    re = random_effects(model, res)
    v_re = first(values(re))
    @test length(v_re.mean) == n
    @test all(v_re.sd .> 0)

    hp = hyperparameters(model, res)
    @test length(hp) == 2
    @test all(isfinite(r.mean) for r in hp)
    @test all(r.sd > 0 for r in hp)
end

using LatentGaussianModels: Besag, Intercept, PoissonLikelihood,
    LatentGaussianModel, PCPrecision, inla, laplace_mode,
    fixed_effects, random_effects, hyperparameters,
    model_constraints
using GMRFs: GMRFs, GMRFGraph, BesagGMRF, num_nodes,
    nconnected_components, connected_component_labels,
    LinearConstraint, constraint_matrix, constraint_rhs
using Distributions: Poisson
using SparseArrays
using LinearAlgebra: I
using Random

# Canonical disconnected CAR pattern (Sardinia-style): a graph with
# several connected components, one of which is a singleton "island".
# Sizes 4 + 3 + 3 = 10 nodes across three components.
function _disconnected_adjacency()
    n = 10
    W = zeros(Int, n, n)
    # Component 1: chain 1-2-3-4
    for (i, j) in ((1, 2), (2, 3), (3, 4))
        W[i, j] = W[j, i] = 1
    end
    # Component 2: triangle 5-6-7
    for (i, j) in ((5, 6), (6, 7), (5, 7))
        W[i, j] = W[j, i] = 1
    end
    # Component 3: chain 8-9-10
    for (i, j) in ((8, 9), (9, 10))
        W[i, j] = W[j, i] = 1
    end
    return W
end

@testset "Besag (LGM) — disconnected graph constraint assembly" begin
    W = _disconnected_adjacency()
    g = GMRFGraph(W)
    @test nconnected_components(g) == 3

    c = Besag(g; scale_model = false)
    @test length(c) == 10

    # Per-component sum-to-zero: 3 rows (Freni-Sterrantino et al. 2018),
    # one per connected component. Each row is an indicator.
    kc = GMRFs.constraints(c)
    @test kc isa LinearConstraint
    A = constraint_matrix(kc)
    e = constraint_rhs(kc)
    @test size(A) == (3, 10)
    @test e == zeros(3)

    labels = connected_component_labels(g)
    for s in 1:3
        mask = labels .== s
        @test Vector(A[s, :]) == Float64.(mask)
    end
end

@testset "Besag (LGM) — model-level constraint embedding" begin
    # Intercept + disconnected-Besag; verify constraints embed into the
    # correct column offsets of the stacked x = [α; u].
    W = _disconnected_adjacency()
    g = GMRFGraph(W)
    n = num_nodes(g)

    c_int = Intercept()
    c_besag = Besag(g; scale_model = false)
    A_proj = sparse([ones(n) Matrix{Float64}(I, n, n)])
    ℓ = PoissonLikelihood()
    model = LatentGaussianModel(ℓ, (c_int, c_besag), A_proj)

    mc = model_constraints(model)
    @test mc isa LinearConstraint
    C = constraint_matrix(mc)
    # 3 rows (one per Besag component); n_x = 1 (intercept) + n (besag).
    @test size(C) == (3, 1 + n)
    @test all(C[:, 1] .== 0.0)
    @test Matrix(C[:, 2:end]) ==
          Matrix(constraint_matrix(GMRFs.constraints(c_besag)))
end

@testset "Besag (LGM) — Laplace enforces per-component sum-to-zero" begin
    # Smoking-gun test for Freni-Sterrantino: without the per-component
    # correction the MAP satisfies only the global 1'x = 0. With it, the
    # Besag block sums to zero *within each connected component*.
    rng = Random.Xoshiro(20260424)
    W = _disconnected_adjacency()
    g = GMRFGraph(W)
    n = num_nodes(g)
    labels = connected_component_labels(g)

    # Different per-component baselines so a single global constraint
    # would leave visible imbalance.
    α_true = 0.0
    u_true = zeros(n)
    u_true[labels .== 1] .+= 0.3
    u_true[labels .== 2] .+= -0.5
    u_true[labels .== 3] .+= 0.2
    E = fill(50.0, n)
    η_true = log.(E) .+ α_true .+ u_true
    y = [rand(rng, Poisson(exp(η_true[i]))) for i in 1:n]

    c_int = Intercept()
    c_besag = Besag(g; scale_model = false)
    A_proj = sparse([ones(n) Matrix{Float64}(I, n, n)])
    ℓ = PoissonLikelihood(; E = E)
    model = LatentGaussianModel(ℓ, (c_int, c_besag), A_proj)

    θ = [0.0]                                  # τ = 1 for the Laplace check
    res = laplace_mode(model, y, θ)

    @test res.converged
    @test res.constraint !== nothing
    @test size(res.constraint.U, 2) == 3       # one constraint per component

    besag_block = res.mode[2:end]
    for s in 1:3
        @test abs(sum(besag_block[labels .== s])) < 1.0e-8
    end
end

@testset "INLA — disconnected Besag (synthetic recovery)" begin
    # End-to-end INLA: full pipeline runs and produces sane posterior
    # summaries with a disconnected Besag spatial block.
    rng = Random.Xoshiro(20260424)
    W = _disconnected_adjacency()
    g = GMRFGraph(W)
    n = num_nodes(g)

    α_true = 0.2
    τ_true = 2.0

    # Sample u ~ BesagGMRF(g, τ_true); the GMRFs sampler respects the
    # per-component null space automatically.
    u = rand(rng, BesagGMRF(g; τ = τ_true, scale_model = false))
    E = fill(40.0, n)
    η_true = log.(E) .+ α_true .+ u
    y = [rand(rng, Poisson(exp(η_true[i]))) for i in 1:n]

    c_int = Intercept()
    c_besag = Besag(g; scale_model = false,
                   hyperprior = PCPrecision(1.0, 0.01))
    A_proj = sparse([ones(n) Matrix{Float64}(I, n, n)])
    ℓ = PoissonLikelihood(; E = E)
    model = LatentGaussianModel(ℓ, (c_int, c_besag), A_proj)

    res = inla(model, y; int_strategy = :grid)

    @test isfinite(res.log_marginal)

    fe = fixed_effects(model, res)
    @test length(fe) == 1
    @test isfinite(fe[1].mean) && fe[1].sd > 0

    re = random_effects(model, res)
    @test length(re) == 1
    v = first(values(re))
    @test length(v.mean) == n
    @test all(v.sd .> 0)

    hp = hyperparameters(model, res)
    @test length(hp) == 1                      # only the Besag τ
    @test isfinite(hp[1].mean) && hp[1].sd > 0
end

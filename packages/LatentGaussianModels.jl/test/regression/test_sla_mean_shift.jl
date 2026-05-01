using LatentGaussianModels
using LatentGaussianModels: GaussianLikelihood, PoissonLikelihood,
                            LogLink, IdentityLink, Intercept, IID, BYM2, PCPrecision,
                            LatentGaussianModel, inla, laplace_mode,
                            ∇³_η_log_density
using GMRFs: GMRFGraph, num_nodes
using Distributions: Poisson
using SparseArrays
using LinearAlgebra: I, Symmetric, diag
using Random

# The SLA mean-shift helper is intentionally not exported.
const _sla_mean_shift = LatentGaussianModels._sla_mean_shift

# Rook-adjacency on a small grid. Reused from
# `test_inla_poisson_bym2.jl`; kept local here to keep the test file
# self-contained.
function _grid_adjacency_sla(nr::Int, nc::Int)
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

@testset "SLA mean-shift — Gaussian-likelihood collapse" begin
    # Gaussian + IdentityLink ⇒ ∇³_η log p ≡ 0 ⇒ Δx = 0 componentwise.
    # The `:simplified_laplace` and `:gaussian` posterior summaries must
    # agree to machine precision.
    rng = Random.Xoshiro(20260428)
    n = 40
    y = 0.2 .+ randn(rng, n)
    ℓ = GaussianLikelihood()
    A = sparse(reshape(ones(n), n, 1))
    model = LatentGaussianModel(ℓ, (Intercept(),), A)

    res_g = inla(model, y; int_strategy=:grid)
    res_sl = inla(model, y; int_strategy=:grid,
        latent_strategy=:simplified_laplace)

    @test maximum(abs, res_g.x_mean .- res_sl.x_mean) < 1.0e-12
    @test maximum(abs, res_g.x_var .- res_sl.x_var) < 1.0e-12
end

@testset "SLA mean-shift — formula matches dense linear algebra" begin
    # On a tiny Poisson + Intercept + IID model (no constraints), the
    # Rue-Martino mean shift Δx = ½ H⁻¹ Aᵀ (h³ ⊙ σ²_η) can be evaluated
    # directly with dense `inv(H)`. Compare against `_sla_mean_shift`.
    rng = Random.Xoshiro(20260428)
    n = 8
    E = fill(1.5, n)
    ℓ = PoissonLikelihood(; E=E)
    y = rand(rng, Poisson(2.0), n)
    A = sparse([ones(n) Matrix{Float64}(I, n,n)])
    model = LatentGaussianModel(ℓ, (Intercept(), IID(n)), A)

    θ = [-1.0]                                     # log-precision for IID
    lp = laplace_mode(model, y, θ)
    @test lp.constraint === nothing                # no constraints here

    Δx_impl = _sla_mean_shift(lp, model, y)

    # Dense reference path.
    H = Matrix(lp.precision)
    Hinv = inv(Symmetric(H))
    Adense = Matrix(A)
    σ²_η_ref = diag(Adense * Hinv * Adense')
    η̂ = Adense * lp.mode
    h³ = ∇³_η_log_density(ℓ, y, η̂, Float64[])
    Δx_ref = 0.5 .* (Hinv * (Adense' * (h³ .* σ²_η_ref)))

    @test maximum(abs, Δx_impl .- Δx_ref) < 1.0e-8
    @test !all(iszero, Δx_impl)                    # genuinely non-trivial
end

@testset "SLA mean-shift — BYM2 sum-to-zero constraint preservation" begin
    # BYM2 carries a sum-to-zero constraint on the spatial block. The
    # mean-shift step projects Δx onto null(C); applying it must keep
    # `C (x̂ + Δx) = e` to numerical precision.
    rng = Random.Xoshiro(20260428)
    g = GMRFGraph(_grid_adjacency_sla(5, 5))
    n = num_nodes(g)

    α_true = 0.3
    τ_true = 4.0
    φ_true = 0.5
    E = fill(20.0, n)

    # Synthetic data (small grid; we only need a non-degenerate fit).
    v_iid = randn(rng, n)
    b_true = (sqrt(1 - φ_true) .* v_iid) ./ sqrt(τ_true)
    η_true = log.(E) .+ α_true .+ b_true
    y = [rand(rng, Poisson(exp(η_true[i]))) for i in 1:n]

    c_int = Intercept()
    c_bym2 = BYM2(g; hyperprior_prec=PCPrecision(1.0, 0.01))
    A = sparse([ones(n) Matrix{Float64}(I, n,n) zeros(n, n)])
    ℓ = PoissonLikelihood(; E=E)
    model = LatentGaussianModel(ℓ, (c_int, c_bym2), A)

    # Use a single Laplace fit so we can pick up `lp.constraint` directly.
    θ̂ = [log(τ_true), 0.0]
    lp = laplace_mode(model, y, θ̂)
    @test lp.constraint !== nothing

    Δx = _sla_mean_shift(lp, model, y)
    C = lp.constraint.C
    e = lp.constraint.e
    # Newton mode already satisfies `C x̂ = e` (laplace_mode projects on
    # exit). Δx must lie in null(C) so the shifted mode does too.
    @test maximum(abs, C * Δx) < 1.0e-8
    @test maximum(abs, C * (lp.mode .+ Δx) .- e) < 1.0e-8

    # End-to-end: full INLA run with `:simplified_laplace` runs cleanly
    # on this BYM2 model and produces a finite, sane summary.
    res_sl = inla(model, y; int_strategy=:grid,
        latent_strategy=:simplified_laplace)
    @test all(isfinite, res_sl.x_mean)
    @test all(res_sl.x_var .≥ 0)
    @test isfinite(res_sl.log_marginal)
end

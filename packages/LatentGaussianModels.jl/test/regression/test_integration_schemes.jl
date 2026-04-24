using LatentGaussianModels: GaussianLikelihood, PoissonLikelihood, Intercept,
    IID, BYM2, LatentGaussianModel, inla, INLA,
    Grid, GaussHermite, CCD
using GMRFs: GMRFGraph
using LinearAlgebra: I
using SparseArrays
using Random

# Hand-built 4-node chain graph — shared by CCD tests below.
_chain4_adjacency() = [0 1 0 0;
                       1 0 1 0;
                       0 1 0 1;
                       0 0 1 0]

@testset "int_strategy — :auto selects Grid for dim(θ) ≤ 2" begin
    rng = Random.Xoshiro(20260424)
    n = 80
    y = 0.3 .+ 0.4 .* randn(rng, n)

    # dim(θ) = 1: Gaussian likelihood + Intercept (no component hyperparams).
    model_1d = LatentGaussianModel(GaussianLikelihood(), (Intercept(),),
                                   sparse(ones(n, 1)))
    res_1d = inla(model_1d, y)                               # :auto default
    @test length(res_1d.θ_points) == 5                       # Grid(5) on 1D

    # dim(θ) = 2: Gaussian + IID → 1 + 1 = 2.
    A_iid = sparse([ones(n) Matrix{Float64}(I, n, n)])
    model_2d = LatentGaussianModel(GaussianLikelihood(),
                                   (Intercept(), IID(n)), A_iid)
    res_2d = inla(model_2d, y)                               # :auto default
    @test length(res_2d.θ_points) == 25                      # Grid(5) tensor in 2D
end

@testset "int_strategy — :auto selects CCD for dim(θ) ≥ 3" begin
    rng = Random.Xoshiro(20260424)
    W = _chain4_adjacency()
    g = GMRFGraph(W)
    n = 4

    # dim(θ) = 3: Gaussian τ_ε + BYM2 (τ_x, φ).
    A = sparse([ones(n) Matrix{Float64}(I, n, n) zeros(n, n)])
    y = 0.3 .+ 0.2 .* randn(rng, n)
    model = LatentGaussianModel(GaussianLikelihood(),
                                (Intercept(), BYM2(g)), A)

    res = inla(model, y)                                     # :auto default
    # CCD in 3D: 1 center + 6 axial + 8 corners = 15 points.
    @test length(res.θ_points) == 15
    @test isfinite(res.log_marginal)
    @test isapprox(sum(res.θ_weights), 1.0; atol = 1.0e-10)
end

@testset "int_strategy — :grid / :ccd / :gauss_hermite symbols" begin
    rng = Random.Xoshiro(7)
    n = 60
    y = randn(rng, n)
    model = LatentGaussianModel(GaussianLikelihood(), (Intercept(),),
                                sparse(ones(n, 1)))

    # Symbol :grid → Grid default (5 points in 1D).
    res_g = inla(model, y; int_strategy = :grid)
    @test length(res_g.θ_points) == 5

    # Symbol :gauss_hermite → GaussHermite default (5 points in 1D).
    res_gh = inla(model, y; int_strategy = :gauss_hermite)
    @test length(res_gh.θ_points) == 5
    @test isapprox(sum(res_gh.θ_weights), 1.0; atol = 1.0e-10)

    # Symbol :ccd on 1D degenerates to Grid(7) per integration.jl:128.
    res_ccd_1d = inla(model, y; int_strategy = :ccd)
    @test length(res_ccd_1d.θ_points) == 7
end

@testset "int_strategy — explicit scheme objects round-trip" begin
    rng = Random.Xoshiro(20260424)
    n = 100
    y = 0.5 .+ 0.6 .* randn(rng, n)
    model = LatentGaussianModel(GaussianLikelihood(), (Intercept(),),
                                sparse(ones(n, 1)))

    res_g = inla(model, y; int_strategy = Grid(n_per_dim = 9, span = 4.0))
    @test length(res_g.θ_points) == 9

    res_gh = inla(model, y; int_strategy = GaussHermite(n_per_dim = 11))
    @test length(res_gh.θ_points) == 11

    # Grid and GaussHermite should give posterior means that agree at
    # high resolution.
    @test isapprox(res_g.x_mean[1], res_gh.x_mean[1]; rtol = 1.0e-3)
end

@testset "int_strategy — CCD explicit at dim(θ) = 3" begin
    rng = Random.Xoshiro(20260424)
    W = _chain4_adjacency()
    g = GMRFGraph(W)
    n = 4
    A = sparse([ones(n) Matrix{Float64}(I, n, n) zeros(n, n)])
    y = 0.3 .+ 0.2 .* randn(rng, n)
    model = LatentGaussianModel(GaussianLikelihood(),
                                (Intercept(), BYM2(g)), A)

    res = inla(model, y; int_strategy = CCD())
    @test length(res.θ_points) == 15                         # 1 + 2m + 2^m for m=3
    @test all(isfinite, res.θ_weights)
    @test isapprox(sum(res.θ_weights), 1.0; atol = 1.0e-10)
end

@testset "int_strategy — unknown symbol raises" begin
    rng = Random.Xoshiro(0)
    n = 40
    y = randn(rng, n)
    model = LatentGaussianModel(GaussianLikelihood(), (Intercept(),),
                                sparse(ones(n, 1)))

    @test_throws ArgumentError inla(model, y; int_strategy = :foo)
end

@testset "int_strategy — Grid vs GaussHermite agree on posterior mean" begin
    # Cross-validation between two quadrature rules on a 1D θ problem:
    # Gaussian + Intercept with unknown τ_ε. At dim(θ) = 1 both rules
    # are well-tested tails-to-tails; at higher dim the Grid's far-tail
    # points can push Laplace past its PD envelope (tracked separately).
    rng = Random.Xoshiro(20260424)
    n = 150
    y = 0.2 .+ 0.4 .* randn(rng, n)

    model = LatentGaussianModel(GaussianLikelihood(), (Intercept(),),
                                sparse(ones(n, 1)))

    res_g = inla(model, y; int_strategy = Grid(n_per_dim = 21, span = 4.0))
    res_gh = inla(model, y; int_strategy = GaussHermite(n_per_dim = 11))

    @test isapprox(res_g.x_mean[1], res_gh.x_mean[1]; rtol = 1.0e-3)
    @test isapprox(res_g.log_marginal, res_gh.log_marginal; rtol = 1.0e-2)
end

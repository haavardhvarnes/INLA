using LatentGaussianModels: GaussianLikelihood, PoissonLikelihood, Intercept,
                            IID, BYM2, LatentGaussianModel, inla, INLA,
                            Grid, GaussHermite, CCD, PCPrecision,
                            compute_skewness_corrections, integration_nodes
using GMRFs: GMRFGraph
using LinearAlgebra: I, Symmetric, eigen, Diagonal
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
    A_iid = sparse([ones(n) Matrix{Float64}(I, n,n)])
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
    A = sparse([ones(n) Matrix{Float64}(I, n,n) zeros(n, n)])
    y = 0.3 .+ 0.2 .* randn(rng, n)
    model = LatentGaussianModel(GaussianLikelihood(),
        (Intercept(), BYM2(g)), A)

    res = inla(model, y)                                     # :auto default
    # CCD in 3D: 1 center + 6 axial + 8 corners = 15 points.
    @test length(res.θ_points) == 15
    @test isfinite(res.log_marginal)
    @test isapprox(sum(res.θ_weights), 1.0; atol=1.0e-10)
end

@testset "int_strategy — :grid / :ccd / :gauss_hermite symbols" begin
    rng = Random.Xoshiro(7)
    n = 60
    y = randn(rng, n)
    model = LatentGaussianModel(GaussianLikelihood(), (Intercept(),),
        sparse(ones(n, 1)))

    # Symbol :grid → Grid default (5 points in 1D).
    res_g = inla(model, y; int_strategy=:grid)
    @test length(res_g.θ_points) == 5

    # Symbol :gauss_hermite → GaussHermite default (5 points in 1D).
    res_gh = inla(model, y; int_strategy=:gauss_hermite)
    @test length(res_gh.θ_points) == 5
    @test isapprox(sum(res_gh.θ_weights), 1.0; atol=1.0e-10)

    # Symbol :ccd on 1D degenerates to Grid(7) per integration.jl:128.
    res_ccd_1d = inla(model, y; int_strategy=:ccd)
    @test length(res_ccd_1d.θ_points) == 7
end

@testset "int_strategy — explicit scheme objects round-trip" begin
    rng = Random.Xoshiro(20260424)
    n = 100
    y = 0.5 .+ 0.6 .* randn(rng, n)
    model = LatentGaussianModel(GaussianLikelihood(), (Intercept(),),
        sparse(ones(n, 1)))

    res_g = inla(model, y; int_strategy=Grid(n_per_dim=9, span=4.0))
    @test length(res_g.θ_points) == 9

    res_gh = inla(model, y; int_strategy=GaussHermite(n_per_dim=11))
    @test length(res_gh.θ_points) == 11

    # Grid and GaussHermite should give posterior means that agree at
    # high resolution.
    @test isapprox(res_g.x_mean[1], res_gh.x_mean[1]; rtol=1.0e-3)
end

@testset "int_strategy — CCD explicit at dim(θ) = 3" begin
    rng = Random.Xoshiro(20260424)
    W = _chain4_adjacency()
    g = GMRFGraph(W)
    n = 4
    A = sparse([ones(n) Matrix{Float64}(I, n,n) zeros(n, n)])
    y = 0.3 .+ 0.2 .* randn(rng, n)
    model = LatentGaussianModel(GaussianLikelihood(),
        (Intercept(), BYM2(g)), A)

    res = inla(model, y; int_strategy=CCD())
    @test length(res.θ_points) == 15                         # 1 + 2m + 2^m for m=3
    @test all(isfinite, res.θ_weights)
    @test isapprox(sum(res.θ_weights), 1.0; atol=1.0e-10)
end

@testset "int_strategy — unknown symbol raises" begin
    rng = Random.Xoshiro(0)
    n = 40
    y = randn(rng, n)
    model = LatentGaussianModel(GaussianLikelihood(), (Intercept(),),
        sparse(ones(n, 1)))

    @test_throws ArgumentError inla(model, y; int_strategy=:foo)
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

    res_g = inla(model, y; int_strategy=Grid(n_per_dim=21, span=4.0))
    res_gh = inla(model, y; int_strategy=GaussHermite(n_per_dim=11))

    @test isapprox(res_g.x_mean[1], res_gh.x_mean[1]; rtol=1.0e-3)
    @test isapprox(res_g.log_marginal, res_gh.log_marginal; rtol=1.0e-2)
end

@testset "Grid — asymmetric skewness corrections (constructor)" begin
    # Default: both sides nothing (symmetric).
    g0 = Grid(n_per_dim=5, span=3.0)
    @test g0.stdev_corr_pos === nothing
    @test g0.stdev_corr_neg === nothing

    # Both vectors: stored.
    g1 = Grid(n_per_dim=5, span=3.0,
              stdev_corr_pos=[2.0, 1.5],
              stdev_corr_neg=[1.0, 0.8])
    @test g1.stdev_corr_pos == [2.0, 1.5]
    @test g1.stdev_corr_neg == [1.0, 0.8]

    # Mixed nothing/vector → reject.
    @test_throws ArgumentError Grid(n_per_dim=5, span=3.0,
                                    stdev_corr_pos=[2.0])
    @test_throws ArgumentError Grid(n_per_dim=5, span=3.0,
                                    stdev_corr_neg=[1.0])

    # Different lengths → reject.
    @test_throws ArgumentError Grid(n_per_dim=5, span=3.0,
                                    stdev_corr_pos=[1.0, 1.0],
                                    stdev_corr_neg=[1.0])

    # Non-positive entries → reject.
    @test_throws ArgumentError Grid(n_per_dim=5, span=3.0,
                                    stdev_corr_pos=[0.0],
                                    stdev_corr_neg=[1.0])
    @test_throws ArgumentError Grid(n_per_dim=5, span=3.0,
                                    stdev_corr_pos=[1.0],
                                    stdev_corr_neg=[-1.0])
end

@testset "Grid — integration_nodes applies asymmetric stretch" begin
    # 1D, θ̂ = 0, Σ = 1 → halfσ = 1 in eigen basis. Without stretch
    # points are z_1d = [-2, 0, 2] for n_per_dim = 3, span = 2.
    θ̂ = [0.0]
    Σ = reshape([1.0], 1, 1)

    # Symmetric reference.
    sym = Grid(n_per_dim=3, span=2.0)
    pts_sym, _ = integration_nodes(sym, θ̂, Σ)
    @test [p[1] for p in pts_sym] ≈ [-2.0, 0.0, 2.0]

    # Asymmetric: pos stretch = 2, neg stretch = 1 → z = -2, 0, 4.
    asym = Grid(n_per_dim=3, span=2.0,
                stdev_corr_pos=[2.0], stdev_corr_neg=[1.0])
    pts_asym, _ = integration_nodes(asym, θ̂, Σ)
    @test [p[1] for p in pts_asym] ≈ [-2.0, 0.0, 4.0]

    # Weights: standard normal Δz · φ(z) per dim — UNCHANGED by stretch
    # (the stretch only relocates points; IS reweight handles the rest).
    _, lws_sym = integration_nodes(sym, θ̂, Σ)
    _, lws_asym = integration_nodes(asym, θ̂, Σ)
    @test lws_sym ≈ lws_asym

    # 2D check: with Σ = diag(1, 1), the eigenbasis is some
    # orthonormal pair (sign of each eigenvector is implementation-
    # defined for degenerate eigenvalues — see LAPACK's `dsyevr`).
    # Project each point back into the eigen-coords and verify the
    # *set* of stretched z-coords matches expectation.
    θ̂2 = [0.0, 0.0]
    Σ2 = Matrix{Float64}(I, 2, 2)
    asym2 = Grid(n_per_dim=3, span=1.0,
                 stdev_corr_pos=[2.0, 3.0],
                 stdev_corr_neg=[0.5, 0.4])
    pts2, _ = integration_nodes(asym2, θ̂2, Σ2)
    F = eigen(Symmetric(Σ2))
    halfσ = F.vectors * Diagonal(sqrt.(max.(F.values, 0.0)))
    # Recover eigen-frame z values for each point by inverting halfσ.
    halfσ_inv = inv(halfσ)
    z_eigen = [halfσ_inv * p for p in pts2]
    z1 = sort(unique(round.([z[1] for z in z_eigen]; digits=10)))
    z2 = sort(unique(round.([z[2] for z in z_eigen]; digits=10)))
    # First eigen-axis: stretches stdev_corr_pos[1] = 2, neg[1] = 0.5;
    # second eigen-axis: pos[2] = 3, neg[2] = 0.4. Span = 1, so the
    # base z values are {-1, 0, 1}.
    @test z1 ≈ [-0.5, 0.0, 2.0]
    @test z2 ≈ [-0.4, 0.0, 3.0]
end

@testset "compute_skewness_corrections — symmetric posterior" begin
    # Standard normal log-posterior in 1D: log π(θ) = -θ²/2.
    # At θ̂ = 0 with Σ = 1 (halfσ = 1):
    #   Δ+ = log π(0) - log π(1) = 1/2,
    #   Δ- = log π(0) - log π(-1) = 1/2,
    # → stretches both = sqrt(0.5 / 0.5) = 1.
    θ̂ = [0.0]
    Σ = reshape([1.0], 1, 1)
    log_post = θ -> -0.5 * sum(abs2, θ)
    pos, neg = compute_skewness_corrections(log_post, θ̂, Σ)
    @test pos ≈ [1.0]  rtol=1.0e-12
    @test neg ≈ [1.0]  rtol=1.0e-12
end

@testset "compute_skewness_corrections — asymmetric posterior matches σ⁺/σ⁻" begin
    # Piecewise log-posterior: σ_left = 1, σ_right = 2.
    #   log π(θ) = -θ²/2 for θ ≤ 0,
    #   log π(θ) = -θ²/8 for θ > 0.
    # halfσ probe = 1 (Σ = 1), so:
    #   Δ+ = 0 - (-1/8) = 1/8 → pos = sqrt(0.5/0.125) = 2,
    #   Δ- = 0 - (-1/2) = 1/2 → neg = sqrt(0.5/0.5)   = 1.
    θ̂ = [0.0]
    Σ = reshape([1.0], 1, 1)
    log_post = θ -> begin
        x = θ[1]
        x ≤ 0 ? -0.5 * x^2 : -0.125 * x^2
    end
    pos, neg = compute_skewness_corrections(log_post, θ̂, Σ)
    @test pos ≈ [2.0]  rtol=1.0e-10
    @test neg ≈ [1.0]  rtol=1.0e-10
end

@testset "compute_skewness_corrections — flat axis stays at 1" begin
    # Drop in log-π below `threshold` along an axis → leave stretch at 1.
    θ̂ = [0.0]
    Σ = reshape([1.0], 1, 1)
    log_post = θ -> -1.0e-6 * sum(abs2, θ)   # essentially flat
    pos, neg = compute_skewness_corrections(log_post, θ̂, Σ;
                                            threshold=0.05)
    @test pos == [1.0]
    @test neg == [1.0]
end

@testset "compute_skewness_corrections — clamps runaway stretches" begin
    # Very steep wall on the right side (Δ+ very large) would give a
    # tiny pos stretch; cap it from below at 1/max_stretch.
    θ̂ = [0.0]
    Σ = reshape([1.0], 1, 1)
    log_post = θ -> begin
        x = θ[1]
        x ≤ 0 ? -0.5 * x^2 : -1.0e6 * x^2
    end
    pos, neg = compute_skewness_corrections(log_post, θ̂, Σ;
                                            max_stretch=5.0)
    @test pos[1] ≈ 1 / 5.0
    @test neg ≈ [1.0]
end

@testset "compute_skewness_corrections — input validation" begin
    θ̂ = [0.0]
    Σ = reshape([1.0], 1, 1)
    log_post = θ -> -0.5 * sum(abs2, θ)
    @test_throws ArgumentError compute_skewness_corrections(
        log_post, θ̂, Σ; threshold=0.0)
    @test_throws ArgumentError compute_skewness_corrections(
        log_post, θ̂, Σ; max_stretch=1.0)
    # log_post that returns -Inf at θ̂ → reject.
    @test_throws ArgumentError compute_skewness_corrections(
        θ -> -Inf, θ̂, Σ)
end

@testset "INLA — skewness_correction=true on dim(θ)=1 runs end-to-end" begin
    # Use a Gaussian likelihood with Intercept (dim θ = 1, log τ_lik).
    # Stretches will be ≈ 1 because the posterior in log-precision is
    # close to Gaussian at moderate n; the test verifies the pipeline
    # plumbing (option flag → compute_skewness_corrections → Grid
    # rebuild → integration_nodes) without breaking the posterior
    # numerics. End-to-end agreement with the symmetric path under a
    # near-Gaussian posterior is the load-bearing assertion.
    rng = Random.Xoshiro(20260504)
    n = 100
    y = 0.3 .+ 0.5 .* randn(rng, n)
    model = LatentGaussianModel(GaussianLikelihood(), (Intercept(),),
        sparse(ones(n, 1)))

    res_sym = inla(model, y;
        int_strategy=Grid(n_per_dim=11, span=3.0),
        skewness_correction=false)
    res_skew = inla(model, y;
        int_strategy=Grid(n_per_dim=11, span=3.0),
        skewness_correction=true)

    @test length(res_skew.θ_points) == 11
    @test isfinite(res_skew.log_marginal)
    @test isapprox(sum(res_skew.θ_weights), 1.0; atol=1.0e-10)
    # Near-Gaussian posterior: skewness-corrected fit should be very
    # close to the symmetric one.
    @test isapprox(res_sym.x_mean[1], res_skew.x_mean[1]; rtol=1.0e-3)
    @test isapprox(res_sym.log_marginal, res_skew.log_marginal; rtol=1.0e-3)
end

@testset "INLA — skewness_correction=true is a no-op for non-Grid schemes" begin
    # When the resolved scheme is not Grid, the flag has no effect;
    # the design should be identical to the uncorrected fit.
    rng = Random.Xoshiro(11)
    n = 60
    y = randn(rng, n)
    model = LatentGaussianModel(GaussianLikelihood(), (Intercept(),),
        sparse(ones(n, 1)))

    res_off = inla(model, y; int_strategy=GaussHermite(n_per_dim=7),
        skewness_correction=false)
    res_on = inla(model, y; int_strategy=GaussHermite(n_per_dim=7),
        skewness_correction=true)
    @test [p[1] for p in res_off.θ_points] ≈ [p[1] for p in res_on.θ_points]
    @test res_off.θ_weights ≈ res_on.θ_weights
end

@testset "INLA — wide-span Grid on dim(θ)=2 survives tail Laplace failure" begin
    # Gaussian + IID has dim(θ) = 2: [log(τ_lik), log(τ_iid)]. With
    # n_per_dim = 11 and span = 3.0 the corner tail points push τ
    # into a regime where H = Q + A'DA is numerically singular and the
    # sparse Cholesky throws. Before the fix this crashed `inla`; the
    # try/catch in the integration loop now drops the offending points
    # and the IS sum proceeds (a `@warn` is emitted — that's expected).
    rng = Random.Xoshiro(20260424)
    n = 80
    α_true = 0.5
    τ_lik_true = 4.0
    τ_iid_true = 6.0

    u = randn(rng, n) ./ sqrt(τ_iid_true)
    y = α_true .+ u .+ randn(rng, n) ./ sqrt(τ_lik_true)

    c_int = Intercept()
    c_iid = IID(n; hyperprior=PCPrecision(1.0, 0.01))
    A = sparse([ones(n) Matrix{Float64}(I, n,n)])
    ℓ = GaussianLikelihood()
    model = LatentGaussianModel(ℓ, (c_int, c_iid), A)

    res = inla(model, y; int_strategy=Grid(n_per_dim=11, span=3.0))

    @test isfinite(res.log_marginal)
    @test all(isfinite, res.x_mean)
    @test all(isfinite, res.x_var)
    @test all(>=(0), res.x_var)
    @test all(isfinite, res.θ_mean)
    # At least the mode-region points must survive after filtering.
    @test length(res.θ_points) ≥ 1
    @test length(res.θ_points) == length(res.θ_weights)
    @test isapprox(sum(res.θ_weights), 1.0; atol=1.0e-10)
end

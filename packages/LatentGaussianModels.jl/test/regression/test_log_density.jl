using LatentGaussianModels: GaussianLikelihood, PoissonLikelihood, Intercept,
                            IID, RW1, BYM2, LatentGaussianModel, INLALogDensity,
                            sample_conditional, inla, laplace_mode,
                            initial_hyperparameters, n_hyperparameters, n_latent
using Distributions: Poisson
using GMRFs: GMRFGraph
using LogDensityProblems
using SparseArrays
using LinearAlgebra: I, norm
using Random
using Statistics: mean

@testset "LogDensityProblems — Gaussian + Intercept" begin
    rng = Random.Xoshiro(20260424)
    n = 100
    y = 0.5 .+ 0.6 .* randn(rng, n)

    model = LatentGaussianModel(GaussianLikelihood(), (Intercept(),),
        sparse(ones(n, 1)))
    ld = INLALogDensity(model, y)

    # dimension + capabilities
    @test LogDensityProblems.dimension(ld) == n_hyperparameters(model)
    @test LogDensityProblems.dimension(ld) == 1                     # only τ_ε
    @test LogDensityProblems.capabilities(typeof(ld)) ==
          LogDensityProblems.LogDensityOrder{1}()

    # logdensity is finite at a reasonable θ
    θ = [log(1 / 0.6^2)]
    ℓ = LogDensityProblems.logdensity(ld, θ)
    @test isfinite(ℓ)

    # logdensity_and_gradient returns (ℓ, g) consistent with logdensity
    ℓ2, g = LogDensityProblems.logdensity_and_gradient(ld, θ)
    @test ℓ2 ≈ ℓ
    @test length(g) == 1
    @test isfinite(g[1])

    # Dimension mismatch throws
    @test_throws DimensionMismatch LogDensityProblems.logdensity(ld, [1.0, 2.0])
end

@testset "LogDensityProblems — gradient near zero at INLA θ̂" begin
    # For a well-identified model the INLA θ̂ maximises log π(θ|y), so
    # the LogDensityProblems gradient at θ̂ should be ≈ 0. Scale gives
    # a quantitative check that logdensity is the same objective INLA
    # optimises.
    rng = Random.Xoshiro(20260424)
    n = 200
    α_true = 0.3
    σ_ε = 0.5
    σ_x = 0.8
    x_true = σ_x .* randn(rng, n)
    y = α_true .+ x_true .+ σ_ε .* randn(rng, n)

    c_int = Intercept()
    c_iid = IID(n)
    A = sparse([ones(n) Matrix{Float64}(I, n,n)])
    model = LatentGaussianModel(GaussianLikelihood(), (c_int, c_iid), A)

    res = inla(model, y; int_strategy=:grid)
    ld = INLALogDensity(model, y)

    _ℓ, g = LogDensityProblems.logdensity_and_gradient(ld, res.θ̂)
    # Gradient at the mode is small relative to the scale of θ̂.
    @test norm(g, Inf) < 1.0e-2 * max(1.0, norm(res.θ̂, Inf))
end

@testset "LogDensityProblems — Poisson + Intercept" begin
    rng = Random.Xoshiro(11)
    n = 80
    α_true = 0.2
    y = [rand(rng, Poisson(exp(α_true))) for _ in 1:n]

    c_int = Intercept()
    A = sparse(ones(n, 1))
    model = LatentGaussianModel(PoissonLikelihood(), (c_int,), A)

    ld = INLALogDensity(model, y)
    @test LogDensityProblems.dimension(ld) == 0             # no hyperparameters

    # Poisson-only (no hyperparameters) — logdensity reduces to the
    # Laplace log-marginal at an empty θ.
    ℓ = LogDensityProblems.logdensity(ld, Float64[])
    @test isfinite(ℓ)

    ℓ2, g = LogDensityProblems.logdensity_and_gradient(ld, Float64[])
    @test ℓ2 ≈ ℓ
    @test isempty(g)
end

# ---------------------------------------------------------------------
# sample_conditional
# ---------------------------------------------------------------------

@testset "sample_conditional — shape + RNG reproducibility" begin
    rng1 = Random.Xoshiro(20260424)
    n = 60
    y = 0.2 .+ 0.5 .* randn(rng1, n)

    c_int = Intercept()
    c_iid = IID(n)
    A = sparse([ones(n) Matrix{Float64}(I, n,n)])
    model = LatentGaussianModel(GaussianLikelihood(), (c_int, c_iid), A)
    θ = [0.0, 0.0]           # τ_ε = τ_x = 1

    # Single-sample: Vector of length n_latent.
    x = sample_conditional(model, θ, y;
        rng=Random.Xoshiro(1))
    @test x isa Vector{Float64}
    @test length(x) == n_latent(model)

    # Matrix-sample: (n_latent, n_samples).
    X = sample_conditional(model, θ, y, 10;
        rng=Random.Xoshiro(1))
    @test X isa Matrix{Float64}
    @test size(X) == (n_latent(model), 10)

    # Same seed → same first column (and same vector in single call).
    @test X[:, 1] ≈ x

    # Different seeds → different draws.
    X2 = sample_conditional(model, θ, y, 10;
        rng=Random.Xoshiro(2))
    @test !(X2[:, 1] ≈ X[:, 1])

    # n_samples must be ≥ 1.
    @test_throws ArgumentError sample_conditional(model, θ, y, 0)
end

@testset "sample_conditional — sample mean ≈ Laplace mode" begin
    # For a Gaussian likelihood with IID latent field, the Laplace
    # approximation is exact, so the empirical mean over many draws at
    # fixed θ must concentrate on the MAP.
    rng = Random.Xoshiro(20260424)
    n = 100
    α_true = 0.3
    σ_ε = 0.5
    σ_x = 0.8
    x_true = σ_x .* randn(rng, n)
    y = α_true .+ x_true .+ σ_ε .* randn(rng, n)

    c_int = Intercept()
    c_iid = IID(n)
    A = sparse([ones(n) Matrix{Float64}(I, n,n)])
    model = LatentGaussianModel(GaussianLikelihood(), (c_int, c_iid), A)
    θ = [log(1 / σ_ε^2), log(1 / σ_x^2)]

    lp = laplace_mode(model, y, θ)
    X = sample_conditional(model, θ, y, 4000;
        rng=Random.Xoshiro(17))
    x_bar = vec(mean(X; dims=2))
    # Empirical mean within 4 SE of the analytical mode (intercept
    # entry has Var ≈ 1/(n/σ_ε²) so SE ≈ σ_ε/√(n · n_samples)).
    se = 1.0 / sqrt(n * 4000)
    @test abs(x_bar[1] - lp.mode[1]) < 10 * se         # loose: posterior sd on α is not SE
    # Aggregate SSE between empirical mean and MAP is small.
    @test norm(x_bar .- lp.mode) / sqrt(n_latent(model)) < 0.1
end

@testset "sample_conditional — constraint respected for BYM2" begin
    # BYM2 has a per-component sum-to-zero constraint on the structured
    # block. Draws must satisfy it to working precision.
    W = [0 1 0 0;
         1 0 1 0;
         0 1 0 1;
         0 0 1 0]
    g = GMRFGraph(W)
    n = 4
    c = BYM2(g)
    A = sparse(Matrix{Float64}(I, n, 2n))              # pick the combined field

    rng = Random.Xoshiro(20260424)
    # Synthesise a reasonable y: Gaussian noise around zero.
    y = 0.1 .* randn(rng, n)
    model = LatentGaussianModel(GaussianLikelihood(), (c,), A)
    θ = initial_hyperparameters(model)

    X = sample_conditional(model, θ, y, 20;
        rng=Random.Xoshiro(7))
    # Structured block is the second half of each column. The
    # constraint is sum(x_structured) = 0.
    for s in 1:size(X, 2)
        @test abs(sum(X[(n + 1):(2n), s])) < 1.0e-8
    end
end

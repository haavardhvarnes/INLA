using LatentGaussianModels: BYM2, PCBYM2Phi, PCPrecision, WeakPrior,
    precision_matrix, log_hyperprior, nhyperparameters,
    initial_hyperparameters, log_prior_density, user_scale, prior_name,
    DEFAULT_BYM2_PHI_ALPHA
using GMRFs: GMRFGraph, BesagGMRF, laplacian_matrix, scale_factor,
    nconnected_components, constraints, constraint_matrix, num_nodes

# 4-node path graph: 1 - 2 - 3 - 4
const _W_PATH = [0 1 0 0;
                 1 0 1 0;
                 0 1 0 1;
                 0 0 1 0]

# 6-node two-component graph: {1-2-3} ⊔ {4-5-6}
const _W_DISCONNECTED = [0 1 0 0 0 0;
                         1 0 1 0 0 0;
                         0 1 0 0 0 0;
                         0 0 0 0 1 0;
                         0 0 0 1 0 1;
                         0 0 0 0 1 0]

@testset "PCBYM2Phi — construction and validation" begin
    # Dummy γ from a 4-node path
    g = GMRFGraph(_W_PATH)
    L = Matrix(laplacian_matrix(g))
    c = scale_factor(g)
    eigs = sort(eigvals(Symmetric(L)))
    γ = c .* eigs[2:end]      # drop the single zero eigenvalue
    @test length(γ) == 3
    @test all(γ .> 0)

    p = PCBYM2Phi(0.5, 2 / 3, γ)
    @test prior_name(p) == :pc_bym2_phi
    @test user_scale(p, 0.0) ≈ 0.5
    @test user_scale(p, 100.0) ≈ 1.0 atol = 1.0e-6
    @test user_scale(p, -100.0) ≈ 0.0 atol = 1.0e-6

    # Validation
    @test_throws ArgumentError PCBYM2Phi(-0.1, 0.5, γ)
    @test_throws ArgumentError PCBYM2Phi(0.5, 1.5, γ)
    @test_throws ArgumentError PCBYM2Phi(0.5, 0.5, Float64[])
    @test_throws ArgumentError PCBYM2Phi(0.5, 0.5, [-1.0, 1.0, 2.0])
end

@testset "PCBYM2Phi — density integrates to 1" begin
    g = GMRFGraph(_W_PATH)
    L = Matrix(laplacian_matrix(g))
    c = scale_factor(g)
    eigs = sort(eigvals(Symmetric(L)))
    γ = c .* eigs[2:end]

    p = PCBYM2Phi(0.5, 2 / 3, γ)
    # Integrate exp(log π(θ)) over a wide θ range; density lives on ℝ
    θ_grid = range(-20, 20; length = 8000)
    dθ = step(θ_grid)
    vals = [exp(log_prior_density(p, θ)) for θ in θ_grid]
    @test all(isfinite, vals)
    total = sum(vals) * dθ
    @test isapprox(total, 1.0; rtol = 1.0e-2)
end

@testset "PCBYM2Phi — quantile matches (U, α)" begin
    # Choose γ that makes d(φ) non-degenerate
    g = GMRFGraph(_W_PATH)
    L = Matrix(laplacian_matrix(g))
    c = scale_factor(g)
    eigs = sort(eigvals(Symmetric(L)))
    γ = c .* eigs[2:end]

    for (U, α) in [(0.5, 2 / 3), (0.3, 0.4), (0.7, 0.9)]
        p = PCBYM2Phi(U, α, γ)
        # Numerically compute P(φ < U) via trapezoid on θ up to logit(U)
        θ_U = log(U / (1 - U))
        θ_grid = range(-25, θ_U; length = 6000)
        dθ = step(θ_grid)
        cdf = sum(exp(log_prior_density(p, θ)) for θ in θ_grid) * dθ
        @test isapprox(cdf, α; rtol = 2.0e-2, atol = 1.0e-2)
    end
end

@testset "BYM2 — basic contract and precision structure" begin
    g = GMRFGraph(_W_PATH)
    n = num_nodes(g)
    c = BYM2(g)
    @test length(c) == 2n
    @test nhyperparameters(c) == 2
    @test initial_hyperparameters(c) == [0.0, 0.0]

    # τ = 2, φ = 0.3
    τ = 2.0
    φ = 0.3
    θ = [log(τ), log(φ / (1 - φ))]
    Q = precision_matrix(c, θ)
    @test size(Q) == (2n, 2n)
    @test issymmetric(Matrix(Q))

    # Block structure check
    Qd = Matrix(Q)
    a = τ / (1 - φ)
    b = -sqrt(τ * φ) / (1 - φ)
    d = φ / (1 - φ)
    # Q_11 = a · I
    @test Qd[1:n, 1:n] ≈ a * I(n)
    # Q_12 = Q_21 = b · I
    @test Qd[1:n, (n + 1):(2n)] ≈ b * I(n)
    @test Qd[(n + 1):(2n), 1:n] ≈ b * I(n)
    # Q_22 = R_scaled + d · I
    sf = scale_factor(g)
    Lmat = Matrix(laplacian_matrix(g))
    R_scaled = sf .* Lmat
    @test Qd[(n + 1):(2n), (n + 1):(2n)] ≈ R_scaled + d * I(n)
end

@testset "BYM2 — limit checks" begin
    g = GMRFGraph(_W_PATH)
    n = num_nodes(g)
    c = BYM2(g)

    # φ → 0 (logit φ very negative): Q_11 → τ·I, Q_12 → 0, u block → Besag
    τ = 1.5
    θ = [log(τ), -30.0]
    Q = Matrix(precision_matrix(c, θ))
    @test Q[1:n, 1:n] ≈ τ * I(n) rtol = 1.0e-6
    @test maximum(abs, Q[1:n, (n + 1):(2n)]) ≤ 1.0e-5
    sf = scale_factor(g)
    Lmat = Matrix(laplacian_matrix(g))
    # Q_22 ≈ R_scaled for small φ
    @test Q[(n + 1):(2n), (n + 1):(2n)] ≈ sf .* Lmat rtol = 1.0e-3 atol = 1.0e-4
end

@testset "BYM2 — log_hyperprior" begin
    g = GMRFGraph(_W_PATH)
    c = BYM2(g)
    θ = [0.0, 0.0]
    lp = log_hyperprior(c, θ)
    @test isfinite(lp)
    # Sum of τ and φ priors
    lp_τ = log_prior_density(c.hyperprior_prec, θ[1])
    lp_φ = log_prior_density(c.hyperprior_phi, θ[2])
    @test lp ≈ lp_τ + lp_φ
end

@testset "BYM2 — constraints (connected and disconnected)" begin
    # Connected graph: one sum-to-zero row, on u block only
    g = GMRFGraph(_W_PATH)
    c = BYM2(g)
    n = num_nodes(g)
    con = constraints(c)
    A = constraint_matrix(con)
    @test size(A) == (1, 2n)
    # b block is zero
    @test all(==(0), A[:, 1:n])
    # u block is all ones (single component)
    @test all(==(1), A[:, (n + 1):(2n)])

    # Disconnected graph: one sum-to-zero row per component
    g2 = GMRFGraph(_W_DISCONNECTED)
    c2 = BYM2(g2)
    n2 = num_nodes(g2)
    @test nconnected_components(g2) == 2
    con2 = constraints(c2)
    A2 = constraint_matrix(con2)
    @test size(A2) == (2, 2n2)
    @test all(==(0), A2[:, 1:n2])
    # Row sums (on u block) equal component sizes (both 3 here)
    @test sum(A2[1, (n2 + 1):(2n2)]) == 3
    @test sum(A2[2, (n2 + 1):(2n2)]) == 3
end

@testset "BYM2 — custom hyperprior override" begin
    g = GMRFGraph(_W_PATH)
    # Use explicit WeakPrior on φ — no prior mass but evaluates to 0.
    c = BYM2(g; hyperprior_phi = WeakPrior())
    @test log_hyperprior(c, [0.0, 0.0]) ==
          log_prior_density(c.hyperprior_prec, 0.0)
end

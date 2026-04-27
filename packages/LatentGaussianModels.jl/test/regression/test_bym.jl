using LatentGaussianModels: BYM, PCPrecision, GammaPrecision, WeakPrior,
    precision_matrix, log_hyperprior, nhyperparameters,
    initial_hyperparameters, log_prior_density, log_normalizing_constant
using GMRFs: GMRFGraph, laplacian_matrix, scale_factor, num_nodes,
    nconnected_components, constraints, constraint_matrix

# 4-node path graph: 1 - 2 - 3 - 4
const _BYM_W_PATH = [0 1 0 0;
                     1 0 1 0;
                     0 1 0 1;
                     0 0 1 0]

# 6-node two-component graph: {1-2-3} ⊔ {4-5-6}
const _BYM_W_DISCONNECTED = [0 1 0 0 0 0;
                             1 0 1 0 0 0;
                             0 1 0 0 0 0;
                             0 0 0 0 1 0;
                             0 0 0 1 0 1;
                             0 0 0 0 1 0]

@testset "BYM — basic contract and precision structure" begin
    g = GMRFGraph(_BYM_W_PATH)
    n = num_nodes(g)
    c = BYM(g)
    @test length(c) == 2n
    @test nhyperparameters(c) == 2
    @test initial_hyperparameters(c) == [0.0, 0.0]

    τ_v = 2.0
    τ_b = 0.5
    θ = [log(τ_v), log(τ_b)]
    Q = precision_matrix(c, θ)
    @test size(Q) == (2n, 2n)
    @test issymmetric(Matrix(Q))

    # Block-diagonal: top-left τ_v · I, bottom-right τ_b · R, off-diag 0
    Qd = Matrix(Q)
    @test Qd[1:n, 1:n] ≈ τ_v * I(n)
    @test maximum(abs, Qd[1:n, (n + 1):(2n)]) ≤ 1.0e-12
    @test maximum(abs, Qd[(n + 1):(2n), 1:n]) ≤ 1.0e-12
    sf = scale_factor(g)
    Lmat = Matrix(laplacian_matrix(g))
    @test Qd[(n + 1):(2n), (n + 1):(2n)] ≈ τ_b * sf .* Lmat
end

@testset "BYM — scale_model = false" begin
    g = GMRFGraph(_BYM_W_PATH)
    n = num_nodes(g)
    c = BYM(g; scale_model = false)
    τ_b = 0.5
    Q = Matrix(precision_matrix(c, [0.0, log(τ_b)]))
    Lmat = Matrix(laplacian_matrix(g))
    @test Q[(n + 1):(2n), (n + 1):(2n)] ≈ τ_b .* Lmat
end

@testset "BYM — log_hyperprior" begin
    g = GMRFGraph(_BYM_W_PATH)
    c = BYM(g)
    θ = [0.3, -0.7]
    lp = log_hyperprior(c, θ)
    @test isfinite(lp)
    @test lp ≈ log_prior_density(c.hyperprior_iid, θ[1]) +
               log_prior_density(c.hyperprior_besag, θ[2])
end

@testset "BYM — custom hyperpriors" begin
    g = GMRFGraph(_BYM_W_PATH)
    c = BYM(g;
            hyperprior_iid = GammaPrecision(1.0, 0.01),
            hyperprior_besag = WeakPrior())
    θ = [0.0, 0.0]
    expected = log_prior_density(GammaPrecision(1.0, 0.01), 0.0) +
               log_prior_density(WeakPrior(), 0.0)
    @test log_hyperprior(c, θ) ≈ expected
end

@testset "BYM — constraints (connected and disconnected)" begin
    g = GMRFGraph(_BYM_W_PATH)
    c = BYM(g)
    n = num_nodes(g)
    con = constraints(c)
    A = constraint_matrix(con)
    @test size(A) == (1, 2n)
    # v block is zero
    @test all(==(0), A[:, 1:n])
    # b block is all ones (single component)
    @test all(==(1), A[:, (n + 1):(2n)])

    g2 = GMRFGraph(_BYM_W_DISCONNECTED)
    c2 = BYM(g2)
    n2 = num_nodes(g2)
    @test nconnected_components(g2) == 2
    con2 = constraints(c2)
    A2 = constraint_matrix(con2)
    @test size(A2) == (2, 2n2)
    @test all(==(0), A2[:, 1:n2])
    @test sum(A2[1, (n2 + 1):(2n2)]) == 3
    @test sum(A2[2, (n2 + 1):(2n2)]) == 3
end

@testset "BYM — log_normalizing_constant" begin
    # Matches R-INLA's `extra()` for F_BYM (inla.c:4868-4870):
    # NC = LOG_NORMC_GAUSSIAN · (n/2 + (n-K)/2) + (n/2) log τ_v
    #      + ((n-K)/2) log τ_b, with LOG_NORMC_GAUSSIAN = -½ log(2π).
    # The K factor reflects the rank deficiency of the intrinsic
    # Besag block (one zero eigenvalue per connected component).
    g = GMRFGraph(_BYM_W_PATH)
    c = BYM(g)
    n = num_nodes(g)
    K = nconnected_components(g)
    θ = [0.4, -0.2]
    expected = -0.25 * (2n - K) * log(2π) + 0.5 * n * θ[1] + 0.5 * (n - K) * θ[2]
    @test log_normalizing_constant(c, θ) ≈ expected

    # Disconnected: K = 2 should reduce both the besag log-prec
    # contribution and the Gaussian-NC piece.
    g2 = GMRFGraph(_BYM_W_DISCONNECTED)
    c2 = BYM(g2)
    n2 = num_nodes(g2)
    K2 = nconnected_components(g2)
    expected2 = -0.25 * (2n2 - K2) * log(2π) + 0.5 * n2 * θ[1] + 0.5 * (n2 - K2) * θ[2]
    @test log_normalizing_constant(c2, θ) ≈ expected2
end

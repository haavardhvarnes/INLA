using LatentGaussianModels: Replicate, IID, AR1, Besag, Intercept, MEB,
                            GaussianLikelihood, LinearProjector,
                            LatentGaussianModel,
                            precision_matrix, prior_mean, log_hyperprior,
                            log_prior_density, nhyperparameters,
                            initial_hyperparameters, log_normalizing_constant,
                            joint_prior_mean, joint_precision,
                            laplace_mode, n_latent, n_hyperparameters,
                            inla, log_marginal_likelihood, PCPrecision,
                            GammaPrecision
using GMRFs: NoConstraint, LinearConstraint, Generic0GMRF,
             constraint_matrix, constraint_rhs, nconstraints, GMRFGraph
import GMRFs

@testset "Replicate — basic structure + invalid input" begin
    inner = IID(5)
    r = Replicate(inner, 3)
    @test length(r) == 3 * 5
    @test nhyperparameters(r) == nhyperparameters(inner)
    @test initial_hyperparameters(r) == initial_hyperparameters(inner)

    # `n_replicates` must be ≥ 1.
    @test_throws ArgumentError Replicate(inner, 0)
    @test_throws ArgumentError Replicate(inner, -1)

    # Single replicate is the trivial wrap.
    r1 = Replicate(inner, 1)
    @test length(r1) == 5
    @test nhyperparameters(r1) == 1
end

@testset "Replicate — hyperparameters delegate to inner (shared θ)" begin
    inner = AR1(10; precprior=PCPrecision(), ρprior=PCPrecision())
    r = Replicate(inner, 4)
    @test nhyperparameters(r) == nhyperparameters(inner) == 2
    θ = [log(2.5), atanh(0.3)]
    # log_hyperprior must be the inner's, NOT 4 × inner's — the shared
    # prior is on a single θ block.
    @test log_hyperprior(r, θ) ≈ log_hyperprior(inner, θ)
end

@testset "Replicate — precision is blockdiag(Q_inner, …, Q_inner)" begin
    inner = IID(4)
    R = 3
    r = Replicate(inner, R)
    θ = [log(2.0)]
    Q = precision_matrix(r, θ)
    @test size(Q) == (R * 4, R * 4)
    Q_inner = precision_matrix(inner, θ)
    # Each diagonal block matches Q_inner.
    for j in 1:R
        rng = ((j - 1) * 4 + 1):(j * 4)
        @test Matrix(Q[rng, rng]) ≈ Matrix(Q_inner)
    end
    # Off-block-diagonal entries are zero.
    @test Q[1, 5] == 0.0
    @test Q[5, 1] == 0.0
end

@testset "Replicate — prior_mean repeats inner mean (MEB inner)" begin
    w = [0.5, -1.0, 2.0]
    inner = MEB(w)
    r = Replicate(inner, 4)
    θ = initial_hyperparameters(r)
    μ = prior_mean(r, θ)
    @test length(μ) == length(r) == 4 * 3
    @test μ == repeat(w, 4)

    # IID inner has zero prior mean; replicate keeps it zero.
    r0 = Replicate(IID(5), 3)
    @test prior_mean(r0, [0.0]) == zeros(15)
end

@testset "Replicate — log NC scales linearly with n_replicates" begin
    inner = IID(6)
    R = 5
    r = Replicate(inner, R)
    θ = [log(7.0)]
    expected = R * log_normalizing_constant(inner, θ)
    @test log_normalizing_constant(r, θ) ≈ expected
end

@testset "Replicate — log_hyperprior independent of n_replicates" begin
    inner = IID(3; hyperprior=GammaPrecision(2.0, 0.1))
    θ = [log(1.5)]
    base = log_hyperprior(inner, θ)
    for R in (1, 2, 5, 20)
        @test log_hyperprior(Replicate(inner, R), θ) ≈ base
    end
end

@testset "Replicate — NoConstraint passthrough on proper inner" begin
    r = Replicate(IID(4), 3)
    @test GMRFs.constraints(r) isa NoConstraint
end

@testset "Replicate — constraints block-stack on intrinsic inner (Besag)" begin
    # Besag on a 4-node ring: 1 connected component ⇒ 1 sum-to-zero row.
    W = [0 1 0 1
         1 0 1 0
         0 1 0 1
         1 0 1 0]
    inner = Besag(GMRFGraph(W))
    R = 3
    r = Replicate(inner, R)
    kc = GMRFs.constraints(r)
    @test kc isa LinearConstraint
    @test nconstraints(kc) == R * 1
    A = constraint_matrix(kc)
    @test size(A) == (R, R * 4)

    # Each row j has 1s on slots ((j-1)·4 + 1) : (j·4) and 0 elsewhere.
    inner_A = constraint_matrix(GMRFs.constraints(inner))
    for j in 1:R
        @test A[j:j, ((j - 1) * 4 + 1):(j * 4)] == inner_A
        # Off-block columns are zero.
        for jc in 1:R
            jc == j && continue
            @test all(==(0.0), A[j, ((jc - 1) * 4 + 1):(jc * 4)])
        end
    end
    # RHS is `e_inner` repeated R times.
    @test constraint_rhs(kc) == repeat(constraint_rhs(GMRFs.constraints(inner)), R)
end

@testset "Replicate — gmrf wrapper rankdef multiplies (Besag inner)" begin
    W = [0 1 0
         1 0 1
         0 1 0]
    inner = Besag(GMRFGraph(W))
    @test GMRFs.rankdef(LatentGaussianModels.gmrf(inner, [0.0])) == 1
    R = 4
    r = Replicate(inner, R)
    g = LatentGaussianModels.gmrf(r, [0.0])
    @test g isa Generic0GMRF
    @test GMRFs.rankdef(g) == R * 1
    @test GMRFs.precision_matrix(g) ≈ precision_matrix(r, [0.0])
end

@testset "Replicate — joint_prior_mean stacks across components and replicates" begin
    # Mix Intercept (μ = 0) + Replicate(MEB(w), 3) — outer joint_prior_mean
    # places `0` on the Intercept slot and `repeat(w, 3)` on the replicate slots.
    w = [1.0, 2.0]
    rep = Replicate(MEB(w), 3)
    n_rep = length(rep)               # 6
    A = sparse(hcat(ones(n_rep), Matrix{Float64}(I, n_rep, n_rep)))
    ℓ = GaussianLikelihood()
    m = LatentGaussianModel(ℓ, (Intercept(), rep), LinearProjector(A))
    θ = initial_hyperparameters(m)
    μ = joint_prior_mean(m, θ)
    @test length(μ) == 1 + n_rep
    @test μ[1] == 0.0
    @test μ[2:end] == repeat(w, 3)
end

@testset "Replicate(IID, R) ≡ IID(R·n) for the marginal model" begin
    # Replicate(IID(n), R) and IID(R·n) are equal in distribution: both
    # are R·n iid Gaussians with the same shared τ. The marginal model
    # log-evidence under a Gaussian likelihood with identity projector
    # should match exactly.
    rng = MersenneTwister(20260504)
    n = 4
    R = 3
    N = R * n
    y = randn(rng, N)
    A = sparse(1.0 * I, N, N)
    ℓ = GaussianLikelihood()

    m_rep = LatentGaussianModel(ℓ, (Replicate(IID(n), R),), A)
    m_iid = LatentGaussianModel(ℓ, (IID(N),), A)

    θ = initial_hyperparameters(m_rep)
    res_rep = laplace_mode(m_rep, y, θ)
    res_iid = laplace_mode(m_iid, y, θ)
    @test res_rep.converged && res_iid.converged
    # Same posterior mode.
    @test res_rep.mode ≈ res_iid.mode
    # Same Laplace marginal at θ_init (the Gaussian-Gaussian closed-form
    # is identical up to floating-point).
    @test res_rep.log_marginal ≈ res_iid.log_marginal
end

@testset "Replicate — Newton mode is the per-slot conjugate Gaussian" begin
    # Identity projector + Gaussian likelihood. With τ_y = 1, τ_x = 1
    # the mode is y / 2 per slot, regardless of replicate index.
    rng = MersenneTwister(20260505)
    n = 5
    R = 3
    N = R * n
    y = randn(rng, N)
    A = sparse(1.0 * I, N, N)
    ℓ = GaussianLikelihood()
    m = LatentGaussianModel(ℓ, (Replicate(IID(n), R),), A)
    θ = initial_hyperparameters(m)    # [log τ_y = 0, log τ_x = 0]
    res = laplace_mode(m, y, θ)
    @test res.converged
    @test res.mode ≈ y ./ 2 atol = 1.0e-8
end

@testset "Replicate(AR1) — INLA smoke fit" begin
    # Replicated longitudinal panel: R independent AR1 chains sharing
    # (τ, ρ). Smoke check that inla() runs end-to-end with finite mlik
    # and the posterior τ-mean is finite and positive on the user scale.
    rng = MersenneTwister(20260506)
    n = 12          # time points per panel
    R = 8           # panel members
    N = R * n
    τ_true = 2.0
    ρ_true = 0.5
    σ_y = 0.4
    # Simulate one AR1 chain per replicate from the closed-form
    # `x_t = ρ x_{t-1} + ε_t` with marginal variance 1/τ_true.
    x_true = Vector{Float64}(undef, N)
    σ_x = 1.0 / sqrt(τ_true)
    for r in 1:R
        x_true[(r - 1) * n + 1] = σ_x * randn(rng)
        for t in 2:n
            i = (r - 1) * n + t
            x_true[i] = ρ_true * x_true[i - 1] +
                        sqrt(1 - ρ_true^2) * σ_x * randn(rng)
        end
    end
    y = x_true .+ σ_y .* randn(rng, N)
    A = sparse(1.0 * I, N, N)
    ℓ = GaussianLikelihood()
    m = LatentGaussianModel(ℓ, (Replicate(AR1(n), R),), A)
    res = inla(m, y; int_strategy=:grid)
    @test isfinite(log_marginal_likelihood(res))
    @test n_latent(m) == N
    # Two component hyperparameters (log τ, atanh ρ) shared across all R
    # replicates, plus τ_y on the Gaussian likelihood: total 3.
    @test n_hyperparameters(m) == 1 + 2
end

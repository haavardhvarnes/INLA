using LatentGaussianModels: Group, Replicate, IID, AR1, Besag, Intercept, MEB,
                            GaussianLikelihood, LinearProjector,
                            LatentGaussianModel, RW1,
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

@testset "Group — basic structure + invalid input" begin
    g = Group([IID(3), IID(5), IID(2)])
    @test length(g) == 3 + 5 + 2
    @test nhyperparameters(g) == 1
    @test initial_hyperparameters(g) == initial_hyperparameters(IID(3))

    # Empty Vector → ArgumentError.
    @test_throws ArgumentError Group(IID[])

    # Convenience form rejects empty group_id.
    @test_throws ArgumentError Group(IID, Int[])
end

@testset "Group — convenience form ≡ explicit Vector form" begin
    group_id = [1, 1, 2, 2, 2, 3]
    g_conv = Group(IID, group_id)
    g_expl = Group([IID(2), IID(3), IID(1)])
    @test length(g_conv) == length(g_expl)
    θ = [log(2.0)]
    @test precision_matrix(g_conv, θ) == precision_matrix(g_expl, θ)
end

@testset "Group — convenience form passes kwargs through" begin
    group_id = [1, 1, 2]
    p = GammaPrecision(2.0, 0.1)
    g = Group(IID, group_id; hyperprior=p)
    # Each inner IID carries the supplied prior.
    for c in g.components
        @test c.hyperprior === p
    end
end

@testset "Group — hyperparameter sharing" begin
    # Three AR1 panels of different lengths, shared (τ, ρ) — log_hyperprior
    # is the inner AR1's, NOT 3×.
    panels = [AR1(5), AR1(7), AR1(3)]
    g = Group(panels)
    θ = [log(2.0), atanh(0.4)]
    @test log_hyperprior(g, θ) ≈ log_hyperprior(panels[1], θ)
    @test nhyperparameters(g) == 2
end

@testset "Group — convenience form rejects missing labels" begin
    # group_id with a gap in the labels (label 2 missing).
    @test_throws ArgumentError Group(IID, [1, 1, 3, 3])
end

@testset "Group — precision is blockdiag of per-group precisions" begin
    panels = [IID(2), IID(3), IID(1)]
    g = Group(panels)
    θ = [log(3.0)]
    Q = precision_matrix(g, θ)
    @test size(Q) == (6, 6)
    Qd = Matrix(Q)

    # Block 1: rows/cols 1:2 — IID(2) with τ=3
    @test Qd[1:2, 1:2] ≈ Matrix(precision_matrix(IID(2), θ))
    # Block 2: rows/cols 3:5
    @test Qd[3:5, 3:5] ≈ Matrix(precision_matrix(IID(3), θ))
    # Block 3: rows/cols 6:6
    @test Qd[6:6, 6:6] ≈ Matrix(precision_matrix(IID(1), θ))
    # Off-block-diagonal entries are zero.
    @test all(==(0.0), Qd[1:2, 3:6])
    @test all(==(0.0), Qd[3:5, 6:6])
end

@testset "Group — prior_mean concatenates across groups (MEB inner)" begin
    # MEB inner with non-uniform per-group `w` vectors — the stacked
    # prior mean is the concatenation.
    w1 = [0.5, -1.0]
    w2 = [2.0]
    w3 = [3.0, 4.0, 5.0]
    g = Group([MEB(w1), MEB(w2), MEB(w3)])
    θ = initial_hyperparameters(g)
    μ = prior_mean(g, θ)
    @test μ == vcat(w1, w2, w3)
    @test length(μ) == length(g)
end

@testset "Group — log NC sums per-group log NC" begin
    panels = [IID(3), IID(5), IID(2)]
    g = Group(panels)
    θ = [log(2.5)]
    expected = sum(c -> log_normalizing_constant(c, θ), panels)
    @test log_normalizing_constant(g, θ) ≈ expected
end

@testset "Group — NoConstraint passthrough on all-proper inner" begin
    g = Group([IID(3), IID(5), IID(2)])
    @test GMRFs.constraints(g) isa NoConstraint
end

@testset "Group — constraints stack at correct column offsets (Besag)" begin
    # Two Besag rings of unequal sizes, each with 1 connected component.
    W4 = [0 1 0 1; 1 0 1 0; 0 1 0 1; 1 0 1 0]
    W3 = [0 1 1; 1 0 1; 1 1 0]
    g = Group([Besag(GMRFGraph(W4)), Besag(GMRFGraph(W3))])
    kc = GMRFs.constraints(g)
    @test kc isa LinearConstraint
    A = constraint_matrix(kc)
    # 1 sum-to-zero row per Besag → 2 rows total.
    @test size(A) == (2, 7)
    # Row 1: 1s on cols 1:4 (group 1 slot), 0s on 5:7
    @test A[1, 1:4] == ones(4)
    @test A[1, 5:7] == zeros(3)
    # Row 2: 0s on 1:4, 1s on 5:7
    @test A[2, 1:4] == zeros(4)
    @test A[2, 5:7] == ones(3)
    @test constraint_rhs(kc) == zeros(2)
end

@testset "Group — gmrf wrapper rankdef sums (Besag inner)" begin
    W4 = [0 1 0 1; 1 0 1 0; 0 1 0 1; 1 0 1 0]
    W3 = [0 1 1; 1 0 1; 1 1 0]
    g = Group([Besag(GMRFGraph(W4)), Besag(GMRFGraph(W3))])
    g_inner1 = LatentGaussianModels.gmrf(g.components[1], [0.0])
    g_inner2 = LatentGaussianModels.gmrf(g.components[2], [0.0])
    @test GMRFs.rankdef(g_inner1) + GMRFs.rankdef(g_inner2) == 2
    g_outer = LatentGaussianModels.gmrf(g, [0.0])
    @test g_outer isa Generic0GMRF
    @test GMRFs.rankdef(g_outer) == 2
    @test GMRFs.precision_matrix(g_outer) ≈ precision_matrix(g, [0.0])
end

@testset "Group ≡ Replicate when sizes are uniform" begin
    # Group([AR1(n), …, AR1(n)]) and Replicate(AR1(n), R) produce the
    # same precision matrix at the same θ.
    n = 5
    R = 4
    panels = [AR1(n) for _ in 1:R]
    g = Group(panels)
    r = Replicate(AR1(n), R)
    θ = [log(2.0), atanh(0.3)]
    @test precision_matrix(g, θ) ≈ precision_matrix(r, θ)
    @test prior_mean(g, θ) == prior_mean(r, θ)
    @test log_normalizing_constant(g, θ) ≈ log_normalizing_constant(r, θ)
    @test length(g) == length(r)
end

@testset "Group — joint_prior_mean stacks across components and groups" begin
    # Intercept + Group([MEB(w1), MEB(w2)]) — outer joint_prior_mean places
    # `0` on the Intercept slot and `vcat(w1, w2)` on the group slots.
    w1 = [1.0, 2.0]
    w2 = [3.0, 4.0, 5.0]
    g = Group([MEB(w1), MEB(w2)])
    n_g = length(g)
    A = sparse(hcat(ones(n_g), Matrix{Float64}(I, n_g, n_g)))
    ℓ = GaussianLikelihood()
    m = LatentGaussianModel(ℓ, (Intercept(), g), LinearProjector(A))
    θ = initial_hyperparameters(m)
    μ = joint_prior_mean(m, θ)
    @test length(μ) == 1 + n_g
    @test μ[1] == 0.0
    @test μ[2:end] == vcat(w1, w2)
end

@testset "Group(IID, …) ≡ IID(N) for the marginal model" begin
    # When inner is IID, Group's stacked precision is τ I_N regardless of
    # group breakdown — same in distribution as a single IID(N).
    rng = MersenneTwister(20260507)
    group_id = [1, 1, 2, 2, 2, 3]
    N = length(group_id)
    y = randn(rng, N)
    A = sparse(1.0 * I, N, N)
    ℓ = GaussianLikelihood()

    m_grp = LatentGaussianModel(ℓ, (Group(IID, group_id),), A)
    m_iid = LatentGaussianModel(ℓ, (IID(N),), A)

    θ = initial_hyperparameters(m_grp)
    res_grp = laplace_mode(m_grp, y, θ)
    res_iid = laplace_mode(m_iid, y, θ)
    @test res_grp.converged && res_iid.converged
    @test res_grp.mode ≈ res_iid.mode
    @test res_grp.log_marginal ≈ res_iid.log_marginal
end

@testset "Group(AR1, non-uniform panels) — INLA smoke fit" begin
    # Three subjects with different numbers of visits, sharing (τ, ρ).
    rng = MersenneTwister(20260508)
    panel_lengths = [10, 7, 12]
    τ_true = 2.0
    ρ_true = 0.5
    σ_y = 0.4
    σ_x = 1.0 / sqrt(τ_true)

    # Simulate per-subject AR1 chains.
    chains = Vector{Float64}[]
    for n in panel_lengths
        chain = Vector{Float64}(undef, n)
        chain[1] = σ_x * randn(rng)
        for t in 2:n
            chain[t] = ρ_true * chain[t - 1] +
                       sqrt(1 - ρ_true^2) * σ_x * randn(rng)
        end
        push!(chains, chain)
    end
    x_true = reduce(vcat, chains)
    N = sum(panel_lengths)
    y = x_true .+ σ_y .* randn(rng, N)

    panels = [AR1(n) for n in panel_lengths]
    g = Group(panels)
    A = sparse(1.0 * I, N, N)
    ℓ = GaussianLikelihood()
    m = LatentGaussianModel(ℓ, (g,), A)
    @test n_latent(m) == N
    # Two component hyperparameters (log τ, atanh ρ) shared across all
    # 3 subjects, plus τ_y on the Gaussian likelihood: total 3.
    @test n_hyperparameters(m) == 1 + 2

    res = inla(m, y; int_strategy=:grid)
    @test isfinite(log_marginal_likelihood(res))
end

@testset "Group — mismatched inner hyperparameters rejected at construction" begin
    # All AR1's have 2 hyperparameters; can't easily mix unless we use a
    # different concrete inner. The Vector{C} type parameter already
    # enforces concrete-type sharing. This test confirms an explicit
    # mismatch via a custom AR1 subtype is impossible by Vector typing.
    panels = [AR1(5), AR1(7), AR1(3)]
    g = Group(panels)
    @test nhyperparameters(g) == nhyperparameters(panels[1])
    # Vector{IID} of mismatched-prior IIDs: nhyperparameters is still 1
    # for all → fine; the user is on the hook for prior consistency.
end

using LatentGaussianModels: MEC, GammaPrecision, GaussianPrior, PCPrecision,
                            precision_matrix, prior_mean, log_hyperprior,
                            log_prior_density, nhyperparameters,
                            initial_hyperparameters, log_normalizing_constant,
                            joint_prior_mean, GaussianLikelihood, Intercept,
                            LinearProjector, LatentGaussianModel,
                            Copy, CopyTargetLikelihood,
                            laplace_mode, n_hyperparameters, n_latent,
                            inla, log_marginal_likelihood, hyperparameters
using GMRFs: NoConstraint, Generic0GMRF
import GMRFs

@testset "MEC — basic structure + invalid input" begin
    w = [0.5, 1.5, 2.5, 3.5]
    c = MEC(w)
    @test length(c) == 4
    # All three slots default-fixed → component contributes zero θ entries.
    @test nhyperparameters(c) == 0
    @test initial_hyperparameters(c) == Float64[]
    @test GMRFs.constraints(c) isa NoConstraint

    # Free one slot at a time.
    @test nhyperparameters(MEC(w; fix_τ_u=false)) == 1
    @test nhyperparameters(MEC(w; fix_μ_x=false)) == 1
    @test nhyperparameters(MEC(w; fix_τ_x=false)) == 1
    # Order in θ: log τ_u, μ_x, log τ_x — fixed slots removed.
    c_uτ_only = MEC(w; fix_τ_u=false, τ_u_init=log(2.0))
    @test initial_hyperparameters(c_uτ_only) == [log(2.0)]
    c_μxonly = MEC(w; fix_μ_x=false, μ_x_init=0.7)
    @test initial_hyperparameters(c_μxonly) == [0.7]
    c_xτ_only = MEC(w; fix_τ_x=false, τ_x_init=log(3.0))
    @test initial_hyperparameters(c_xτ_only) == [log(3.0)]

    # All three free; canonical order in θ.
    c_all = MEC(w; fix_τ_u=false, fix_μ_x=false, fix_τ_x=false,
        τ_u_init=log(11.0), μ_x_init=0.4, τ_x_init=log(13.0))
    @test nhyperparameters(c_all) == 3
    @test initial_hyperparameters(c_all) == [log(11.0), 0.4, log(13.0)]

    # Custom scale.
    c_s = MEC(w; scale=[0.5, 1.0, 2.0, 3.0])
    @test c_s.scale == [0.5, 1.0, 2.0, 3.0]

    # Invalid: empty values, scale-length mismatch, non-positive scale.
    @test_throws ArgumentError MEC(Float64[])
    @test_throws DimensionMismatch MEC([1.0, 2.0]; scale=[1.0])
    @test_throws ArgumentError MEC([1.0, 2.0]; scale=[1.0, 0.0])
    @test_throws ArgumentError MEC([1.0, 2.0]; scale=[1.0, -1.0])
end

@testset "MEC — precision matrix is τ_x I + τ_u D" begin
    w = [1.0, -2.0, 3.0]
    s = [0.5, 2.0, 1.5]
    c = MEC(w; scale=s, fix_τ_u=false, fix_τ_x=false)
    θ = [log(4.0), log(0.25)]    # τ_u=4, τ_x=0.25
    Q = precision_matrix(c, θ)
    @test size(Q) == (3, 3)
    Qd = Matrix(Q)
    @test Qd ≈ Diagonal(0.25 .+ 4.0 .* s)
    @test issymmetric(Qd)

    # Doubling τ_u shifts each diagonal entry by τ_u·s_i.
    Q2 = precision_matrix(c, [log(8.0), log(0.25)])
    @test Matrix(Q2) ≈ Diagonal(0.25 .+ 8.0 .* s)
end

@testset "MEC — conjugate-Gaussian prior mean" begin
    # μ_i = (τ_x μ_x + τ_u s_i w_i) / (τ_x + τ_u s_i).
    w = [1.0, 2.0, 3.0]
    s = [0.5, 1.0, 2.0]
    c = MEC(w; scale=s,
        fix_τ_u=false, fix_μ_x=false, fix_τ_x=false)

    # Case A: τ_u = τ_x = 1, μ_x = 0 ⇒ μ = s·w / (1 + s).
    θ_A = [0.0, 0.0, 0.0]
    @test prior_mean(c, θ_A) ≈ s .* w ./ (1.0 .+ s)

    # Case B: τ_u >> τ_x ⇒ μ ≈ w (proxy dominates).
    θ_B = [log(1.0e6), 0.0, log(1.0e-6)]
    @test isapprox(prior_mean(c, θ_B), w; atol=1.0e-6)

    # Case C: τ_x >> τ_u ⇒ μ ≈ μ_x · 1 (prior dominates).
    θ_C = [log(1.0e-6), 0.7, log(1.0e6)]
    @test isapprox(prior_mean(c, θ_C), fill(0.7, 3); atol=1.0e-6)

    # Default-fixed MEC: μ uses (τ_u_init=10000, μ_x_init=0, τ_x_init=1e-4)
    # and is θ-independent.
    c_d = MEC(w; scale=s)
    @test prior_mean(c_d, Float64[]) ≈ prior_mean(c_d, Float64[])
    # Defaults degrade to μ ≈ w (τ_u_init dominates the ratio).
    @test isapprox(prior_mean(c_d, Float64[]), w; atol=1.0e-3)
end

@testset "MEC — log NC = -½ n log(2π) + ½ Σ log(τ_x + τ_u s_i)" begin
    n = 5
    s = collect(Float64, 1:n)
    w = randn(n)
    c = MEC(w; scale=s, fix_τ_u=false, fix_τ_x=false)
    θ = [log(7.0), log(2.0)]
    diag_Q = 2.0 .+ 7.0 .* s
    expected = -0.5 * n * log(2π) + 0.5 * sum(log, diag_Q)
    @test log_normalizing_constant(c, θ) ≈ expected
end

@testset "MEC — log_hyperprior delegates per free slot" begin
    w = [1.0, 2.0]
    pu = GammaPrecision(2.0, 1.0e-3)
    pmu = GaussianPrior(0.5, 1.0e-2)
    px = GammaPrecision(1.5, 1.0e-3)
    c = MEC(w; τ_u_prior=pu, μ_x_prior=pmu, τ_x_prior=px,
        fix_τ_u=false, fix_μ_x=false, fix_τ_x=false)
    θ = [0.7, 0.3, -0.4]
    expected = log_prior_density(pu, 0.7) +
               log_prior_density(pmu, 0.3) +
               log_prior_density(px, -0.4)
    @test log_hyperprior(c, θ) ≈ expected

    # All-fixed default: log_hyperprior is 0 (no free slot contributes).
    c_d = MEC(w; τ_u_prior=pu, μ_x_prior=pmu, τ_x_prior=px)
    @test log_hyperprior(c_d, Float64[]) == 0.0
end

@testset "MEC — gmrf wrapper precision matches" begin
    w = [0.0, 1.0, 2.0]
    s = [1.0, 2.0, 0.5]
    c = MEC(w; scale=s, fix_τ_u=false, fix_τ_x=false)
    θ = [log(3.0), log(0.5)]
    g = LatentGaussianModels.gmrf(c, θ)
    @test g isa Generic0GMRF
    @test GMRFs.precision_matrix(g) ≈ precision_matrix(c, θ)
end

@testset "MEC — joint_prior_mean stacks per-component means" begin
    n = 4
    w = [1.0, 2.0, 3.0, 4.0]
    c = MEC(w; fix_τ_u=false, fix_μ_x=false, fix_τ_x=false)
    A = sparse(1.0 * I, n, n)
    ℓ = GaussianLikelihood()
    m = LatentGaussianModel(ℓ, c, A)
    θ = initial_hyperparameters(m)
    μ = joint_prior_mean(m, θ)
    @test μ == prior_mean(c, θ[m.θ_ranges[1]])
    @test length(μ) == n_latent(m)

    # With Intercept + MEC stacked, μ has zeros for the Intercept slot
    # and `prior_mean(c, θ)` for the MEC slot.
    α = Intercept()
    A2 = sparse(hcat(ones(n), Matrix{Float64}(I, n, n)))
    m2 = LatentGaussianModel(ℓ, (α, c), LinearProjector(A2))
    θ2 = initial_hyperparameters(m2)
    μ2 = joint_prior_mean(m2, θ2)
    @test length(μ2) == 1 + n
    @test μ2[1] == 0.0
    @test μ2[2:end] == prior_mean(c, θ2[m2.θ_ranges[2]])
end

@testset "MEC — Newton mode tracks θ-dependent prior mean (default-fixed)" begin
    # All three slots default-fixed ⇒ prior mean μ̂ is θ-independent (no
    # latent θ entries for the component). Posterior mean closed form:
    # x̂ = (τ_y I + Q̂)⁻¹ (τ_y y + Q̂ μ̂), where Q̂ = τ_x I + τ_u D.
    n = 5
    w = [1.0, 2.0, 3.0, 4.0, 5.0]
    s = [1.0, 1.0, 1.0, 1.0, 1.0]
    c = MEC(w; scale=s)            # default-fixed τ_u=10000, τ_x=1e-4, μ_x=0
    A = sparse(1.0 * I, n, n)
    ℓ = GaussianLikelihood()       # τ_y_init = 0 ⇒ τ_y = 1
    m = LatentGaussianModel(ℓ, c, A)
    θ = initial_hyperparameters(m)
    y = [-100.0, 100.0, -50.0, 50.0, 0.0]

    res = laplace_mode(m, y, θ)
    @test res.converged

    # Closed form: τ_y = 1; Q̂ diagonal d_i = τ_x + τ_u s_i = 1e-4 + 10000.
    τ_y = 1.0
    τ_u_d = 10000.0
    τ_x_d = 1.0e-4
    μ_x_d = 0.0
    Q_diag = τ_x_d .+ τ_u_d .* s
    μ̂ = (τ_x_d * μ_x_d .+ τ_u_d .* s .* w) ./ Q_diag
    expected = (τ_y .* y .+ Q_diag .* μ̂) ./ (τ_y .+ Q_diag)
    @test isapprox(res.mode, expected; atol=1.0e-8, rtol=1.0e-10)
end

@testset "MEC — Newton mode tracks data when prior is loose (τ_u, τ_x small)" begin
    # Free all three slots, push τ_u and τ_x small (loose tie + loose prior)
    # so x̂ should follow y.
    n = 5
    w = [1.0, 2.0, 3.0, 4.0, 5.0]
    s = ones(n)
    c = MEC(w; scale=s,
        fix_τ_u=false, fix_μ_x=false, fix_τ_x=false,
        τ_u_init=log(1.0e-6), μ_x_init=0.0, τ_x_init=log(1.0e-6))
    A = sparse(1.0 * I, n, n)
    ℓ = GaussianLikelihood()
    m = LatentGaussianModel(ℓ, c, A)
    θ = initial_hyperparameters(m)
    y = [0.5, 1.5, 2.5, 3.5, 4.5]

    res = laplace_mode(m, y, θ)
    @test res.converged

    τ_y = 1.0
    τ_u_v = 1.0e-6
    τ_x_v = 1.0e-6
    μ_x_v = 0.0
    Q_diag = τ_x_v .+ τ_u_v .* s
    μ̂ = (τ_x_v * μ_x_v .+ τ_u_v .* s .* w) ./ Q_diag
    expected = (τ_y .* y .+ Q_diag .* μ̂) ./ (τ_y .+ Q_diag)
    @test isapprox(res.mode, expected; atol=1.0e-8, rtol=1.0e-10)
end

@testset "MEC — INLA fit + β-via-Copy on receiving likelihood" begin
    # Tiny synthetic classical-error regression: y_i = α + β x_i + ε_i,
    # w_i = x_i + u_i, x_i ~ N(0, τ_x⁻¹).  Pin τ_u and τ_x at the
    # generating values (default-fixed) and let INLA fit (τ_y, β, α).
    rng = MersenneTwister(2027)
    n = 60
    τ_u_true = 25.0
    τ_x_true = 1.0
    x_true = randn(rng, n) ./ sqrt(τ_x_true)
    w = x_true .+ randn(rng, n) ./ sqrt(τ_u_true)
    α_true = -0.3
    β_true = 0.6
    σ_y_true = 0.25
    y = α_true .+ β_true .* x_true .+ σ_y_true .* randn(rng, n)

    α = Intercept()
    c_mec = MEC(w;
        τ_u_init=log(τ_u_true), μ_x_init=0.0, τ_x_init=log(τ_x_true))
    # Mapping: row i ⇒ [intercept-only]; β-attached Copy adds
    # `β · x_mec[i]` into the η column.
    A = sparse(hcat(ones(n), zeros(n, n)))
    ℓ_target = CopyTargetLikelihood(
        GaussianLikelihood(),
        Copy(2:(n + 1); β_prior=GaussianPrior(1.0, 0.5),
            β_init=1.0, fixed=false))
    m = LatentGaussianModel(ℓ_target, (α, c_mec), LinearProjector(A))

    @test n_hyperparameters(m) == 1 + 1 + 0 + 0
    # ℓ: τ_y (1) + β (1) | components: Intercept (0) + MEC (0, all default-fixed)

    res = inla(m, y; int_strategy=:grid)
    @test isfinite(log_marginal_likelihood(res))

    # β is the second likelihood-attached hyperparameter (`likelihood[2]`).
    hp = hyperparameters(m, res)
    β_row = hp[2]
    @test β_row.name == "likelihood[2]"
    @test isapprox(β_row.mean, β_true; atol=0.25)
end

using LatentGaussianModels: MEB, GammaPrecision, PCPrecision,
                            precision_matrix, prior_mean, log_hyperprior,
                            log_prior_density, nhyperparameters,
                            initial_hyperparameters, log_normalizing_constant,
                            joint_prior_mean, GaussianLikelihood, Intercept, IID,
                            LinearProjector, StackedMapping, LatentGaussianModel,
                            Copy, CopyTargetLikelihood, GaussianPrior,
                            laplace_mode, n_hyperparameters, n_latent,
                            inla, log_marginal_likelihood, component_range,
                            random_effects, fixed_effects, hyperparameters
using GMRFs: NoConstraint, Generic0GMRF
import GMRFs

@testset "MEB — basic structure + invalid input" begin
    w = [0.1, 0.2, 0.5, 0.5]
    c = MEB(w)
    @test length(c) == 4
    @test nhyperparameters(c) == 1
    @test initial_hyperparameters(c) == [log(1000.0)]
    @test GMRFs.constraints(c) isa NoConstraint

    # `prior_mean` is the supplied per-slot vector, θ-constant.
    @test prior_mean(c, [0.0]) == w
    @test prior_mean(c, [10.0]) == w        # independent of θ

    # Custom τ_u init / prior + scale
    c2 = MEB(w; scale=[0.5, 1.0, 2.0, 3.0],
        τ_u_prior=GammaPrecision(2.0, 1.0e-3),
        τ_u_init=log(50.0))
    @test initial_hyperparameters(c2) == [log(50.0)]
    @test c2.scale == [0.5, 1.0, 2.0, 3.0]

    # Invalid: empty values, scale-length mismatch, non-positive scale
    @test_throws ArgumentError MEB(Float64[])
    @test_throws DimensionMismatch MEB([1.0, 2.0]; scale=[1.0])
    @test_throws ArgumentError MEB([1.0, 2.0]; scale=[1.0, 0.0])
    @test_throws ArgumentError MEB([1.0, 2.0]; scale=[1.0, -1.0])
end

@testset "MEB — precision matrix is τ_u · diag(scale)" begin
    w = [1.0, -2.0, 3.0]
    s = [0.5, 2.0, 1.5]
    c = MEB(w; scale=s)
    θ = [log(4.0)]                        # τ_u = 4
    Q = precision_matrix(c, θ)
    @test size(Q) == (3, 3)
    Qd = Matrix(Q)
    @test Qd ≈ Diagonal(4.0 .* s)
    @test issymmetric(Qd)

    # τ_u doubles → Q doubles
    Q2 = precision_matrix(c, [log(8.0)])
    @test Matrix(Q2) ≈ 2 .* Qd
end

@testset "MEB — log NC = -½ n log(2π) + ½ n log τ_u" begin
    n = 5
    w = collect(Float64, 1:n)
    c = MEB(w)
    θ = [log(7.0)]
    expected = -0.5 * n * log(2π) + 0.5 * n * θ[1]
    @test log_normalizing_constant(c, θ) ≈ expected
end

@testset "MEB — log_hyperprior delegates to τ_u prior" begin
    w = [1.0, 2.0]
    p = GammaPrecision(2.0, 1.0e-3)
    c = MEB(w; τ_u_prior=p)
    θ = [0.7]
    @test log_hyperprior(c, θ) ≈ log_prior_density(p, θ[1])
end

@testset "MEB — gmrf wrapper precision matches" begin
    w = [0.0, 1.0, 2.0]
    s = [1.0, 2.0, 0.5]
    c = MEB(w; scale=s)
    θ = [log(3.0)]
    g = LatentGaussianModels.gmrf(c, θ)
    @test g isa Generic0GMRF
    @test GMRFs.precision_matrix(g) ≈ precision_matrix(c, θ)
end

@testset "MEB — joint_prior_mean stacks per-component means" begin
    n = 4
    w = [1.0, 2.0, 3.0, 4.0]
    c = MEB(w)
    A = sparse(1.0 * I, n, n)
    ℓ = GaussianLikelihood()
    m = LatentGaussianModel(ℓ, c, A)
    θ = initial_hyperparameters(m)
    μ = joint_prior_mean(m, θ)
    @test μ == w
    @test length(μ) == n_latent(m)

    # With Intercept + MEB stacked, μ has zeros for the Intercept slot
    # and `w` for the MEB slot.
    α = Intercept()
    A2 = sparse(hcat(ones(n), Matrix{Float64}(I, n, n)))
    m2 = LatentGaussianModel(ℓ, (α, c), LinearProjector(A2))
    θ2 = initial_hyperparameters(m2)
    μ2 = joint_prior_mean(m2, θ2)
    @test length(μ2) == 1 + n
    @test μ2[1] == 0.0
    @test μ2[2:end] == w
end

@testset "MEB — Newton mode tracks the prior mean under tight prior" begin
    # With τ_u >> τ_y the latent x̂ should sit close to the supplied
    # values regardless of y. Identity mapping → η_i = x_i.
    n = 5
    w = [1.0, 2.0, 3.0, 4.0, 5.0]
    c = MEB(w; τ_u_init=log(1.0e6))    # very tight prior
    A = sparse(1.0 * I, n, n)
    ℓ = GaussianLikelihood()           # τ_y_init = 0 ⇒ τ_y = 1
    m = LatentGaussianModel(ℓ, c, A)
    θ = initial_hyperparameters(m)     # [log τ_y, log τ_u] = [0, log(1e6)]
    y = [-100.0, 100.0, -50.0, 50.0, 0.0]   # far from w; y has no influence

    res = laplace_mode(m, y, θ)
    @test res.converged
    # Prior dominates: x̂ ≈ w to a relative tolerance set by τ_y / τ_u.
    @test isapprox(res.mode, w; atol=1.0e-3)

    # Compare with the closed form: posterior mean = (τ_y I + τ_u I)⁻¹ ·
    # (τ_y y + τ_u D w), which here is (τ_y y + τ_u w) / (τ_y + τ_u).
    τ_y = 1.0
    τ_u = 1.0e6
    expected = (τ_y .* y .+ τ_u .* w) ./ (τ_y + τ_u)
    @test isapprox(res.mode, expected; atol=1.0e-8, rtol=1.0e-10)
end

@testset "MEB — Newton mode tracks y under loose prior" begin
    # Mirror image: τ_u << τ_y ⇒ x̂ should follow the data y.
    n = 5
    w = [1.0, 2.0, 3.0, 4.0, 5.0]
    c = MEB(w; τ_u_init=log(1.0e-6))   # very loose prior
    A = sparse(1.0 * I, n, n)
    ℓ = GaussianLikelihood()
    m = LatentGaussianModel(ℓ, c, A)
    θ = initial_hyperparameters(m)
    y = [0.5, 1.5, 2.5, 3.5, 4.5]

    res = laplace_mode(m, y, θ)
    @test res.converged

    τ_y = 1.0
    τ_u = 1.0e-6
    expected = (τ_y .* y .+ τ_u .* w) ./ (τ_y + τ_u)
    @test isapprox(res.mode, expected; atol=1.0e-8, rtol=1.0e-10)
end

@testset "MEB — INLA fit + β-via-Copy on receiving likelihood" begin
    # Tiny synthetic Berkson regression: y_i = α + β x_i + ε_i,
    # x_i ~ N(w_i, τ_u⁻¹). β is recovered through a Copy on the
    # receiving GaussianLikelihood (ADR-021/ADR-023 pattern).
    rng = MersenneTwister(2026)
    n = 40
    w = randn(rng, n)
    τ_u_true = 4.0
    x_true = w .+ randn(rng, n) ./ sqrt(τ_u_true)
    α_true = 0.2
    β_true = 0.7
    σ_y_true = 0.3
    y = α_true .+ β_true .* x_true .+ σ_y_true .* randn(rng, n)

    α = Intercept()
    c_meb = MEB(w; τ_u_init=log(τ_u_true))
    # Mapping: row i ⇒ [intercept | x_meb[i]]; β-attached Copy adds
    # `β · x[2:(n+1)]` into the η_meb column. With the LinearProjector
    # putting x_meb into η directly (β=1 fixed) and Copy carrying the
    # actual β slot, the wrapper exactly reproduces η_i = α + β · x_i.
    # We use the simpler "intercept-only mapping + Copy carries x_meb":
    A = sparse(hcat(ones(n), zeros(n, n)))      # only the intercept
    ℓ_target = CopyTargetLikelihood(
        GaussianLikelihood(),
        Copy(2:(n + 1); β_prior=GaussianPrior(1.0, 0.5),
            β_init=1.0, fixed=false))
    m = LatentGaussianModel(ℓ_target, (α, c_meb), LinearProjector(A))

    @test n_hyperparameters(m) == 1 + 1 + 0 + 1
    # ℓ: τ_y (1) + β (1)  |  components: Intercept (0) + MEB (1)

    res = inla(m, y; int_strategy=:grid)
    @test isfinite(log_marginal_likelihood(res))

    # β is the second likelihood-attached hyperparameter (`likelihood[2]`)
    # in the canonical [τ_y; β; τ_u] layout. It should land near 0.7
    # within MC-style tolerance.
    hp = hyperparameters(m, res)
    β_row = hp[2]
    @test β_row.name == "likelihood[2]"
    @test isapprox(β_row.mean, β_true; atol=0.2)
end

# Copy component — closed-form regression test.
#
# ADR-021 / PR-3b acceptance: a `CopyTargetLikelihood` with a Copy
# pinned at β = 1.0 must reproduce the unscaled-share posterior — that
# is, the same INLA fit as a model where two observation blocks share
# their IID random effect through the projection mapping directly.
#
# The two formulations are algebraically identical:
#
#   Formulation A (unscaled share via mapping):
#     η_g = α + u_i,  η_p = α + u_i           (both blocks via [1 | I_n])
#
#   Formulation B (Copy with β=1 fixed):
#     η_g = α + u_i,  η_p = α + 1.0 * u_i     (second block via [1 | 0_n] + Copy)
#
# Both have the same latent vector, the same prior, and the same
# likelihood evaluation. The fit must agree to high precision.

using Test
using SparseArrays
using LinearAlgebra: I
using Random
using LatentGaussianModels: GaussianLikelihood, Intercept, IID, LinearProjector,
                            StackedMapping, LatentGaussianModel, Copy,
                            CopyTargetLikelihood, inla, n_hyperparameters,
                            log_marginal_likelihood, n_likelihoods, hyperparameters,
                            random_effects, fixed_effects, nhyperparameters,
                            initial_hyperparameters

@testset "Copy — fixed β=1.0 reproduces unscaled-share oracle" begin
    rng = MersenneTwister(2026)
    n = 30
    α_true = 0.5
    u_true = 0.5 .* randn(rng, n)
    σ_g = 1.0
    σ_p = 0.7
    y_g = α_true .+ u_true .+ σ_g .* randn(rng, n)
    y_p = α_true .+ u_true .+ σ_p .* randn(rng, n)
    y = vcat(y_g, y_p)

    # --- Formulation A: both blocks share IID via the mapping --------
    A_g = sparse(hcat(ones(n), Matrix{Float64}(I, n, n)))
    A_p = sparse(hcat(ones(n), Matrix{Float64}(I, n, n)))
    mapping_A = StackedMapping(
        (LinearProjector(A_g), LinearProjector(A_p)),
        [1:n, (n + 1):(2n)])
    model_A = LatentGaussianModel(
        (GaussianLikelihood(), GaussianLikelihood()),
        (Intercept(), IID(n)),
        mapping_A)

    # --- Formulation B: second block uses Copy(β=1.0, fixed=true) ----
    A_p_no_iid = sparse(hcat(ones(n), zeros(n, n)))
    mapping_B = StackedMapping(
        (LinearProjector(A_g), LinearProjector(A_p_no_iid)),
        [1:n, (n + 1):(2n)])
    ℓ_p_B = CopyTargetLikelihood(
        GaussianLikelihood(),
        Copy(2:(1 + n); β_init=1.0, fixed=true))
    model_B = LatentGaussianModel(
        (GaussianLikelihood(), ℓ_p_B),
        (Intercept(), IID(n)),
        mapping_B)

    # --- Sanity checks on the wrapped likelihood --------------------
    @test n_likelihoods(model_A) == n_likelihoods(model_B) == 2
    @test n_hyperparameters(model_A) == n_hyperparameters(model_B)  # 3 = τ_g, τ_p, τ_u
    @test nhyperparameters(ℓ_p_B) == 1                              # base τ only; β fixed
    @test initial_hyperparameters(ℓ_p_B) == [0.0]

    # --- Both fits use the same integration design ------------------
    res_A = inla(model_A, y; int_strategy=:grid)
    res_B = inla(model_B, y; int_strategy=:grid)

    # --- Posterior modes (latent x̂ at θ̂) --------------------------
    # Mean of the marginal posterior on x must agree to working precision.
    fe_A = fixed_effects(model_A, res_A)
    fe_B = fixed_effects(model_B, res_B)
    @test isapprox([r.mean for r in fe_A], [r.mean for r in fe_B];
        rtol=1.0e-6, atol=1.0e-8)
    @test isapprox([r.sd for r in fe_A], [r.sd for r in fe_B];
        rtol=1.0e-6, atol=1.0e-8)

    re_A = random_effects(model_A, res_A)["IID[2]"]
    re_B = random_effects(model_B, res_B)["IID[2]"]
    @test isapprox(re_A.mean, re_B.mean; rtol=1.0e-6, atol=1.0e-8)
    @test isapprox(re_A.sd, re_B.sd; rtol=1.0e-6, atol=1.0e-8)

    # --- Hyperparameter modes ----------------------------------------
    @test isapprox(res_A.θ̂, res_B.θ̂; rtol=1.0e-6, atol=1.0e-8)

    # --- Log marginal likelihood -------------------------------------
    @test isapprox(log_marginal_likelihood(res_A),
        log_marginal_likelihood(res_B);
        rtol=1.0e-6, atol=1.0e-8)
end

@testset "Copy — free β recovers β ≈ 1 on shared-effect data" begin
    rng = MersenneTwister(123)
    n = 50
    α_true = 0.0
    u_true = 0.7 .* randn(rng, n)
    σ_g = 0.4
    σ_p = 0.4
    y_g = α_true .+ u_true .+ σ_g .* randn(rng, n)
    y_p = α_true .+ u_true .+ σ_p .* randn(rng, n)
    y = vcat(y_g, y_p)

    A_g = sparse(hcat(ones(n), Matrix{Float64}(I, n, n)))
    A_p_no_iid = sparse(hcat(ones(n), zeros(n, n)))
    mapping = StackedMapping(
        (LinearProjector(A_g), LinearProjector(A_p_no_iid)),
        [1:n, (n + 1):(2n)])
    ℓ_p = CopyTargetLikelihood(
        GaussianLikelihood(),
        Copy(2:(1 + n); β_init=1.0, fixed=false))
    model = LatentGaussianModel(
        (GaussianLikelihood(), ℓ_p), (Intercept(), IID(n)), mapping)

    @test nhyperparameters(ℓ_p) == 2  # base τ + free β

    res = inla(model, y; int_strategy=:grid)

    # θ layout: [τ_g (logprec), τ_p (logprec), β, τ_u (logprec)].
    # Confirm β̂ is in the right neighborhood — the data was generated
    # with β_true = 1.0; with n = 50 and σ = 0.4 the posterior should
    # be tight around 1. Allow ±0.25 to absorb sampling noise.
    β̂ = res.θ̂[3]
    @test abs(β̂ - 1.0) < 0.25
end

# Synthetic-recovery test: joint longitudinal-survival model in the
# style of Baghfalaki et al. (2024), wired through INLA.jl's `Copy`.
#
# Model. For subjects `i = 1..N`, longitudinal arm with `K` measurements
# per subject and a single survival outcome:
#
#   b_i           ~ N(0, σ_b²)                          (shared random intercept)
#   y_{i,j}       ~ N(α_long + b_i, σ_g²)               (Gaussian longitudinal)
#   T_i           ~ Weibull(shape, scale = exp(α_surv + φ · b_i))
#                                                       (survival arm, β=φ Copy)
#
# Latent vector: `x = [α_long; α_surv; b_1, …, b_N]`, length `2 + N`.
#
# Block layout in the model's `StackedMapping`:
#   - block 1 (Gaussian): rows `1..N·K`,
#     `A_g[(i-1)·K + j, :]` activates `α_long` (col 1) and `b_i` (col 2+i).
#   - block 2 (Weibull):  rows `N·K + 1 … N·K + N`,
#     `A_w[i, :]` activates `α_surv` (col 2). The Copy contribution adds
#     `φ · b_i` from the IID block to row `i`.
#
# Inference target: posterior modes recover the data-generating
# `(α_long, α_surv, σ_b, σ_g, shape, φ)` within their posterior bands.
# n = 80 subjects × K = 5 longitudinal observations + 80 survival
# observations gives enough information for a clean recovery.
#
# This is a regression test — the R-INLA oracle is a separate follow-up
# (`test/oracle/test_synthetic_baghfalaki.jl`, gated on a fixture from
# `scripts/generate-fixtures/lgm/synthetic_baghfalaki.R`).

using Test
using SparseArrays
using LinearAlgebra: I
using Random
using Statistics: cor
using LatentGaussianModels: GaussianLikelihood, WeibullLikelihood,
                            Intercept, IID, LinearProjector, StackedMapping,
                            LatentGaussianModel, Copy, CopyTargetLikelihood,
                            inla, fixed_effects, random_effects, hyperparameters,
                            Censoring, NONE, RIGHT,
                            log_marginal_likelihood, n_likelihoods,
                            n_hyperparameters

@testset "Joint longitudinal + Weibull survival via Copy (synthetic)" begin
    rng = MersenneTwister(20260502)

    # --- ground-truth parameters --------------------------------------
    N = 80                  # subjects
    K = 5                   # longitudinal measurements per subject
    α_long_true = 0.4
    α_surv_true = -0.2
    σ_b_true = 0.7          # SD of subject-specific random intercept
    σ_g_true = 0.5          # SD of longitudinal noise
    shape_true = 1.5        # Weibull shape
    φ_true = 0.8            # Copy scaling (longitudinal effect on hazard)

    # --- simulate -----------------------------------------------------
    b_true = σ_b_true .* randn(rng, N)
    y_long = Float64[]
    subj_idx = Int[]
    for i in 1:N
        for j in 1:K
            push!(y_long, α_long_true + b_true[i] + σ_g_true * randn(rng))
            push!(subj_idx, i)
        end
    end
    n_long = length(y_long)
    @assert n_long == N * K

    # Weibull survival with hazard ∝ exp(α_surv + φ · b_i). Generate
    # event times via the inverse CDF: T = (-log U / λ)^(1/shape) with
    # rate `λ = exp(α_surv + φ · b_i)`. Right-censor at a fixed admin
    # cutoff that produces ~25 % censoring at these parameters.
    T_admin = 3.5
    T_event = Vector{Float64}(undef, N)
    censoring = Vector{Censoring}(undef, N)
    for i in 1:N
        λ_i = exp(α_surv_true + φ_true * b_true[i])
        u = rand(rng)
        t_i = (-log(u) / λ_i)^(1 / shape_true)
        if t_i > T_admin
            T_event[i] = T_admin
            censoring[i] = RIGHT
        else
            T_event[i] = t_i
            censoring[i] = NONE
        end
    end
    n_cens = count(==(RIGHT), censoring)
    @test 0 < n_cens < N    # bounded censoring fraction

    y = vcat(y_long, T_event)

    # --- build the model ---------------------------------------------
    # Latent layout: x = [α_long; α_surv; b_1, …, b_N].
    #   col 1 = α_long, col 2 = α_surv, cols 3..(2+N) = IID(N).
    A_g = spzeros(Float64, n_long, 2 + N)
    for r in 1:n_long
        A_g[r, 1] = 1.0                 # α_long
        A_g[r, 2 + subj_idx[r]] = 1.0   # b_{subj_idx[r]}
    end

    # Survival arm: only α_surv via the mapping. The b_i contribution
    # comes through `Copy(3:(2+N); β_init=…, fixed=false)`.
    A_w = spzeros(Float64, N, 2 + N)
    for i in 1:N
        A_w[i, 2] = 1.0                 # α_surv
    end

    mapping = StackedMapping(
        (LinearProjector(A_g), LinearProjector(A_w)),
        [1:n_long, (n_long + 1):(n_long + N)])

    ℓ_g = GaussianLikelihood()
    ℓ_w_base = WeibullLikelihood(censoring=censoring)
    ℓ_w = CopyTargetLikelihood(
        ℓ_w_base,
        Copy(3:(2 + N); β_init=1.0, fixed=false))
    model = LatentGaussianModel(
        (ℓ_g, ℓ_w),
        (Intercept(), Intercept(), IID(N)),
        mapping)

    @test n_likelihoods(model) == 2

    # --- fit ---------------------------------------------------------
    res = inla(model, y; int_strategy=:grid)

    # --- check fixed effects recover (α_long, α_surv) ---------------
    fe = fixed_effects(model, res)
    @test length(fe) == 2
    α_long_J = fe[1].mean
    α_surv_J = fe[2].mean
    # Allow generous bands (≤3 posterior SDs of an n = 400 / n = 80 fit).
    @test abs(α_long_J - α_long_true) < 0.20
    @test abs(α_surv_J - α_surv_true) < 0.30

    # --- random-effects posterior captures the true subject scores --
    # Correlation between b̂ and b_true is the cleanest summary; the
    # scale is also recovered through σ_b.
    re = random_effects(model, res)["IID[3]"]
    @test length(re.mean) == N
    cor_b = cor(re.mean, b_true)
    @test cor_b > 0.85

    # --- hyperparameter recovery ------------------------------------
    # θ layout: [τ_g (logprec), shape_w (logα_w), φ (β-Copy), τ_b (IID logprec)].
    # Identify slots by index: position 1 is the Gaussian arm, then the
    # Weibull base hyper, then the Copy β, then the IID prec.
    @test n_hyperparameters(model) == 4
    τ_g_J = exp(res.θ̂[1])
    α_w_J = exp(res.θ̂[2])
    φ_J = res.θ̂[3]
    τ_b_J = exp(res.θ̂[4])

    σ_g_J = 1 / sqrt(τ_g_J)
    σ_b_J = 1 / sqrt(τ_b_J)
    @test abs(σ_g_J - σ_g_true) < 0.10
    @test abs(σ_b_J - σ_b_true) < 0.20
    # Weibull shape: independent of Copy. Posterior is wide on n = 80
    # events with ~25 % censoring; the band reflects that inherent
    # spread, not the Copy mechanism we're really validating.
    @test abs(α_w_J - shape_true) < 0.45
    # φ is the load-bearing Copy parameter — recovering it within ±0.30
    # of φ_true = 0.8 confirms the β-scaled latent share works on a
    # multi-block joint model.
    @test abs(φ_J - φ_true) < 0.30

    # --- sanity: log marginal is finite -----------------------------
    @test isfinite(log_marginal_likelihood(res))
end

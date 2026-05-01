# End-to-end synthetic-recovery test for WeibullCureLikelihood. R-INLA
# does not have a `weibullcure` family (cf. ADR-018), so this is the
# tier-1 calibration: simulate from the generative model with known
# parameters, fit via `inla(...)`, and check posterior recovery within
# moderate-n tolerances.

using LatentGaussianModels: WeibullCureLikelihood, NONE, RIGHT, Censoring,
                            Intercept, FixedEffects, LatentGaussianModel, inla,
                            fixed_effects, hyperparameters

@testset "INLA — WeibullCure (synthetic recovery, n = 400)" begin
    rng = Random.Xoshiro(20260430)
    n = 400
    α_true = 0.3       # intercept (linear predictor)
    β_true = 0.6       # covariate slope
    α_w_true = 1.2       # Weibull shape
    p_true = 0.30      # cure fraction

    x = randn(rng, n)
    λ = @. exp(α_true + β_true * x)
    cured = rand(rng, n) .< p_true
    U = rand(rng, n)
    T_event = @. (-log(U) / λ)^(1 / α_w_true)
    T_true = [cured[i] ? Inf : T_event[i] for i in 1:n]
    C = 0.5 .+ 5.5 .* rand(rng, n)
    event = [t ≤ c ? 1 : 0 for (t, c) in zip(T_true, C)]
    y = [min(t, c) for (t, c) in zip(T_true, C)]

    cens = Censoring[e == 1 ? NONE : RIGHT for e in event]
    ℓ = WeibullCureLikelihood(censoring=cens)
    A = sparse(hcat(ones(n), reshape(x, n, 1)))
    model = LatentGaussianModel(ℓ, (Intercept(), FixedEffects(1)), A)

    res = inla(model, y)

    fe = fixed_effects(model, res)
    @test length(fe) == 2
    # Recovery within ~4σ — the moderate-n cure model has substantial
    # posterior uncertainty driven by the cure fraction and the heavy
    # right-censoring it induces.
    @test abs(fe[1].mean - α_true) < 4 * fe[1].sd
    @test abs(fe[2].mean - β_true) < 4 * fe[2].sd

    hp = hyperparameters(model, res)
    @test length(hp) == 2

    # Internal-scale θ̂ = [log α_w, logit p]; check the *mode* lies in a
    # generous envelope around the truth (heavy posterior tails on logit p
    # at moderate n are typical for cure models).
    @test abs(res.θ̂[1] - log(α_w_true)) < 1.0
    @test abs(res.θ̂[2] - log(p_true / (1 - p_true))) < 2.0

    @test isfinite(res.log_marginal)
end

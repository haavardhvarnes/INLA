using LatentGaussianModels: GaussianLikelihood, PoissonLikelihood, Intercept,
                            IID, BYM2, LatentGaussianModel, inla, inla_summary,
                            fixed_effects, random_effects, hyperparameters
using GMRFs: GMRFGraph
using LinearAlgebra: I
using Printf: @sprintf
using SparseArrays
using Random

# ---------------------------------------------------------------------
# Reference: R-INLA's `summary.inla` layout (simplified, abridged).
# ---------------------------------------------------------------------
# R-INLA prints (in order):
#   Call:           <the inla(...) invocation string>
#   Time used:      Pre / Running / Post / Total seconds
#   Fixed effects:
#                   name   mean   sd   0.025quant   0.5quant
#                   0.975quant   mode   kld
#   Random effects:
#                   Name   Model                  (metadata only here)
#   Model hyperparameters:
#                   name   mean   sd   0.025quant   0.5quant
#                   0.975quant   mode
#   Expected number of effective parameters(std dev): X (Y)
#   Number of equivalent replicates: Z
#   Deviance Information Criterion (DIC): ...       (if diagnostics requested)
#   Watanabe-Akaike information criterion (WAIC): ... (idem)
#   Marginal log-Likelihood: V
# ---------------------------------------------------------------------
# Our `inla_summary` layout (locked in below):
#   "INLA fit ─────..."  summary header box with log p(y) inline
#   "Fixed effects:"                 — header + per-row table
#   "Random effects (posterior summaries per component):"
#                                     (we surface posterior mean/sd
#                                      aggregates, R-INLA surfaces
#                                      component metadata)
#   "Hyperparameters (internal scale):"
#                                     (we report log-precision;
#                                      R-INLA reports user-scale
#                                      precision under "Model hyperparameters")
# Diagnostics (DIC/WAIC/CPO/PIT) are computed on demand by
# `dic`/`waic`/`cpo`/`pit` and are not part of the summary layout,
# per CLAUDE.md "diagnostics are computed on demand".
# ---------------------------------------------------------------------

function _capture_summary(model, res; level::Real=0.95)
    io = IOBuffer()
    inla_summary(io, model, res; level=level)
    return String(take!(io))
end

@testset "inla_summary layout — Gaussian + Intercept" begin
    rng = Random.Xoshiro(20260424)
    n = 100
    α_true = 0.5
    σ = 0.4
    y = α_true .+ σ .* randn(rng, n)

    model = LatentGaussianModel(GaussianLikelihood(), (Intercept(),),
        sparse(ones(n, 1)))
    res = inla(model, y; int_strategy=:grid)
    out = _capture_summary(model, res)

    # --- header block ---
    @test startswith(out, "INLA fit")
    @test occursin(r"latent dim\s+=\s+1", out)
    @test occursin(r"θ dim\s+=\s+1", out)
    @test occursin(r"design pts\s+=\s+5", out)
    @test occursin(r"log p\(y\)\s+=\s+-?[\d.]+", out)

    # --- Fixed effects section ---
    @test occursin("Fixed effects:", out)
    # Column headers: name, mean, sd, lower-quantile, upper-quantile.
    @test occursin(r"name\s+mean\s+sd", out)
    # The one row we have — an Intercept.
    @test occursin("Intercept[1]", out)

    # --- Hyperparameters section (we print internal scale) ---
    @test occursin("Hyperparameters", out)
    # Likelihood contributes exactly one (log-precision) hyperparameter.
    @test occursin("likelihood[1]", out)

    # --- section ordering ---
    idx_fe = findfirst("Fixed effects:", out)
    idx_hp = findfirst("Hyperparameters", out)
    @test idx_fe !== nothing && idx_hp !== nothing
    @test first(idx_fe) < first(idx_hp)
end

@testset "inla_summary — section order + random effects block appears" begin
    # Gaussian + Intercept + IID ⟹ dim(x) = n+1, one fixed effect, one
    # random-effect block, two hyperparameters.
    rng = Random.Xoshiro(20260424)
    n = 80
    x_true = 0.3 .* randn(rng, n)
    y = 0.2 .+ x_true .+ 0.4 .* randn(rng, n)

    c_int = Intercept()
    c_iid = IID(n)
    A = sparse([ones(n) Matrix{Float64}(I, n,n)])
    model = LatentGaussianModel(GaussianLikelihood(), (c_int, c_iid), A)
    res = inla(model, y; int_strategy=:grid)
    out = _capture_summary(model, res)

    @test occursin("Fixed effects:", out)
    @test occursin("Random effects", out)
    @test occursin("Hyperparameters", out)

    # Exactly one random-effect block ("IID[2]") with n entries.
    @test occursin("IID[2]", out)
    @test occursin(r"n=\s*80", out)

    # Order: INLA fit → Fixed → Random → Hyperparameters.
    i_header = findfirst("INLA fit", out)
    i_fe = findfirst("Fixed effects:", out)
    i_re = findfirst("Random effects", out)
    i_hp = findfirst("Hyperparameters", out)
    @test first(i_header) < first(i_fe) < first(i_re) < first(i_hp)
end

@testset "inla_summary — numeric columns match accessors" begin
    # The summary is just a render of fixed_effects/hyperparameters.
    # Lock that invariant: numeric values in the summary must be the
    # same values the public accessors return, formatted to 6 sig figs.
    rng = Random.Xoshiro(20260424)
    n = 100
    y = 0.5 .+ 0.4 .* randn(rng, n)

    model = LatentGaussianModel(GaussianLikelihood(), (Intercept(),),
        sparse(ones(n, 1)))
    res = inla(model, y; int_strategy=:grid)
    out = _capture_summary(model, res)

    fe = fixed_effects(model, res)
    # @sprintf("%.6g", mean) should appear verbatim in the output row.
    fe_mean_str = @sprintf("%.6g", fe[1].mean)
    @test occursin(fe_mean_str, out)

    hp = hyperparameters(model, res)
    hp_mean_str = @sprintf("%.6g", hp[1].mean)
    @test occursin(hp_mean_str, out)

    # log_marginal appears in the header.
    lm_str = @sprintf("%.6g", res.log_marginal)
    @test occursin(lm_str, out)
end

@testset "inla_summary — Base.show(MIME\"text/plain\") brief header" begin
    # `Base.show(io, MIME"text/plain", ::INLAResult)` renders a compact
    # one-screen overview that mentions the accessors. This is the
    # `REPL> res` experience.
    rng = Random.Xoshiro(7)
    n = 50
    y = randn(rng, n)
    model = LatentGaussianModel(GaussianLikelihood(), (Intercept(),),
        sparse(ones(n, 1)))
    res = inla(model, y; int_strategy=:grid)

    io = IOBuffer()
    show(io, MIME"text/plain"(), res)
    out = String(take!(io))

    @test startswith(out, "INLAResult")
    @test occursin(r"n_x\s+=\s+1", out)
    @test occursin(r"dim\(θ\)\s+=\s+1", out)
    @test occursin("log p(y)", out)
    # Hint points users at the structured accessors, not an ad-hoc dump.
    @test occursin("fixed_effects", out)
    @test occursin("random_effects", out)
    @test occursin("hyperparameters", out)
end

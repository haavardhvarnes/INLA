# Oracle test: classical (non-reparametrised) BYM on Scotland lip
# cancer vs R-INLA. Companion to test_scotland_bym2.jl that exercises
# the `model = "bym"` pathway with separate τ_v (iid) and τ_b (besag).
#
# Checked quantities:
#   - posterior means of fixed effects, 7% relative tolerance
#   - τ_b (Besag precision): Julia's point estimate (`exp(θ̂[2])`) must
#     fall inside R-INLA's 95% credible interval. A tighter point-to-
#     point comparison is deferred to v0.2 once an integrated θ-marginal
#     accessor lands (`replan-2026-04-28.md` Phase K) — see the comment
#     above the @testset for the Jacobian explanation.
#   - τ_v (IID precision) only sanity-checked: known weakly identified
#     in classical BYM, posterior is heavy-tailed.
#   - mlik: 2% relative tolerance (matches R-INLA's `extra()` for F_BYM).
#
# Fixture: scripts/generate-fixtures/lgm/scotland_bym.R.

include("load_fixture.jl")

using Test
using SparseArrays
using LinearAlgebra: I
using LatentGaussianModels: PoissonLikelihood, Intercept, FixedEffects,
    BYM, LatentGaussianModel, inla, PCPrecision,
    fixed_effects, hyperparameters, log_marginal_likelihood
using GMRFs: GMRFGraph

const BYM_FIXTURE = "scotland_bym"

const BYM_FIXED_EFFECT_TOL = 0.07
# Classical BYM is non-identified (Eberly & Carlin 2000): only τ_b/τ_v
# is constrained by data, posteriors on each are heavy-tailed. On
# Scotland (K=4 connected components: 53-node main + 3 island singletons)
# the τ_b summaries Julia and R-INLA expose are not the same statistic:
# Julia returns `exp(θ̂[2])`, the value at the mode of the *internal-
# scale* (log τ_b) marginal, while R-INLA's `summary.hyperpar`
# columns (`mode`, `0.5quant`, `mean`) are all summaries of the
# *user-scale* density `p(τ_b)`, related by the Jacobian
# `p(τ_b) = (1/τ_b) p_int(log τ_b)`. The Jacobian alone shifts a
# user-scale mode below `exp(mode_int)` by a factor near `exp(-σ²)`,
# so even a perfectly-correct Julia fit shows a 50–80 % gap on a
# direct number-to-number comparison. Until an integrated θ-marginal
# accessor (Martins et al. 2013 §3.3) lands in v0.2 (`plans/replan-
# 2026-04-28.md` Phase K), we assert only that Julia's point estimate
# falls inside R-INLA's 95 % credible interval — a bulletproof
# consistency check that does not depend on which summary statistic
# is being compared. Per-CC Sørbye-Rue scaling (Freni-Sterrantino
# et al. 2018) is implemented and is c-invariant under PCPrecision
# priors, so it doesn't shift τ_b on its own.
const BYM_MLIK_REL_TOL     = 0.02

_rel_bym(a, b) = abs(a - b) / max(abs(b), 1.0)

function _bym_row(frame, name::AbstractString, col::AbstractString)
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $(join(rn, "; ")))")
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

@testset "scotland_bym vs R-INLA" begin
    if !has_oracle_fixture(BYM_FIXTURE)
        @test_skip "oracle fixture $BYM_FIXTURE not generated (see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(BYM_FIXTURE)
        @test fx["name"] == BYM_FIXTURE

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate with the current R script"
        else
            inp = fx["input"]
            y = Int.(inp["cases"])
            E = Float64.(inp["expected"])
            x = Float64.(inp["x"])
            W = inp["W"]
            n = length(y)

            # Latent layout: [α; β; v; b]. Predictor η_i = α + β x_i + v_i + b_i.
            ℓ = PoissonLikelihood(; E = E)
            c_int  = Intercept()
            c_beta = FixedEffects(1)
            c_bym  = BYM(GMRFGraph(W);
                         hyperprior_iid   = PCPrecision(1.0, 0.01),
                         hyperprior_besag = PCPrecision(1.0, 0.01))
            A = sparse(hcat(
                ones(n),                        # intercept → α
                reshape(x, n, 1),               # AFF slope → β
                Matrix{Float64}(I, n, n),       # v_i contribution
                Matrix{Float64}(I, n, n),       # b_i contribution
            ))
            model = LatentGaussianModel(ℓ, (c_int, c_beta, c_bym), A)

            res = inla(model, y; int_strategy = :grid)

            # --- Fixed effects --------------------------------------------
            sf = fx["summary_fixed"]
            α_R = _bym_row(sf, "(Intercept)", "mean")
            β_R = _bym_row(sf, "x", "mean")
            fe = fixed_effects(model, res)
            @test length(fe) == 2
            @test _rel_bym(fe[1].mean, α_R) < BYM_FIXED_EFFECT_TOL
            @test _rel_bym(fe[2].mean, β_R) < BYM_FIXED_EFFECT_TOL

            # --- Hyperparameters ------------------------------------------
            # Internal θ = [log τ_v, log τ_b]. R-INLA reports user-scale
            # precisions. τ_v is weakly identified in classical BYM —
            # only sanity-check finiteness. For τ_b: Julia's point
            # estimate must fall inside R-INLA's 95 % credible interval.
            # See the BYM_TAU_B notes above for why a sharper assertion
            # is deferred to v0.2.
            sh = fx["summary_hyperpar"]
            τ_b_R_lo = _bym_row(sh, "Precision for region (spatial component)", "0.025quant")
            τ_b_R_hi = _bym_row(sh, "Precision for region (spatial component)", "0.975quant")
            τ_b_J = exp(res.θ̂[2])
            @test τ_b_R_lo ≤ τ_b_J ≤ τ_b_R_hi

            hp = hyperparameters(model, res)
            @test length(hp) == 2
            @test all(isfinite(r.mean) && r.sd > 0 for r in hp)

            # --- Marginal log-likelihood ----------------------------------
            mlik_R = Float64(fx["mlik"][1])
            mlik_J = log_marginal_likelihood(res)
            @test _rel_bym(mlik_J, mlik_R) < BYM_MLIK_REL_TOL
        end
    end
end

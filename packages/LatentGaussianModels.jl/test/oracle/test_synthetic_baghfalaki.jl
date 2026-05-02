# Oracle test: joint longitudinal-Gaussian + Weibull-PH survival regression
# on synthetic Baghfalaki et al. (2024)-style data vs R-INLA. Promotes the
# regression test [test/regression/test_inla_joint_baghfalaki.jl] to a
# proper oracle backed by an R-INLA fit on the *same dataset*. The
# data-generating process and R-INLA fit are produced by
# [scripts/generate-fixtures/lgm/synthetic_baghfalaki.R] and replayed here
# from `fx["input"]` so both fits see identical observations.
#
# Model:
#   b_i           ~ N(0, σ_b²)
#   y_{i,j}       ~ N(α_long + b_i, σ_g²)
#   T_i           ~ Weibull(α_w, exp(α_surv + φ · b_i))    (PH form)
# with right-censoring at T_admin. The Copy scaling β = φ ties the two
# arms together; β is the load-bearing Phase G PR3 / ADR-021 parameter.
#
# **Marginal log-likelihood**: kept as `@test isfinite(...)` only. The
# Baghfalaki joint inherits the polynomial-form Laplace mlik gap from the
# Weibull arm (see header of `test_synthetic_weibull_survival.jl` for the
# full chain — cubic-corrected Hessian + sample=0 evaluation in
# `GMRFLib`'s polynomial-form Laplace). Closure requires modifying
# `src/inference/laplace.jl`; deferred to v0.3. Fixed effects,
# hyperparameters, and random-effect posteriors agree tightly.

include("load_fixture.jl")

using Test
using SparseArrays
using Statistics: cor
using LatentGaussianModels: GaussianLikelihood, WeibullLikelihood,
                            Intercept, IID, LinearProjector, StackedMapping,
                            LatentGaussianModel, Copy, CopyTargetLikelihood,
                            inla, fixed_effects, random_effects, hyperparameters,
                            Censoring, NONE, RIGHT,
                            log_marginal_likelihood, n_likelihoods,
                            n_hyperparameters

const BAGH_FIXTURE = "synthetic_baghfalaki"

# Tolerances. Comparison is `exp(θ̂_J)` (user-scale value at the
# internal-scale mode) vs R-INLA's `mean` column (user-scale posterior
# mean); for asymmetric log-precision posteriors these differ
# systematically, hence widened bands on the precisions and shape.
# Fixed effects are symmetric Gaussians, so mean ≈ mode there.
const BAGH_FE_MEAN_TOL = 0.10   # |Δmean| / max(|R|, 1)
const BAGH_FE_SD_TOL = 0.20   # |Δsd|   / R-sd
const BAGH_HYPER_MEAN_TOL = 0.20   # |Δmean| / max(|R|, 1) for τ_g, α_w, τ_b
const BAGH_HYPER_SD_TOL = 0.40   # |Δsd|   / R-sd          for τ_g, α_w, τ_b
const BAGH_PHI_MEAN_TOL = 0.15   # |Δφ|    / max(|R|, 1)   (β-Copy, direct)
const BAGH_PHI_SD_TOL = 0.50   # |Δsd_φ| / R-sd-φ — Copy β posterior is
#                                       asymmetric; Hessian-derived width
#                                       runs ~40% wider than R-INLA's
#                                       polynomial-form Laplace estimate.
const BAGH_RE_COR_MIN = 0.99   # cor(b̂_J, b̂_R) — same data, same model

_rel_bagh(a, b) = abs(a - b) / max(abs(b), 1.0)

function _bagh_row(frame, name::AbstractString, col::AbstractString)
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $(join(rn, "; ")))")
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

@testset "synthetic_baghfalaki vs R-INLA" begin
    if !has_oracle_fixture(BAGH_FIXTURE)
        @test_skip "oracle fixture $BAGH_FIXTURE not generated " *
                   "(see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(BAGH_FIXTURE)
        @test fx["name"] == BAGH_FIXTURE
        @test haskey(fx, "summary_fixed")
        @test haskey(fx, "summary_hyperpar")
        @test haskey(fx, "summary_random")

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate"
        else
            inp = fx["input"]
            y_long = Float64.(inp["y_long"])
            subj_idx = Int.(inp["subj_long"])
            T_event = Float64.(inp["T_event"])
            ev_indicator = Int.(inp["ev_indicator"])

            n_long = length(y_long)
            N = length(T_event)
            @test length(subj_idx) == n_long
            @test length(ev_indicator) == N

            # Censoring vector: ev=1 → observed event (NONE), ev=0 → right-censored.
            cens = Censoring[ev == 1 ? NONE : RIGHT for ev in ev_indicator]

            # --- build the model -----------------------------------------
            # Latent layout: x = [α_long; α_surv; b_1, …, b_N].
            A_g = spzeros(Float64, n_long, 2 + N)
            for r in 1:n_long
                A_g[r, 1] = 1.0                # α_long
                A_g[r, 2 + subj_idx[r]] = 1.0  # b_{subj_idx[r]}
            end
            A_w = spzeros(Float64, N, 2 + N)
            for i in 1:N
                A_w[i, 2] = 1.0                # α_surv
            end

            mapping = StackedMapping(
                (LinearProjector(A_g), LinearProjector(A_w)),
                [1:n_long, (n_long + 1):(n_long + N)])

            ℓ_g = GaussianLikelihood()
            ℓ_w_base = WeibullLikelihood(censoring = cens)
            ℓ_w = CopyTargetLikelihood(
                ℓ_w_base,
                Copy(3:(2 + N); β_init = 1.0, fixed = false))
            model = LatentGaussianModel(
                (ℓ_g, ℓ_w),
                (Intercept(), Intercept(), IID(N)),
                mapping)

            @test n_likelihoods(model) == 2
            @test n_hyperparameters(model) == 4

            y = vcat(y_long, T_event)
            res = inla(model, y; int_strategy = :grid)

            # --- Fixed effects: posterior mean + sd ----------------------
            sf = fx["summary_fixed"]
            α_long_R = _bagh_row(sf, "intercept_long", "mean")
            α_surv_R = _bagh_row(sf, "intercept_surv", "mean")
            α_long_sd_R = _bagh_row(sf, "intercept_long", "sd")
            α_surv_sd_R = _bagh_row(sf, "intercept_surv", "sd")

            fe = fixed_effects(model, res)
            @test length(fe) == 2
            @test _rel_bagh(fe[1].mean, α_long_R) < BAGH_FE_MEAN_TOL
            @test _rel_bagh(fe[2].mean, α_surv_R) < BAGH_FE_MEAN_TOL
            @test abs(fe[1].sd - α_long_sd_R) / α_long_sd_R < BAGH_FE_SD_TOL
            @test abs(fe[2].sd - α_surv_sd_R) / α_surv_sd_R < BAGH_FE_SD_TOL

            # --- Hyperparameters -----------------------------------------
            # θ layout: [log τ_g, log α_w, φ, log τ_b] — Gauss precision,
            # Weibull shape, Copy β, IID precision.
            sh = fx["summary_hyperpar"]
            τ_g_R = _bagh_row(sh, "Precision for the Gaussian observations", "mean")
            α_w_R = _bagh_row(sh, "alpha parameter for weibullsurv[2]", "mean")
            τ_b_R = _bagh_row(sh, "Precision for b_long_idx", "mean")
            φ_R = _bagh_row(sh, "Beta for b_surv_idx", "mean")

            τ_g_sd_R = _bagh_row(sh, "Precision for the Gaussian observations", "sd")
            α_w_sd_R = _bagh_row(sh, "alpha parameter for weibullsurv[2]", "sd")
            τ_b_sd_R = _bagh_row(sh, "Precision for b_long_idx", "sd")
            φ_sd_R = _bagh_row(sh, "Beta for b_surv_idx", "sd")

            τ_g_J = exp(res.θ̂[1])
            α_w_J = exp(res.θ̂[2])
            φ_J = res.θ̂[3]                    # β-Copy already on user scale
            τ_b_J = exp(res.θ̂[4])

            # Delta-method sd: log-precision/log-shape: sd ≈ Σθ_k * exp(θ̂_k);
            # β-Copy is direct so sd ≈ sqrt(Σθ_3,3).
            τ_g_sd_J = sqrt(res.Σθ[1, 1]) * exp(res.θ̂[1])
            α_w_sd_J = sqrt(res.Σθ[2, 2]) * exp(res.θ̂[2])
            φ_sd_J = sqrt(res.Σθ[3, 3])
            τ_b_sd_J = sqrt(res.Σθ[4, 4]) * exp(res.θ̂[4])

            @test _rel_bagh(τ_g_J, τ_g_R) < BAGH_HYPER_MEAN_TOL
            @test _rel_bagh(α_w_J, α_w_R) < BAGH_HYPER_MEAN_TOL
            @test _rel_bagh(τ_b_J, τ_b_R) < BAGH_HYPER_MEAN_TOL
            @test _rel_bagh(φ_J, φ_R) < BAGH_PHI_MEAN_TOL

            @test abs(τ_g_sd_J - τ_g_sd_R) / τ_g_sd_R < BAGH_HYPER_SD_TOL
            @test abs(α_w_sd_J - α_w_sd_R) / α_w_sd_R < BAGH_HYPER_SD_TOL
            @test abs(τ_b_sd_J - τ_b_sd_R) / τ_b_sd_R < BAGH_HYPER_SD_TOL
            @test abs(φ_sd_J - φ_sd_R) / φ_sd_R < BAGH_PHI_SD_TOL

            hp = hyperparameters(model, res)
            @test length(hp) == 4

            # --- Random-effect agreement ---------------------------------
            # b̂_J vs b̂_R element-wise: same data, same model, both fits
            # should pin the same posterior mean per subject.
            re = random_effects(model, res)["IID[3]"]
            @test length(re.mean) == N
            b̂_R = Float64.(fx["summary_random"]["b_long_idx"]["mean"])
            @test length(b̂_R) == N
            @test cor(re.mean, b̂_R) > BAGH_RE_COR_MIN

            # --- Marginal log-likelihood ---------------------------------
            # Inherits the polynomial-form Laplace gap from the Weibull
            # arm (see header). Closure deferred to v0.3.
            @test isfinite(log_marginal_likelihood(res))
        end
    end
end

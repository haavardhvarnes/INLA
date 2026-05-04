# Oracle test: Student-t regression on synthetic data vs R-INLA.
#
# Smallest possible oracle for the `family = "T"` pathway —
# intercept + one covariate, no latent random effect, n = 200. Tests
# fixed-effect agreement, that the precision and dof modes are within
# a generous tolerance of R-INLA's marginal modes, and that the marginal
# log-likelihood matches within 5%.
#
# Fixture: scripts/generate-fixtures/lgm/synthetic_studentt.R.
# Skipped transparently if the JLD2 fixture has not been generated.
#
# Layout: identity projector (n × 2 design column-stack) + Intercept
# + FixedEffects(1) + StudentTLikelihood. Internal-scale hyperparameters
# `θ̂[1] = log τ`, `θ̂[2] = log(ν − 2)`; user scale via `exp` and
# `exp(·) + 2`.
#
# Hyperparameter tolerances. The marginal posterior on `log(ν − 2)` is
# strongly skewed at n = 200 (R-INLA reports `ν.sd ≈ 10.4` against
# `ν.mean ≈ 15.0`), and `(log τ, log(ν−2))` are strongly correlated.
# Julia's `θ̂` is the *joint* mode while R-INLA's `summary.hyperpar$mode`
# is per-marginal mode after grid integration — these differ by ≈25% on
# `τ` and ≈21% on `ν` here. The strongest end-to-end calibration check
# is the mlik agreement (5% tolerance), which passes at ~0.3%.

include("load_fixture.jl")

using Test
using SparseArrays
using LatentGaussianModels: StudentTLikelihood, Intercept, FixedEffects,
                            LatentGaussianModel, inla, fixed_effects, hyperparameters,
                            log_marginal_likelihood

const T_FIXTURE = "synthetic_studentt"

const T_FIXED_EFFECT_TOL = 0.05
# Joint-mode (Julia) vs marginal-mode (R-INLA) gap; documented above.
const T_TAU_REL_TOL = 0.30
const T_NU_REL_TOL = 0.35
const T_MLIK_REL_TOL = 0.05

_rel_t(a, b) = abs(a - b) / max(abs(b), 1.0)

function _t_row_value(frame, name::AbstractString, col::AbstractString)
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $(join(rn, "; ")))")
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

@testset "synthetic_studentt vs R-INLA" begin
    if !has_oracle_fixture(T_FIXTURE)
        @test_skip "oracle fixture $T_FIXTURE not generated (see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(T_FIXTURE)
        @test fx["name"] == T_FIXTURE
        @test haskey(fx, "summary_fixed")
        @test haskey(fx, "summary_hyperpar")

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate with the current R script"
        else
            inp = fx["input"]
            y = Float64.(inp["y"])
            x = Float64.(inp["x"])
            n = length(y)

            ℓ = StudentTLikelihood()
            c_int = Intercept()
            c_beta = FixedEffects(1)
            A = sparse(hcat(ones(n), reshape(x, n, 1)))
            model = LatentGaussianModel(ℓ, (c_int, c_beta), A)

            res = inla(model, y; int_strategy = :grid)

            # --- Fixed effects ------------------------------------------------
            sf = fx["summary_fixed"]
            α_R = _t_row_value(sf, "(Intercept)", "mean")
            β_R = _t_row_value(sf, "x", "mean")

            fe = fixed_effects(model, res)
            @test length(fe) == 2
            @test _rel_t(fe[1].mean, α_R) < T_FIXED_EFFECT_TOL
            @test _rel_t(fe[2].mean, β_R) < T_FIXED_EFFECT_TOL

            # --- Hyperparameters: τ and ν on user scale ----------------------
            # Compare Julia's joint mode (user scale) to R-INLA's marginal mode
            # with generous tolerances (see header comment).
            sh = fx["summary_hyperpar"]
            τ_R_mode = _t_row_value(sh, "precision for the student-t observations", "mode")
            ν_R_mode = _t_row_value(sh, "degrees of freedom for student-t", "mode")

            τ_J = exp(res.θ̂[1])
            ν_J = exp(res.θ̂[2]) + 2

            @test _rel_t(τ_J, τ_R_mode) < T_TAU_REL_TOL
            @test _rel_t(ν_J, ν_R_mode) < T_NU_REL_TOL

            hp = hyperparameters(model, res)
            @test length(hp) == 2
            @test all(h -> isfinite(h.mean) && h.sd > 0, hp)

            # --- Marginal log-likelihood --------------------------------------
            mlik_R = Float64(fx["mlik"][1])
            mlik_J = log_marginal_likelihood(res)
            @test _rel_t(mlik_J, mlik_R) < T_MLIK_REL_TOL
        end
    end
end

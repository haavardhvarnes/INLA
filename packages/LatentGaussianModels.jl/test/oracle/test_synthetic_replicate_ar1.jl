# Oracle test: replicated AR1 (`Replicate(AR1(n), R)`) on synthetic data
# vs R-INLA's `f(t, model="ar1", replicate=id)`. Smallest oracle for
# the `Replicate` component introduced in Phase I-C PR-3a.
#
# Fixture: scripts/generate-fixtures/lgm/synthetic_replicate_ar1.R.
# Skipped transparently if the JLD2 fixture has not been generated.
#
# Layout: identity projector + GaussianLikelihood + Replicate(AR1(n), R).
# The latent vector is the concatenation of the R per-panel chains
# `[x⁽¹⁾[1..n]; x⁽²⁾[1..n]; …; x⁽ᴿ⁾[1..n]]` of total length R·n;
# the same θ-block (log τ, atanh ρ) flows into every replicate's
# `precision_matrix` call. R-INLA's replicate machinery uses the same
# stacking convention, so the random-effect slot ordering matches one-
# to-one between the two implementations.
#
# Prior matching: the R-side script overrides R-INLA's built-in AR1
# default `prec=0.15` on `logit(ρ)` to `prec=0.25` on `logit(ρ)` so the
# prior is bit-for-bit identical to Julia's AR1 default
# `_NormalAR1ρ(0, σ=1)` on `atanh(ρ)` (note: `logit(ρ) = 2·atanh(ρ)`
# scales the precision by 4×).
#
# Comparison: Julia's `θ̂` is the mode of the hyperparameter posterior on
# the internal scale; we map it back to user scale (`exp` for precisions,
# `tanh` for ρ) and compare to R-INLA's `mode` column. With (R=30, n=20,
# 600 observations) all three hyperparameters are identifiable, but the
# τ_y posterior remains right-tailed (the AR1 latent can absorb residual
# variance), so the τ_y tolerance is wider than the AR1's own τ.

include("load_fixture.jl")

using Test
using SparseArrays
using LinearAlgebra
using LatentGaussianModels: GaussianLikelihood, AR1, Replicate,
                            LatentGaussianModel,
                            inla, hyperparameters,
                            log_marginal_likelihood

const REPL_AR1_FIXTURE = "synthetic_replicate_ar1"

# Tolerances. n=20 per panel, R=30 panels, N=600 observations.
# Comparison is mode-vs-mode (Julia θ̂ → user scale vs R-INLA `mode`).
# τ_y has the widest posterior because the AR1 latent absorbs noise; τ_x
# and ρ are tightly identified by the within-panel autocorrelation.
const REPL_AR1_TAU_Y_REL_TOL = 0.20
const REPL_AR1_TAU_X_REL_TOL = 0.10
const REPL_AR1_RHO_ABS_TOL = 0.05
const REPL_AR1_MLIK_REL_TOL = 0.02

_rel_repl_ar1(a, b) = abs(a - b) / max(abs(b), 1.0)

function _row_repl_ar1(frame, name::AbstractString, col::AbstractString)
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $(join(rn, "; ")))")
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

@testset "synthetic_replicate_ar1 vs R-INLA" begin
    if !has_oracle_fixture(REPL_AR1_FIXTURE)
        @test_skip "oracle fixture $REPL_AR1_FIXTURE not generated (see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(REPL_AR1_FIXTURE)
        @test fx["name"] == REPL_AR1_FIXTURE
        @test haskey(fx, "summary_hyperpar")

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate with the current R script"
        else
            inp = fx["input"]
            y = Float64.(inp["y"])
            n = Int(inp["n"])
            R = Int(inp["R"])
            N = n * R
            @test length(y) == N

            ℓ = GaussianLikelihood()
            comp = Replicate(AR1(n), R)
            A = sparse(1.0 * I, N, N)
            model = LatentGaussianModel(ℓ, (comp,), A)

            # dim(θ) = 3 (τ_y + τ_x + ρ) → :auto picks Grid (≤2 not met,
            # but Grid is the safe fallback for 3D and R-INLA also
            # defaults to the integration scheme that matches small-θ
            # hypercubes here).
            res = inla(model, y; int_strategy = :auto)

            # --- Hyperparameters: mode-vs-mode ---------------------------------
            sh = fx["summary_hyperpar"]
            τ_y_R = _row_repl_ar1(sh, "Precision for the Gaussian observations", "mode")
            τ_x_R = _row_repl_ar1(sh, "Precision for t_idx", "mode")
            ρ_R   = _row_repl_ar1(sh, "Rho for t_idx", "mode")

            # θ̂ ordering: likelihood hyperparams first, then component
            # hyperparams in component order:
            #   θ̂[1] = log τ_y  (Gaussian likelihood)
            #   θ̂[2] = log τ_x  (AR1 precision, shared across replicates)
            #   θ̂[3] = atanh ρ  (AR1 correlation, shared across replicates)
            τ_y_J = exp(res.θ̂[1])
            τ_x_J = exp(res.θ̂[2])
            ρ_J   = tanh(res.θ̂[3])

            @test _rel_repl_ar1(τ_y_J, τ_y_R) < REPL_AR1_TAU_Y_REL_TOL
            @test _rel_repl_ar1(τ_x_J, τ_x_R) < REPL_AR1_TAU_X_REL_TOL
            @test abs(ρ_J - ρ_R) < REPL_AR1_RHO_ABS_TOL

            hp = hyperparameters(model, res)
            @test length(hp) == 3

            # --- Marginal log-likelihood --------------------------------------
            # R-INLA's `mlik` has two entries: [Integration estimate,
            # Gaussian estimate]. Compare against the Integration estimate.
            mlik_J = log_marginal_likelihood(res)
            mlik_R = Float64(fx["mlik"][1])
            @test _rel_repl_ar1(mlik_J, mlik_R) < REPL_AR1_MLIK_REL_TOL
        end
    end
end

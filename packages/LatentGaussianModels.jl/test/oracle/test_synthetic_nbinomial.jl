# Oracle test: NegativeBinomial regression on synthetic data vs R-INLA.
#
# Smallest possible oracle for the `family = "nbinomial"` pathway —
# intercept + one covariate, no latent random effect, n = 200. Tests
# fixed-effect agreement and that the size hyperparameter mode is
# recovered to within 15%.
#
# Fixture: scripts/generate-fixtures/lgm/synthetic_nbinomial.R.
# Skipped transparently if the JLD2 fixture has not been generated.

include("load_fixture.jl")

using Test
using SparseArrays
using LatentGaussianModels: NegativeBinomialLikelihood, Intercept, FixedEffects,
    LatentGaussianModel, inla, fixed_effects, hyperparameters,
    log_marginal_likelihood

const NB_FIXTURE = "synthetic_nbinomial"

# Tolerances. Fixed-effects band is loose because n=200 leaves
# meaningful posterior mass; size hyperparameter is on log-scale and
# matched within 15% on the user scale (R-INLA reports `size` directly).
const NB_FIXED_EFFECT_TOL = 0.05
const NB_SIZE_REL_TOL     = 0.15
const NB_MLIK_REL_TOL     = 0.05

_rel_nb(a, b) = abs(a - b) / max(abs(b), 1.0)

function _row_value(frame, name::AbstractString, col::AbstractString)
    # `auto_unbox = TRUE` in the R-side jsonlite::write_json collapses
    # 1-element vectors to scalars; handle both forms.
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $(join(rn, "; ")))")
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

@testset "synthetic_nbinomial vs R-INLA" begin
    if !has_oracle_fixture(NB_FIXTURE)
        @test_skip "oracle fixture $NB_FIXTURE not generated (see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(NB_FIXTURE)
        @test fx["name"] == NB_FIXTURE
        @test haskey(fx, "summary_fixed")
        @test haskey(fx, "summary_hyperpar")

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate with the current R script"
        else
            inp = fx["input"]
            y = Int.(inp["y"])
            x = Float64.(inp["x"])
            n = length(y)

            ℓ = NegativeBinomialLikelihood()
            c_int  = Intercept()
            c_beta = FixedEffects(1)
            # Latent layout: [α; β]. Linear predictor η_i = α + β x_i.
            A = sparse(hcat(ones(n), reshape(x, n, 1)))
            model = LatentGaussianModel(ℓ, (c_int, c_beta), A)

            # dim(θ) = 1 (likelihood size only) → :auto picks Grid.
            res = inla(model, y; int_strategy = :grid)

            # --- Fixed effects ------------------------------------------------
            sf = fx["summary_fixed"]
            α_R = _row_value(sf, "(Intercept)", "mean")
            β_R = _row_value(sf, "x", "mean")

            fe = fixed_effects(model, res)
            @test length(fe) == 2
            @test _rel_nb(fe[1].mean, α_R) < NB_FIXED_EFFECT_TOL
            @test _rel_nb(fe[2].mean, β_R) < NB_FIXED_EFFECT_TOL

            # --- Hyperparameter: size on user scale ---------------------------
            # R-INLA's "size for the nbinomial observations" is reported on
            # the user scale; our internal scale is log(size).
            sh = fx["summary_hyperpar"]
            size_R = _row_value(sh, "size for the nbinomial observations (1/overdispersion)", "mean")
            size_J = exp(res.θ̂[1])
            @test _rel_nb(size_J, size_R) < NB_SIZE_REL_TOL

            hp = hyperparameters(model, res)
            @test length(hp) == 1
            @test isfinite(hp[1].mean) && hp[1].sd > 0

            # --- Marginal log-likelihood --------------------------------------
            mlik_R = Float64(fx["mlik"][1])
            mlik_J = log_marginal_likelihood(res)
            @test _rel_nb(mlik_J, mlik_R) < NB_MLIK_REL_TOL
        end
    end
end

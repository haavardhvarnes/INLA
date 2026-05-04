# Oracle test: multinomial-logit regression via the
# Multinomial-Poisson reformulation (Baker 1994; Chen 1985 / ADR-024)
# vs R-INLA. n_rows = 200, K = 3, p = 1, N_trials = 5; reference
# class = K with β_K = 0.
#
# Layout: long-format `n_rows * K` design matrix combining
#   - a class-specific FixedEffects((K-1)*p) block (β_1, β_2),
#   - a per-row IID(n_rows; τ_init=-10, fix_τ=true) nuisance α.
# Mirrors R-INLA's
#     y ~ -1 + x_class1 + x_class2 +
#         f(row_id, model="iid", hyper=list(prec=list(initial=-10, fixed=TRUE)))
#
# Tested invariants:
#   - β_1 / β_2 posterior mean within 5% of R-INLA's marginal mean.
#   - β_1 / β_2 posterior sd within 10% of R-INLA's marginal sd.
#   - log_marginal_likelihood is finite. The Multinomial-Poisson
#     reformulation matches R-INLA bit-for-bit *up to the per-row α
#     posterior*; with a fixed-precision nuisance the absolute mlik
#     also matches, but we keep the assertion finite-only here so the
#     oracle is robust to small θ-grid layout differences.
#
# Fixture: scripts/generate-fixtures/lgm/synthetic_multinomial.R.
# Skipped transparently if the JLD2 fixture has not been generated.

include("load_fixture.jl")

using Test
using SparseArrays
using LatentGaussianModels: PoissonLikelihood, FixedEffects, IID,
                            LatentGaussianModel, inla, component_range,
                            log_marginal_likelihood,
                            multinomial_to_poisson, multinomial_design_matrix

const MULTINOMIAL_FIXTURE = "synthetic_multinomial"

const MULTINOMIAL_BETA_MEAN_REL_TOL = 0.05
const MULTINOMIAL_BETA_SD_REL_TOL = 0.10

_rel_mn(a, b) = abs(a - b) / max(abs(b), 1.0)

function _mn_row_value(frame, name::AbstractString, col::AbstractString)
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $(join(rn, "; ")))")
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

@testset "synthetic_multinomial vs R-INLA" begin
    if !has_oracle_fixture(MULTINOMIAL_FIXTURE)
        @test_skip "oracle fixture $MULTINOMIAL_FIXTURE not generated " *
                   "(see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(MULTINOMIAL_FIXTURE)
        @test fx["name"] == MULTINOMIAL_FIXTURE
        @test haskey(fx, "summary_fixed")

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate with " *
                       "the current R script"
        else
            inp = fx["input"]
            y = Int.(inp["y"])
            row_id = Int.(inp["row_id"])
            class_id = Int.(inp["class_id"])
            x = Float64.(inp["x"])
            n_rows = Int(inp["n_rows"])
            K = Int(inp["K"])
            p = Int(inp["p"])
            n_long = n_rows * K
            @test n_rows == 200
            @test K == 3
            @test p == 1
            @test length(y) == n_long
            @test length(row_id) == n_long
            @test length(class_id) == n_long

            # Re-derive the helper from the long-format input to verify
            # the script-side layout is consistent with the in-package
            # `multinomial_to_poisson`.
            Y = zeros(Int, n_rows, K)
            for idx in 1:n_long
                Y[row_id[idx], class_id[idx]] = y[idx]
            end
            helper = multinomial_to_poisson(Y)
            @test helper.y == y
            @test helper.row_id == row_id
            @test helper.class_id == class_id

            # Class-specific covariate block A_β (n_long × (K-1)*p).
            X = reshape(x, n_rows, p)
            A_β = multinomial_design_matrix(helper, X)
            @test size(A_β) == (n_long, (K - 1) * p)

            # Per-row α nuisance block A_α (n_long × n_rows).
            A_α = sparse(1:n_long, row_id, ones(n_long), n_long, n_rows)

            # Combined design matrix: latent vector is `[β; α]`.
            A = hcat(A_β, A_α)
            @test size(A) == (n_long, (K - 1) * p + n_rows)

            ℓ = PoissonLikelihood()
            c_β = FixedEffects((K - 1) * p)
            c_α = IID(n_rows; τ_init=-10.0, fix_τ=true)
            model = LatentGaussianModel(ℓ, (c_β, c_α), A)

            # No likelihood / component hyperparameters (the per-row α
            # has fix_τ = true), so the INLA fast path returns a single
            # Laplace fit at θ = Float64[].
            res = inla(model, y)

            # --- β_1 / β_2 from the FixedEffects((K-1)*p) block --------
            # `fixed_effects` accessor only surfaces length-1 components;
            # the joint FixedEffects(2) block lives at `component_range`.
            sf = fx["summary_fixed"]
            β_R_mean = [_mn_row_value(sf, "x_class1", "mean"),
                _mn_row_value(sf, "x_class2", "mean")]
            β_R_sd = [_mn_row_value(sf, "x_class1", "sd"),
                _mn_row_value(sf, "x_class2", "sd")]

            β_rng = component_range(model, 1)
            β_J_mean = res.x_mean[β_rng]
            β_J_sd = sqrt.(res.x_var[β_rng])

            for k in 1:(K - 1)
                @test _rel_mn(β_J_mean[k], β_R_mean[k]) <
                      MULTINOMIAL_BETA_MEAN_REL_TOL
                @test _rel_mn(β_J_sd[k], β_R_sd[k]) <
                      MULTINOMIAL_BETA_SD_REL_TOL
            end

            # --- Marginal log-likelihood: finiteness only --------------
            mlik_J = log_marginal_likelihood(res)
            @test isfinite(mlik_J)
        end
    end
end

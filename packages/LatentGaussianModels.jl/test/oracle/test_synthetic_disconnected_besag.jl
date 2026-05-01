# Oracle test: Besag random-effect on a synthetic disconnected graph
# (K = 3 components) vs R-INLA. Validates the per-CC sum-to-zero
# machinery (Freni-Sterrantino et al. 2018) end-to-end.
#
# Fixture: scripts/generate-fixtures/lgm/synthetic_disconnected_besag.R.
# n = 12, K = 3 components of sizes (5, 4, 3).

include("load_fixture.jl")

using Test
using SparseArrays
using LinearAlgebra: I
using LatentGaussianModels: PoissonLikelihood, Intercept,
                            Besag, LatentGaussianModel, inla, PCPrecision,
                            fixed_effects, hyperparameters, random_effects,
                            log_marginal_likelihood
using GMRFs: GMRFGraph, nconnected_components

const DISC_FIXTURE = "synthetic_disconnected_besag"

const DISC_INTERCEPT_TOL = 0.10
const DISC_RE_SUM_TOL = 1.0e-6   # constraint should hold to machine precision

_rel_disc(a, b) = abs(a - b) / max(abs(b), 1.0)

function _disc_row(frame, name::AbstractString, col::AbstractString)
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && error("row '$name' not found (have: $(join(rn, "; ")))")
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

@testset "synthetic_disconnected_besag vs R-INLA" begin
    if !has_oracle_fixture(DISC_FIXTURE)
        @test_skip "oracle fixture $DISC_FIXTURE not generated (see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(DISC_FIXTURE)
        @test fx["name"] == DISC_FIXTURE

        if !haskey(fx, "input")
            @test_skip "fixture has no `input` field — regenerate with the current R script"
        else
            inp = fx["input"]
            y = Int.(inp["y"])
            W = inp["W"]
            n = length(y)

            graph = GMRFGraph(W)
            @test nconnected_components(graph) == 3

            # Latent layout: [α; b]. η_i = α + b_i.
            ℓ = PoissonLikelihood()
            c_int = Intercept()
            c_b = Besag(graph; hyperprior=PCPrecision(1.0, 0.01))
            A = sparse(hcat(ones(n), Matrix{Float64}(I, n, n)))
            model = LatentGaussianModel(ℓ, (c_int, c_b), A)

            res = inla(model, y; int_strategy=:grid)

            # --- Fixed effect: intercept ---------------------------------
            sf = fx["summary_fixed"]
            α_R = _disc_row(sf, "(Intercept)", "mean")
            fe = fixed_effects(model, res)
            @test length(fe) == 1
            @test _rel_disc(fe[1].mean, α_R) < DISC_INTERCEPT_TOL

            # --- Hyperparameter sanity ------------------------------------
            # Posterior on τ is extremely heavy-tailed at n = 12; only
            # check finiteness and positivity.
            hp = hyperparameters(model, res)
            @test length(hp) == 1
            @test isfinite(hp[1].mean) && hp[1].sd > 0

            # --- Per-CC sum-to-zero on the Besag posterior means ----------
            # The random-effect posterior must respect each CC's
            # sum-to-zero constraint. CC partition (1-indexed):
            #   CC 1: 1..5,  CC 2: 6..9,  CC 3: 10..12.
            re = random_effects(model, res)
            @test length(re) == 1
            # Besag is the 2nd component → key "Besag[2]".
            b_means = re["Besag[2]"].mean
            @test length(b_means) == n
            @test abs(sum(b_means[1:5])) < DISC_RE_SUM_TOL
            @test abs(sum(b_means[6:9])) < DISC_RE_SUM_TOL
            @test abs(sum(b_means[10:12])) < DISC_RE_SUM_TOL

            # mlik finite (precision too weakly identified for tight match).
            @test isfinite(log_marginal_likelihood(res))
        end
    end
end

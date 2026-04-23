# Oracle test: Scotland lip cancer BYM2 vs R-INLA.
#
# The MVP go/no-go milestone for the Julia-native LGM stack. Checked
# quantities (per plans/testing-strategy.md):
#   - posterior means of fixed effects, 1% relative tolerance
#   - posterior means of τ and φ (hyperparameters), 5% relative tolerance
#   - marginal log-likelihood, 1% relative tolerance
#
# Fixture: scripts/generate-fixtures/lgm/scotland_bym2.R.
# Skipped transparently if the JLD2 fixture has not been generated.
#
# NOTE: this test is currently `@test_skip`'d pending the INLA outer
# loop (M4 milestone). Once `fit(model, y, INLA())` is functional, fill
# in the Julia-side fit and activate the comparisons below.

include("load_fixture.jl")

using Test

const FIXTURE = "scotland_bym2"

@testset "scotland_bym2 vs R-INLA" begin
    if !has_oracle_fixture(FIXTURE)
        @test_skip "oracle fixture $FIXTURE not generated (see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(FIXTURE)
        @test fx["name"] == FIXTURE
        @test haskey(fx, "summary_fixed")
        @test haskey(fx, "summary_hyperpar")

        # Sanity on the R side: intercept and AFF coefficient exist.
        sf = fx["summary_fixed"]
        rn = String.(sf["rownames"])
        @test "(Intercept)" in rn
        @test "x" in rn

        @test_skip "Julia INLA fit comparison — pending M4 (outer loop)."
    end
end

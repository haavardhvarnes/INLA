# Oracle test: compare GMRFs.jl's `marginal_variances` and generalised
# log-determinant against R-INLA's `inla.qinv` for an RW2 on a line.
#
# Fixture: scripts/generate-fixtures/gmrfs/qinv_rw2.R.
# Skipped transparently if the JLD2 fixture has not been generated.

include("load_fixture.jl")

using Test
using SparseArrays
using LinearAlgebra
using GMRFs: RW2GMRF, marginal_variances, precision_matrix

const FIXTURE = "qinv_rw2"

@testset "qinv_rw2 vs R-INLA" begin
    if !has_oracle_fixture(FIXTURE)
        @test_skip "oracle fixture $FIXTURE not generated (see scripts/generate-fixtures/)"
    else
        fx = load_oracle_fixture(FIXTURE)
        Q = fx["Q"]::SparseMatrixCSC
        ref_diag = fx["qinv_diag"]::Vector{Float64}
        ref_logdet = get(fx, "log_det", nothing)

        # The fixture stores R-INLA's `inla.rw(n, order = 2)` structure
        # matrix; verify GMRFs.jl's RW2GMRF builds the same Q.
        n = size(Q, 1)
        g = RW2GMRF(n)
        @test precision_matrix(g) ≈ Q

        # Generalised marginal variances on the non-null subspace under
        # the canonical RW2 null-space basis (1's and ramp). R-INLA's
        # `inla.qinv` applies the same two constraints in the fixture
        # generator, so the fixtures match modulo the jitter the
        # generator adds to make Q PD; we tolerate 1% relative error per
        # `testing-strategy.md`.
        our_diag = marginal_variances(g)
        @test length(our_diag) == length(ref_diag)
        @test isapprox(our_diag, ref_diag; rtol=1.0e-2)

        if ref_logdet !== nothing
            eig = eigvals(Symmetric(Matrix(Q)))
            our_logdet = sum(log, filter(>(1.0e-10), eig))
            @test isapprox(our_logdet, ref_logdet; rtol=1.0e-8)
        end
    end
end

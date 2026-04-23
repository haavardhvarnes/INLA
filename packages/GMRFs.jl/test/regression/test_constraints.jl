using Test
using GMRFs
using Graphs
using SparseArrays

@testset "AbstractConstraint basics" begin
    A = [1.0 1.0 1.0]
    c = LinearConstraint(A)                      # homogeneous defaults
    @test nconstraints(c) == 1
    @test constraint_matrix(c) === A
    @test constraint_rhs(c) == [0.0]

    e = [2.5]
    c2 = LinearConstraint(A, e)
    @test constraint_rhs(c2) === e

    # rhs length mismatch
    @test_throws DimensionMismatch LinearConstraint([1.0 1; 1 1], [0.0])
end

@testset "constraints(::AbstractGMRF) defaults" begin
    @test constraints(IIDGMRF(3)) isa NoConstraint
    @test constraints(AR1GMRF(5; ρ = 0.3)) isa NoConstraint

    rc = constraints(RW1GMRF(5))
    @test rc isa LinearConstraint
    @test nconstraints(rc) == 1
    @test constraint_matrix(rc) == ones(Float64, 1, 5)

    rc2_open = constraints(RW2GMRF(5))
    @test nconstraints(rc2_open) == 2
    @test constraint_matrix(rc2_open)[1, :] == ones(5)
    @test constraint_matrix(rc2_open)[2, :] == 1:5

    rc2_cyc = constraints(RW2GMRF(5; cyclic = true))
    @test nconstraints(rc2_cyc) == 1
end

@testset "Per-component sum-to-zero on disconnected Besag" begin
    W = spzeros(Int, 5, 5)
    W[1, 2] = W[2, 1] = 1
    W[4, 5] = W[5, 4] = 1
    g = GMRFGraph(W)
    cons = sum_to_zero_constraints(g)
    @test nconstraints(cons) == 3   # nodes: {1,2}, {3}, {4,5}
    A = constraint_matrix(cons)
    # node 3 is isolated — its own component
    @test sum(A; dims = 2) == [2.0; 1.0; 2.0;;]
end

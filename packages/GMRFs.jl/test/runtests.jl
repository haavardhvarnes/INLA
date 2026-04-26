using Test
using GMRFs

@testset "GMRFs.jl" begin
    @testset "Regression" begin
        include("regression/test_graph.jl")
        include("regression/test_iid.jl")
        include("regression/test_rw.jl")
        include("regression/test_ar1.jl")
        include("regression/test_seasonal.jl")
        include("regression/test_besag.jl")
        include("regression/test_generic0.jl")
        include("regression/test_constraints.jl")
        include("regression/test_sampling_covariance.jl")
        include("regression/test_logdet_dense.jl")
        include("regression/test_marginals.jl")
        include("regression/test_factorization.jl")
    end
    @testset "Oracle (R-INLA)" begin
        include("oracle/test_qinv_rw2.jl")
    end
    @testset "Quality" begin
        include("quality/test_aqua.jl")
        include("quality/test_jet.jl")
    end
end

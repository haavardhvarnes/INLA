using Test
using GMRFs

@testset "GMRFs.jl" begin
    @testset "Regression" begin
        include("regression/test_graph.jl")
        include("regression/test_iid.jl")
        include("regression/test_rw.jl")
        include("regression/test_ar1.jl")
        include("regression/test_besag.jl")
        include("regression/test_generic0.jl")
        include("regression/test_constraints.jl")
        include("regression/test_sampling_covariance.jl")
        include("regression/test_logdet_dense.jl")
    end
end

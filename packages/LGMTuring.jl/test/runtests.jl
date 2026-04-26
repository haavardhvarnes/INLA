using Test

@testset "LGMTuring.jl" begin
    @testset "regression" begin
        include("regression/test_logdensity.jl")
        include("regression/test_nuts_sample.jl")
        include("regression/test_compare.jl")
    end
    @testset "triangulation" begin
        include("triangulation/test_scotland_bym2.jl")
    end
end

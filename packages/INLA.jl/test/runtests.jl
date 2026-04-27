using Test
using INLA

@testset "INLA umbrella" begin
    @testset "re-exports" begin
        @test isdefined(INLA, :GMRFs)
        @test isdefined(INLA, :LatentGaussianModels)
        @test isdefined(INLA, :INLASPDE)
    end

    @testset "key symbols in scope" begin
        @test isdefined(INLA, :LatentGaussianModel)
        @test isdefined(INLA, :SPDE2)
    end
end

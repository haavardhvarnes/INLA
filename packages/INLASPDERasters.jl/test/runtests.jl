using Test
using INLASPDERasters

@testset "INLASPDERasters.jl" begin
    @testset "smoke — module loads" begin
        @test isdefined(INLASPDERasters, :INLASPDERasters)
        @test isdefined(INLASPDERasters, :extract_at_mesh)
    end

    @testset "M1 — extraction" begin
        include("regression/test_extract_synthetic.jl")
    end
end

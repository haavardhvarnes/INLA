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

    @testset "M2 — prediction" begin
        include("regression/test_predict_synthetic.jl")
    end

    @testset "M3 — uncertainty surfaces" begin
        include("regression/test_quantile_rasters.jl")
    end

    @testset "Quality" begin
        include("quality/test_aqua.jl")
        include("quality/test_jet.jl")
    end
end

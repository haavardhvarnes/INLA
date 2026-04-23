using Test
using LinearAlgebra
using SparseArrays
using INLASPDE

@testset "INLASPDE.jl" begin
    @testset "smoke — module loads" begin
        # Scaffold check: package precompiles and the public module is
        # importable. Milestone testsets are added as M1..M4 land.
        @test isdefined(INLASPDE, :INLASPDE)
    end

    @testset "M1 — FEM assembly" begin
        include("regression/test_fem_small_mesh.jl")
        include("regression/test_mass_lumping.jl")
        include("regression/test_precision_alpha1.jl")
        include("regression/test_precision_alpha2.jl")
        include("regression/test_matern_reproduction.jl")
    end

    # --- future milestone testsets ---------------------------------------
    # @testset "M2 — SPDE2 + PC-Matérn" begin
    #     include("regression/test_spde2_component.jl")
    #     include("regression/test_pc_matern_prior.jl")
    # end
    # @testset "M3 — Mesh generation" begin
    #     include("regression/test_mesh_quality.jl")
    # end
    # @testset "M4 — Projector" begin
    #     include("regression/test_projector.jl")
    # end
    # @testset "M5 — Meuse oracle" begin
    #     include("oracle/test_meuse_spde.jl")
    # end
end

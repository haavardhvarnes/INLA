# JET.jl static analysis. Scoped to the package's own module — JET's
# `target_modules` keeps reports inside `LatentGaussianModels`, so issues
# from upstream packages don't pollute the report.

using JET
using LatentGaussianModels
using Test

@testset "JET — LatentGaussianModels" begin
    rep = JET.report_package(LatentGaussianModels;
                              target_modules = (LatentGaussianModels,),
                              ignore_missing_comparison = true)
    @test isempty(JET.get_reports(rep))
end

# JET.jl static analysis. Scoped to the package's own module — JET's
# `target_modules` keeps reports inside `INLASPDERasters`, so issues
# from upstream packages don't pollute the report.

using JET
using INLASPDERasters
using Test

@testset "JET — INLASPDERasters" begin
    rep = JET.report_package(INLASPDERasters;
                              target_modules = (INLASPDERasters,),
                              ignore_missing_comparison = true)
    @test isempty(JET.get_reports(rep))
end

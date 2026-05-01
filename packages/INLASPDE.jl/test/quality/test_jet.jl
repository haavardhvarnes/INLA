# JET.jl static analysis. Scoped to the package's own module — JET's
# `target_modules` keeps reports inside `INLASPDE`, so issues from
# upstream packages don't pollute the report.

using JET
using INLASPDE
using Test

@testset "JET — INLASPDE" begin
    rep = JET.report_package(INLASPDE;
        target_modules=(INLASPDE,),
        ignore_missing_comparison=true)
    @test isempty(JET.get_reports(rep))
end

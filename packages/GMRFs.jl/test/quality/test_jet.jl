# JET.jl static analysis. Scoped to the package's own module — JET's
# `target_modules` keeps reports inside `GMRFs`, so issues from upstream
# packages don't pollute the report.
#
# We use `report_package` rather than `report_call` because v0.1 wants
# a top-level "no errors lurking" assertion, not a hot-path-specific
# audit; that audit comes in Phase E hot-path benchmarks.

using JET
using GMRFs
using Test

@testset "JET — GMRFs" begin
    rep = JET.report_package(GMRFs;
        target_modules=(GMRFs,),
        ignore_missing_comparison=true)
    @test isempty(JET.get_reports(rep))
end

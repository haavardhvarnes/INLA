using Aqua
using GMRFs
using Test

# `ambiguities = false` in `test_all` because the ambiguity scan walks
# transitive deps (LinearSolve, Distributions, …) where we cannot fix
# upstream issues. The own-module ambiguity scan immediately afterwards
# is what locks down GMRFs's surface.
@testset "Aqua — GMRFs" begin
    Aqua.test_all(GMRFs;
                   ambiguities = false,
                   piracies    = (treat_as_own = [],),
                   stale_deps  = true,
                   deps_compat = (check_extras = false,))
    @testset "ambiguities (own package only)" begin
        Aqua.test_ambiguities(GMRFs)
    end
end

using Aqua
using INLASPDE
using Test

# `ambiguities = false` in `test_all` because the ambiguity scan walks
# transitive deps (Meshes, DelaunayTriangulation, …) where we cannot fix
# upstream issues. The own-module ambiguity scan immediately afterwards
# is what locks down INLASPDE's surface.
@testset "Aqua — INLASPDE" begin
    Aqua.test_all(INLASPDE;
        ambiguities=false,
        piracies=(treat_as_own=[],),
        stale_deps=true,
        deps_compat=(check_extras=false,))
    @testset "ambiguities (own package only)" begin
        Aqua.test_ambiguities(INLASPDE)
    end
end

using Aqua
using LatentGaussianModels
using Test

# `ambiguities = false` in `test_all` because the ambiguity scan walks
# transitive deps (NonlinearSolve, Optimization, …) where we cannot fix
# upstream issues. The own-module ambiguity scan immediately afterwards
# is what locks down LatentGaussianModels's surface.
@testset "Aqua — LatentGaussianModels" begin
    Aqua.test_all(LatentGaussianModels;
                   ambiguities = false,
                   piracies    = (treat_as_own = [],),
                   stale_deps  = true,
                   deps_compat = (check_extras = false,))
    @testset "ambiguities (own package only)" begin
        Aqua.test_ambiguities(LatentGaussianModels)
    end
end

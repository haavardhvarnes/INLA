using Aqua
using INLASPDERasters
using Test

# `ambiguities = false` in `test_all` because the ambiguity scan walks
# transitive deps (Rasters, GDAL_jll, …) where we cannot fix upstream
# issues. The own-module ambiguity scan immediately afterwards is what
# locks down INLASPDERasters's surface.
#
# `stale_deps = false` because GMRFs and LatentGaussianModels appear in
# `[deps]` only as `[sources]` anchors so Pkg can resolve transitive
# path-linked siblings — they are deliberately not `using`'d here. See
# the comment in this package's Project.toml.
@testset "Aqua — INLASPDERasters" begin
    Aqua.test_all(INLASPDERasters;
        ambiguities=false,
        piracies=(treat_as_own=[],),
        stale_deps=false,
        deps_compat=(check_extras=false,))
    @testset "ambiguities (own package only)" begin
        Aqua.test_ambiguities(INLASPDERasters)
    end
end

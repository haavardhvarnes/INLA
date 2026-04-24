"""
    INLASPDERasters

Raster glue for [`INLASPDE`](@ref): extract covariate values from a
`Rasters.Raster` at mesh vertices. Prediction-to-raster and uncertainty
surfaces (M2, M3) will land here as separate milestones.

See `plans/plan.md` for the package roadmap and `CLAUDE.md` for scope
and style rules. This is not a standalone package: it depends on
`INLASPDE` and is only meaningful in conjunction with a fitted SPDE
model.
"""
module INLASPDERasters

using INLASPDE: INLAMesh, num_vertices
using Rasters: Rasters, Raster, X, Y

include("extract.jl")

export extract_at_mesh

end # module

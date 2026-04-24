# Reference fmesher mesh on an L-shaped (concave) boundary. Second of
# three boundaries for the ADR-007 M3 parity gate. A concave polygon
# exercises DelaunayTriangulation's handling of reflex vertices.
#
# Output: fixtures/spde/fmesher_lshape.json
# (→ packages/INLASPDE.jl/test/oracle/fixtures/fmesher_lshape.jld2)

here <- tryCatch(
    normalizePath(dirname(sys.frame(1)$ofile), mustWork = FALSE),
    error = function(e) getwd()
)
if (!nzchar(here)) here <- getwd()
source(file.path(here, "..", "_helpers.R"))

suppressPackageStartupMessages({
    library(fmesher)
})

set.seed(20260424)

# L-shape: [0,2]×[0,1] ∪ [0,1]×[0,2]
boundary <- rbind(
    c(0, 0), c(2, 0), c(2, 1), c(1, 1), c(1, 2), c(0, 2)
)
params <- list(
    max_edge  = 0.25,
    min_angle = 25.0,
    cutoff    = 0.02,
    offset    = 0.0
)

mesh <- fmesher::fm_mesh_2d_inla(
    boundary  = fm_segm(boundary, is.bnd = TRUE),
    max.edge  = params$max_edge,
    min.angle = params$min_angle,
    cutoff    = params$cutoff,
    offset    = params$offset
)

out_path <- file.path(here, "..", "fixtures", "spde", "fmesher_lshape.json")

write_fmesher_fixture(
    path = out_path,
    name = "fmesher_lshape",
    boundary = boundary,
    params = params,
    mesh = mesh,
    meta = list(
        description = "fmesher reference mesh on an L-shaped concave boundary",
        fm_function = "fm_mesh_2d_inla"
    )
)

cat("wrote", out_path, "\n")

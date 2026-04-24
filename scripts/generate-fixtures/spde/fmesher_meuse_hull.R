# Reference fmesher mesh on the convex hull of the Meuse observation
# locations. Third of three boundaries for the ADR-007 M3 parity gate;
# stands in for the "NC-coastline subset" in plan.md because NC coast
# data is not vendored and the convex-hull case is the more faithful
# proxy for the kind of irregular real-world polygon the parity gate
# is meant to catch.
#
# Output: fixtures/spde/fmesher_meuse_hull.json
# (→ packages/INLASPDE.jl/test/oracle/fixtures/fmesher_meuse_hull.jld2)

here <- tryCatch(
    normalizePath(dirname(sys.frame(1)$ofile), mustWork = FALSE),
    error = function(e) getwd()
)
if (!nzchar(here)) here <- getwd()
source(file.path(here, "..", "_helpers.R"))

suppressPackageStartupMessages({
    library(fmesher)
    library(sp)
    library(grDevices)
})

set.seed(20260424)

data(meuse)
loc <- as.matrix(meuse[, c("x", "y")]) / 1000.0   # metres → kilometres

# `chull` returns vertices in clockwise order; Julia's
# `DelaunayTriangulation.jl` requires counter-clockwise boundary
# orientation, so reverse here. fmesher accepts either.
hull_idx <- rev(grDevices::chull(loc))
boundary <- loc[hull_idx, , drop = FALSE]

params <- list(
    max_edge  = 0.25,
    min_angle = 25.0,
    cutoff    = 0.05,
    offset    = 0.0
)

mesh <- fmesher::fm_mesh_2d_inla(
    boundary  = fm_segm(boundary, is.bnd = TRUE),
    max.edge  = params$max_edge,
    min.angle = params$min_angle,
    cutoff    = params$cutoff,
    offset    = params$offset
)

out_path <- file.path(here, "..", "fixtures", "spde", "fmesher_meuse_hull.json")

write_fmesher_fixture(
    path = out_path,
    name = "fmesher_meuse_hull",
    boundary = boundary,
    params = params,
    mesh = mesh,
    meta = list(
        description = "fmesher reference mesh on the convex hull of Meuse locations (in km)",
        fm_function = "fm_mesh_2d_inla",
        dataset = "sp::meuse",
        coordinate_unit = "km"
    )
)

cat("wrote", out_path, "\n")

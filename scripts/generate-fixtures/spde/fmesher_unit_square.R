# Reference fmesher mesh on the unit square. One of three boundaries
# for the ADR-007 M3 parity gate:
#   |n_vertices_J - n_vertices_R| / n_vertices_R ≤ 0.05
#   min_angle_J ≥ max(20°, 0.95 · min_angle_R)
#   max_edge_J ≤ 1.05 · max_edge_R
#
# Output: fixtures/spde/fmesher_unit_square.json
# (→ packages/INLASPDE.jl/test/oracle/fixtures/fmesher_unit_square.jld2)

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

boundary <- rbind(c(0, 0), c(1, 0), c(1, 1), c(0, 1))
params <- list(
    max_edge  = 0.15,
    min_angle = 25.0,
    cutoff    = 0.01,
    offset    = 0.0
)

mesh <- fmesher::fm_mesh_2d_inla(
    boundary  = fm_segm(boundary, is.bnd = TRUE),
    max.edge  = params$max_edge,
    min.angle = params$min_angle,
    cutoff    = params$cutoff,
    offset    = params$offset
)

out_path <- file.path(here, "..", "fixtures", "spde", "fmesher_unit_square.json")

write_fmesher_fixture(
    path = out_path,
    name = "fmesher_unit_square",
    boundary = boundary,
    params = params,
    mesh = mesh,
    meta = list(
        description = "fmesher reference mesh on the unit square",
        fm_function = "fm_mesh_2d_inla"
    )
)

cat("wrote", out_path, "\n")

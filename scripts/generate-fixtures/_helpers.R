# Shared R helpers for generating R-INLA oracle fixtures.
#
# R scripts call these to dump fit data to an intermediate JSON file.
# A companion Julia script (`convert_to_jld2.jl`) then reads the JSON
# and writes the JLD2 fixture into the destination package's
# test/oracle/fixtures/ directory.
#
# Dependencies:
#   jsonlite, Matrix (for sparse output), INLA.

suppressPackageStartupMessages({
    library(jsonlite)
    library(Matrix)
})

# ------------------------------------------------------------------
# Low-level: sparse matrix to triplet list
# ------------------------------------------------------------------

sparse_to_triplet <- function(M) {
    # Force general (non-symmetric-storage) form before triplet
    # conversion. Matrix::Matrix on a symmetric dense matrix can pick
    # dsCMatrix, which stores only one triangle; downstream consumers
    # (e.g. Julia `sparse`) expect both halves to be explicit.
    M <- as(M, "generalMatrix")
    M <- as(M, "TsparseMatrix")
    list(
        i = as.integer(M@i) + 1L,   # 1-based
        j = as.integer(M@j) + 1L,
        v = as.numeric(M@x),
        nrow = as.integer(nrow(M)),
        ncol = as.integer(ncol(M))
    )
}

# ------------------------------------------------------------------
# Marginal: R-INLA posterior marginal is a 2-column matrix (x, y).
# Always returned as a named list so JSON stays structural.
# ------------------------------------------------------------------

marginal_to_list <- function(m) {
    if (is.null(m)) return(NULL)
    list(x = as.numeric(m[, "x"]), y = as.numeric(m[, "y"]))
}

# ------------------------------------------------------------------
# Summary frames: named list for each row
# ------------------------------------------------------------------

summary_frame_to_list <- function(df) {
    if (is.null(df) || nrow(df) == 0) return(list())
    out <- as.list(df)
    out$rownames <- rownames(df)
    out
}

# ------------------------------------------------------------------
# Write a full R-INLA fit summary to JSON.
# ------------------------------------------------------------------

write_inla_fixture <- function(fit, path, name,
                               component_names = character(0),
                               include_marginals = TRUE,
                               meta = list()) {
    fixture <- list(
        name = name,
        inla_version = as.character(packageVersion("INLA")),
        cpu_used = as.numeric(fit$cpu.used),
        summary_fixed = summary_frame_to_list(fit$summary.fixed),
        summary_hyperpar = summary_frame_to_list(fit$summary.hyperpar),
        summary_random = lapply(
            setNames(component_names, component_names),
            function(nm) summary_frame_to_list(fit$summary.random[[nm]])
        ),
        mlik = as.numeric(fit$mlik),
        meta = meta
    )
    if (include_marginals) {
        fixture$marginals_fixed <- lapply(fit$marginals.fixed, marginal_to_list)
        fixture$marginals_hyperpar <- lapply(fit$marginals.hyperpar, marginal_to_list)
        fixture$marginals_random <- lapply(
            setNames(component_names, component_names),
            function(nm) {
                mr <- fit$marginals.random[[nm]]
                if (is.null(mr)) return(NULL)
                lapply(mr, marginal_to_list)
            }
        )
    }
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
    write_json(
        fixture, path,
        auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
    )
    invisible(fixture)
}

# ------------------------------------------------------------------
# Mesh: serialise an fmesher/inla mesh into a JSON-friendly list.
# ------------------------------------------------------------------

mesh_to_list <- function(mesh) {
    # fmesher loc is n × 3 (x, y, z = 0 for planar meshes); keep 2D.
    loc <- as.matrix(mesh$loc[, 1:2, drop = FALSE])
    tv <- as.matrix(mesh$graph$tv)   # n_tri × 3, 1-based indices
    # Min triangle angle (degrees) and max edge length, computed on
    # the mesh we're about to ship. Both sides of the parity test
    # compare Julia-side recomputations to these numbers, so keep the
    # formula explicit here.
    P1 <- loc[tv[, 1], , drop = FALSE]
    P2 <- loc[tv[, 2], , drop = FALSE]
    P3 <- loc[tv[, 3], , drop = FALSE]
    e12 <- sqrt(rowSums((P2 - P1)^2))
    e23 <- sqrt(rowSums((P3 - P2)^2))
    e31 <- sqrt(rowSums((P1 - P3)^2))
    # Law of cosines for the three triangle angles.
    ang <- function(a, b, c) {
        acos(pmin(pmax((b^2 + c^2 - a^2) / (2 * b * c), -1), 1))
    }
    a1 <- ang(e23, e31, e12)
    a2 <- ang(e31, e12, e23)
    a3 <- ang(e12, e23, e31)
    min_angle_deg <- (180 / pi) * min(a1, a2, a3)
    max_edge <- max(e12, e23, e31)
    list(
        loc = lapply(seq_len(nrow(loc)), function(i) as.numeric(loc[i, ])),
        tv  = lapply(seq_len(nrow(tv)),  function(i) as.integer(tv[i, ])),
        n_vertices = as.integer(nrow(loc)),
        n_triangles = as.integer(nrow(tv)),
        min_angle_deg = as.numeric(min_angle_deg),
        max_edge = as.numeric(max_edge)
    )
}

# Serialise a boundary polygon (rectangular matrix or two-column frame)
# as a list of 2-vectors, mirroring the shape used in INLASPDE.jl.
boundary_to_list <- function(poly) {
    poly <- as.matrix(poly[, 1:2, drop = FALSE])
    lapply(seq_len(nrow(poly)), function(i) as.numeric(poly[i, ]))
}

# ------------------------------------------------------------------
# fmesher parity: boundary + params + fmesher output stats/mesh.
# ------------------------------------------------------------------

write_fmesher_fixture <- function(path, name, boundary, params, mesh,
                                  meta = list()) {
    fixture <- list(
        name = name,
        inla_version = as.character(packageVersion("INLA")),
        fmesher_version = as.character(packageVersion("fmesher")),
        boundary = boundary_to_list(boundary),
        params = params,
        mesh = mesh_to_list(mesh),
        meta = meta
    )
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
    write_json(
        fixture, path,
        auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
    )
    invisible(fixture)
}

# ------------------------------------------------------------------
# For GMRFs.jl: write Q, log-determinant, and qinv diagonal for a
# known graph. No full INLA fit required — just numerical references.
# ------------------------------------------------------------------

write_gmrf_fixture <- function(Q, path, name,
                               qinv_diag = NULL,
                               log_det = NULL,
                               meta = list()) {
    fixture <- list(
        name = name,
        inla_version = as.character(packageVersion("INLA")),
        Q = sparse_to_triplet(Q),
        qinv_diag = if (is.null(qinv_diag)) NULL else as.numeric(qinv_diag),
        log_det = if (is.null(log_det)) NULL else as.numeric(log_det),
        meta = meta
    )
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
    write_json(
        fixture, path,
        auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
    )
    invisible(fixture)
}

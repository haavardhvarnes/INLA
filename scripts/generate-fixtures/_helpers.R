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

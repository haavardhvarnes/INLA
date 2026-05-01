# Reference: RW2 precision on a uniform line, with selected diagonal of
# Q^{-1} from R-INLA's `inla.qinv`. Used as an oracle for GMRFs.jl's
# `marginal_variances` implementation and log-determinant computation.
#
# Model: RW2 on n = 50 nodes with sum-to-zero constraint (default in
# INLA for intrinsic models).
#
# Output: fixtures/gmrfs/qinv_rw2.json (converted via convert_to_jld2.jl
# to packages/GMRFs.jl/test/oracle/fixtures/qinv_rw2.jld2).

here <- tryCatch(
    normalizePath(dirname(sys.frame(1)$ofile), mustWork = FALSE),
    error = function(e) getwd()
)
if (!nzchar(here)) here <- getwd()
source(file.path(here, "..", "_helpers.R"))
suppressPackageStartupMessages(library(INLA))

set.seed(20260423)

n <- 50L
# Build RW2 structure matrix (intrinsic, rank n - 2). `inla.rw` was
# unexported in newer R-INLA versions, so reach it via the triple-colon
# operator. The function body is unchanged, so the resulting matrix —
# and therefore the JLD2 fixture bytes — match older releases.
Q_struct <- INLA:::inla.rw(n, order = 2, scale.model = FALSE, sparse = TRUE)

# inla.qinv: diagonal of Q^{-1} under sum-to-zero constraint. Supply the
# two constraints RW2 uses under the hood: 1^T x = 0 and i^T x = 0.
A <- rbind(rep(1, n), seq_len(n))
e <- c(0, 0)
qinv_full <- INLA::inla.qinv(Q_struct, constr = list(A = A, e = e))
qinv_diag <- diag(qinv_full)

# Log-determinant of the constrained precision (generalised log-det on
# the non-null subspace). INLA exposes this via `inla.mesh.project`-style
# internals; for oracle purposes we recompute from eigenvalues.
eig <- eigen(as.matrix(Q_struct), symmetric = TRUE, only.values = TRUE)$values
eig_pos <- eig[eig > 1.0e-10]
log_det_gen <- sum(log(eig_pos))

out_path <- file.path(here, "..", "..", "..",
                      "packages", "GMRFs.jl", "test", "oracle",
                      "fixtures", "qinv_rw2.json")

write_gmrf_fixture(
    Q = Q_struct,
    path = out_path,
    name = "qinv_rw2",
    qinv_diag = qinv_diag,
    log_det = log_det_gen,
    meta = list(
        n = n,
        order = 2,
        scale_model = FALSE,
        constraints = "sum-to-zero + linear",
        notes = "RW2 structure matrix (unscaled); diag(Q^{-1}) under both null-space constraints."
    )
)

cat("wrote ", out_path, "\n")

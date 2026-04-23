#!/usr/bin/env Rscript
# verify-defaults/bym2_phi.R
#
# Probe the running R-INLA for the default PC prior on the BYM2 mixing
# parameter phi, compare against the Julia-side constant, and write a
# CSV snapshot under `output/` that CI can diff against the committed
# version.
#
# Writes:
#   - output/bym2_phi.csv with columns
#       parameter, rinla_value, rinla_version, julia_const, julia_value, match
#
# Julia-side constant under check:
#   LatentGaussianModels.Priors.DEFAULT_BYM2_PHI_ALPHA
#   -> currently set to 2/3 (Riebler et al. 2016). Unverified; see
#      plans/defaults-parity.md.
#
# Exit code:
#   0  if all probed defaults match the Julia-side values
#   2  on mismatch (machine-readable; CI gates on this)
#   1  on R-INLA import or introspection failure

suppressPackageStartupMessages({
    if (!requireNamespace("INLA", quietly = TRUE)) {
        message("INLA package is required. Install from https://www.r-inla.org/.")
        quit(status = 1)
    }
    library(INLA)
})

# --- Julia-side values we are checking against ------------------------
#
# Keep in sync with packages/LatentGaussianModels.jl/src/priors/bym2_phi.jl
# (DEFAULT_BYM2_PHI_ALPHA). If you move the constant, update both places
# in the same PR.
JULIA_DEFAULT_BYM2_PHI_U <- 0.5
JULIA_DEFAULT_BYM2_PHI_ALPHA <- 2 / 3
JULIA_CONST_NAME <- "LatentGaussianModels.DEFAULT_BYM2_PHI_ALPHA"

# --- Read R-INLA-side defaults ----------------------------------------
#
# R-INLA stores prior defaults in `inla.models()$latent$bym2$hyper`. For
# PC priors the parameters are an unnamed length-2 numeric vector where
# the first element is `U` and the second is `alpha`.
rinla_version <- as.character(packageVersion("INLA"))

bym2_info <- tryCatch({
    INLA::inla.models()$latent$bym2
}, error = function(e) {
    message("Could not read inla.models()$latent$bym2: ", conditionMessage(e))
    quit(status = 1)
})

phi_hyper <- bym2_info$hyper$theta2
if (is.null(phi_hyper) || is.null(phi_hyper$prior) || is.null(phi_hyper$param)) {
    message("Unexpected bym2 phi hyper structure; cannot read default (U, alpha).")
    str(phi_hyper)
    quit(status = 1)
}
stopifnot(identical(phi_hyper$prior, "pc") || identical(phi_hyper$prior, "pc.mix"))
rinla_U <- phi_hyper$param[[1]]
rinla_alpha <- phi_hyper$param[[2]]

# --- Compare ----------------------------------------------------------
rows <- data.frame(
    parameter = c("bym2_phi_U", "bym2_phi_alpha"),
    rinla_value = c(rinla_U, rinla_alpha),
    rinla_version = rinla_version,
    julia_const = c("DEFAULT_BYM2_PHI_U (U argument default in BYM2())",
                      JULIA_CONST_NAME),
    julia_value = c(JULIA_DEFAULT_BYM2_PHI_U, JULIA_DEFAULT_BYM2_PHI_ALPHA),
    stringsAsFactors = FALSE
)
rows$match <- isTRUE(all.equal(rows$rinla_value, rows$julia_value, tolerance = 1e-10))

# --- Write snapshot ---------------------------------------------------
out_dir <- file.path("scripts", "verify-defaults", "output")
# When invoked from the project root `Rscript scripts/verify-defaults/bym2_phi.R`
# this path is correct. When invoked from within the scripts/verify-defaults
# dir, fall back to a local output/.
if (!dir.exists(out_dir)) {
    out_dir <- "output"
}
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, "bym2_phi.csv")
write.csv(rows, out_path, row.names = FALSE)
message("Wrote ", out_path)
print(rows)

# --- Exit -------------------------------------------------------------
if (!all(rows$match)) {
    message("Mismatch detected between R-INLA and Julia defaults.")
    message("Update DEFAULT_BYM2_PHI_ALPHA in ",
            "packages/LatentGaussianModels.jl/src/priors/bym2_phi.jl or ",
            "record the intentional divergence in plans/defaults-parity.md.")
    quit(status = 2)
}
quit(status = 0)

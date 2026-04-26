# Reference: Gaussian regression with a generic0 random effect via R-INLA.
# Validates the LGM `Generic0` component end-to-end against R-INLA's
# `model = "generic0"`, where the user supplies a fixed structure
# matrix C (R-INLA convention) and a single precision parameter τ such
# that Q = τ · C.
#
# Setup: random structure matrix `C = LL'` for a random L plus a small
# diagonal — symmetric, full-rank, well-conditioned. n = 8 latent
# coordinates, n_obs = 30 observations with random projector A.
#
# Model:
#   y_i | η_i, σ ~ N(η_i, σ²)
#   η = A · b
#   b ~ N(0, (τ C)⁻¹)
#   τ ~ logGamma(1, 5e-5)            # R-INLA default (`generic0`)
#   1/σ² ~ logGamma(1, 5e-5)         # R-INLA default likelihood prec

here <- tryCatch(
    normalizePath(dirname(sys.frame(1)$ofile), mustWork = FALSE),
    error = function(e) getwd()
)
if (!nzchar(here)) here <- getwd()
source(file.path(here, "..", "_helpers.R"))

suppressPackageStartupMessages({
    library(INLA)
})

set.seed(20260427)

n_lat <- 8L
n_obs <- 30L

# Random SPD structure matrix C of size n_lat × n_lat.
L <- matrix(rnorm(n_lat * n_lat), n_lat, n_lat)
C <- L %*% t(L) + diag(0.1, n_lat)
C <- (C + t(C)) / 2

# Random projector A and observations.
A <- matrix(rnorm(n_obs * n_lat), n_obs, n_lat)
b_true <- rnorm(n_lat) / sqrt(2.0)
sigma_true <- 0.3
y <- as.numeric(A %*% b_true + rnorm(n_obs, sd = sigma_true))

# R-INLA's `generic0` consumes a structure matrix via `Cmatrix`.
Csp <- Matrix::Matrix(C, sparse = TRUE)

dat <- list(
    y = y,
    idx = seq_len(n_lat)
)

# Build the linear predictor manually using the projector A. Tell INLA
# to treat each `dat$y` row as having a known linear combination of the
# generic0 latent. Easiest route in R-INLA: an "A-matrix" via `inla.stack`.
stk <- INLA::inla.stack(
    data = list(y = y),
    A = list(A),
    effects = list(idx = seq_len(n_lat)),
    tag = "obs"
)

formula <- y ~ -1 + f(idx, model = "generic0", Cmatrix = Csp,
                       hyper = list(prec = list(prior = "loggamma",
                                                param = c(1, 5e-5))))

fit <- INLA::inla(
    formula,
    family = "gaussian",
    data = INLA::inla.stack.data(stk),
    control.predictor = list(A = INLA::inla.stack.A(stk), compute = TRUE),
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm", "synthetic_generic0.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "synthetic_generic0",
    component_names = c("idx"),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        n_lat = n_lat,
        n_obs = n_obs,
        family = "gaussian",
        model = "generic0",
        prec_prior = "loggamma(1, 5e-5)"
    )
)

# Append input data so the Julia-side oracle test can re-run the fit.
fixture <- jsonlite::fromJSON(out_path, simplifyVector = FALSE)
fixture$input <- list(
    y = as.numeric(y),
    A = as.numeric(t(A)),                 # row-major, length n_obs * n_lat
    n_obs = n_obs,
    n_lat = n_lat,
    C = sparse_to_triplet(Csp)
)
jsonlite::write_json(
    fixture, out_path,
    auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
)

cat("wrote ", out_path, "\n")

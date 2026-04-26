# Reference: Gaussian regression with a generic1-style random effect
# via R-INLA. R-INLA's `generic1` adds a β-mixing parameter; we defer
# that flavour and only validate the eigenvalue rescaling. To get a
# direct apples-to-apples R-INLA fixture we use `model = "generic0"`
# but pre-rescale the user-supplied structure matrix so its largest
# eigenvalue is 1 — i.e. exactly what `Generic1(...)` does on the
# Julia side at construction. The Julia oracle test then loads the
# fixture and calls `Generic1(C)` directly, with the rescaling
# happening inside the constructor.
#
# Setup: random structure matrix `C = LL'` for a random L plus a small
# diagonal — symmetric, full-rank, well-conditioned. n = 8 latent
# coordinates, n_obs = 30 observations with random projector A.
#
# Model:
#   y_i | η_i, σ ~ N(η_i, σ²)
#   η = A · b
#   b ~ N(0, (τ C̃)⁻¹)  with C̃ = C / λ_max(C)
#   τ ~ logGamma(1, 5e-5)            # R-INLA default
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

set.seed(20260428)

n_lat <- 8L
n_obs <- 30L

# Random SPD structure matrix C of size n_lat × n_lat.
L <- matrix(rnorm(n_lat * n_lat), n_lat, n_lat)
C <- L %*% t(L) + diag(0.1, n_lat)
C <- (C + t(C)) / 2

# Pre-rescale so λ_max(C̃) = 1 (Generic1's defining transformation).
lambda_max <- max(eigen(C, symmetric = TRUE, only.values = TRUE)$values)
C_tilde <- C / lambda_max

# Random projector A and observations.
A <- matrix(rnorm(n_obs * n_lat), n_obs, n_lat)
b_true <- rnorm(n_lat) / sqrt(2.0)
sigma_true <- 0.3
y <- as.numeric(A %*% b_true + rnorm(n_obs, sd = sigma_true))

# Pass the rescaled C̃ to R-INLA's `generic0`.
Csp <- Matrix::Matrix(C_tilde, sparse = TRUE)

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

out_path <- file.path(here, "..", "fixtures", "lgm", "synthetic_generic1.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "synthetic_generic1",
    component_names = c("idx"),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        n_lat = n_lat,
        n_obs = n_obs,
        family = "gaussian",
        model = "generic1 (validated via generic0 + λ_max rescaling)",
        prec_prior = "loggamma(1, 5e-5)"
    )
)

# Append the *original* (unrescaled) C and the original projector A so
# the Julia-side oracle test can re-run Generic1(C), which performs
# its own λ_max rescaling internally.
fixture <- jsonlite::fromJSON(out_path, simplifyVector = FALSE)
fixture$input <- list(
    y = as.numeric(y),
    A = as.numeric(t(A)),                 # row-major, length n_obs * n_lat
    n_obs = n_obs,
    n_lat = n_lat,
    C = sparse_to_triplet(Matrix::Matrix(C, sparse = TRUE)),
    lambda_max = lambda_max               # for sanity-checking on the Julia side
)
jsonlite::write_json(
    fixture, out_path,
    auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
)

cat("wrote ", out_path, "\n")

# Reference: Gaussian observations on a 4x4 grid graph with Leroux
# (1999) CAR random effect, R-INLA's `model = "besagproper2"`. The
# precision is τ ((1 - φ) I + φ R) with R the combinatorial graph
# Laplacian. Validates the LGM `Leroux` component end-to-end.

here <- tryCatch(
    normalizePath(dirname(sys.frame(1)$ofile), mustWork = FALSE),
    error = function(e) getwd()
)
if (!nzchar(here)) here <- getwd()
source(file.path(here, "..", "_helpers.R"))

suppressPackageStartupMessages({
    library(INLA)
})

set.seed(20260430)

# 4x4 lattice adjacency (rook neighbours).
side <- 4L
n <- as.integer(side * side)
W <- matrix(0, n, n)
for (i in seq_len(side)) {
    for (j in seq_len(side)) {
        idx <- (i - 1L) * side + j
        if (j < side) {
            r <- (i - 1L) * side + (j + 1L)
            W[idx, r] <- 1
            W[r, idx] <- 1
        }
        if (i < side) {
            d <- i * side + j
            W[idx, d] <- 1
            W[d, idx] <- 1
        }
    }
}

graph_file <- tempfile(fileext = ".graph")
INLA::inla.write.graph(W, file = graph_file)

# Truth: smooth-ish field, moderate spatial correlation. Five
# observations per region give enough data to identify the Gaussian
# precision separately from the random effect.
alpha_true <- 0.7
b_true <- as.numeric(scale(rnorm(n) + 0.5 * (1:n) / n))   # gradient-ish
sigma_true <- 0.3
reps_per_region <- 5L
n_obs <- n * reps_per_region
region <- rep(seq_len(n), each = reps_per_region)
y <- alpha_true + b_true[region] + rnorm(n_obs, sd = sigma_true)

dat <- list(y = y, region = region)

# `besagproper2` is the Leroux convex-combination CAR in R-INLA.
formula <- y ~ 1 +
    f(region, model = "besagproper2", graph = graph_file,
      hyper = list(
          prec   = list(prior = "pc.prec",   param = c(1, 0.01)),
          lambda = list(prior = "logitbeta", param = c(1, 1))
      ))

fit <- INLA::inla(
    formula,
    family = "gaussian",
    data = dat,
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm", "synthetic_leroux.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "synthetic_leroux",
    component_names = c("region"),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        n = n,
        family = "gaussian",
        model = "besagproper2",
        prec_prior = "pc.prec(1, 0.01)",
        phi_prior  = "logitbeta(1, 1)"
    )
)

fixture <- jsonlite::fromJSON(out_path, simplifyVector = FALSE)
fixture$input <- list(
    y = as.numeric(y),
    region = as.integer(region),
    n = n,
    n_obs = n_obs,
    W = sparse_to_triplet(Matrix::Matrix(W, sparse = TRUE))
)
jsonlite::write_json(
    fixture, out_path,
    auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
)

cat("wrote ", out_path, "\n")

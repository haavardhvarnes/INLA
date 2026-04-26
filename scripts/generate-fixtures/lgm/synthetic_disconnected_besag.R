# Reference: Besag random-effect on a synthetic disconnected graph via
# R-INLA. Validates the per-connected-component sum-to-zero machinery
# (Freni-Sterrantino et al. 2018) on K = 3 components.
#
# Graph: three small islands of sizes (5, 4, 3); within each island a
# path: 1-2-3-4-5, then 6-7-8-9, then 10-11-12. Total n = 12.
#
# Model:
#   y_i | η_i ~ Poisson(exp(η_i))
#   η_i = β_0 + b_i
#   b ~ scaled-Besag(W)              # per-CC Sørbye-Rue scaling
#   τ_b ~ PC(U = 1, α = 0.01)

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

# Build the disconnected adjacency: three path components.
n <- 12L
W <- matrix(0, n, n)
edges <- list(
    c(1, 2), c(2, 3), c(3, 4), c(4, 5),
    c(6, 7), c(7, 8), c(8, 9),
    c(10, 11), c(11, 12)
)
for (e in edges) {
    W[e[1], e[2]] <- 1
    W[e[2], e[1]] <- 1
}

graph_file <- tempfile(fileext = ".graph")
INLA::inla.write.graph(W, file = graph_file)

# Generate Poisson observations with shared intercept and modest spatial
# variation. The point of the fixture is the structural check, not
# strong identifiability.
beta0_true <- 1.0
b_true <- c(
    0.3, 0.0, -0.3, 0.1, -0.1,    # CC 1 (sums to 0)
    0.2, -0.4, 0.1, 0.1,          # CC 2 (sums to 0)
    -0.5, 0.2, 0.3                # CC 3 (sums to 0)
)
stopifnot(abs(sum(b_true[1:5])) < 1e-12)
stopifnot(abs(sum(b_true[6:9])) < 1e-12)
stopifnot(abs(sum(b_true[10:12])) < 1e-12)

mu <- exp(beta0_true + b_true)
y <- rpois(n, mu)

dat <- data.frame(y = y, region = seq_len(n))

formula <- y ~ 1 +
    f(region, model = "besag", graph = graph_file,
      scale.model = TRUE,
      hyper = list(prec = list(prior = "pc.prec", param = c(1, 0.01))))

fit <- INLA::inla(
    formula,
    family = "poisson",
    data = dat,
    control.predictor = list(compute = TRUE),
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm", "synthetic_disconnected_besag.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "synthetic_disconnected_besag",
    component_names = c("region"),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        n = n,
        family = "poisson",
        model = "besag",
        scale_model = TRUE,
        K_components = 3L,
        prec_prior = "pc.prec(1, 0.01)"
    )
)

fixture <- jsonlite::fromJSON(out_path, simplifyVector = FALSE)
fixture$input <- list(
    y = as.integer(dat$y),
    W = sparse_to_triplet(Matrix::Matrix(W, sparse = TRUE))
)
jsonlite::write_json(
    fixture, out_path,
    auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
)

cat("wrote ", out_path, "\n")

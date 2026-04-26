# Reference: classical (non-reparametrised) BYM on Scotland lip cancer
# via R-INLA. Smaller-scope companion to scotland_bym2 — exercises the
# `model = "bym"` pathway with separate τ_v (iid) and τ_b (besag)
# precisions, no φ.
#
# Model:
#   y_i | η_i ~ Poisson(E_i · exp(η_i))
#   η_i = β_0 + x_i β_x + v_i + b_i
#   v_i ~ N(0, 1/τ_v)
#   b   ~ scaled-Besag(W)               # Sørbye-Rue scaled, sum-to-zero
#   τ_v ~ PC(U = 1, α = 0.01)
#   τ_b ~ PC(U = 1, α = 0.01)
#
# Dataset: SpatialEpi::scotland.

here <- tryCatch(
    normalizePath(dirname(sys.frame(1)$ofile), mustWork = FALSE),
    error = function(e) getwd()
)
if (!nzchar(here)) here <- getwd()
source(file.path(here, "..", "_helpers.R"))

suppressPackageStartupMessages({
    library(INLA)
    library(SpatialEpi)
    library(spdep)
})

set.seed(20260426)

data(scotland)
dat <- scotland$data
nb <- spdep::poly2nb(scotland$spatial.polygon)
W <- spdep::nb2mat(nb, style = "B", zero.policy = TRUE)
n <- nrow(W)
graph_file <- tempfile(fileext = ".graph")
INLA::inla.write.graph(W, file = graph_file)

dat$region <- seq_len(n)
dat$x <- scale(dat$AFF)[, 1]

formula <- cases ~ 1 + x +
    f(region, model = "bym", graph = graph_file,
      scale.model = TRUE,
      hyper = list(
          prec.unstruct = list(prior = "pc.prec", param = c(1, 0.01)),
          prec.spatial  = list(prior = "pc.prec", param = c(1, 0.01))
      ))

fit <- INLA::inla(
    formula,
    family = "poisson",
    data = dat,
    E = dat$expected,
    control.predictor = list(compute = TRUE),
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm", "scotland_bym.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "scotland_bym",
    component_names = c("region"),
    include_marginals = TRUE,
    meta = list(
        dataset = "SpatialEpi::scotland",
        n = n,
        family = "poisson",
        model = "bym",
        scale_model = TRUE,
        covariates = "AFF (standardized)",
        prec_iid_prior = "pc.prec(1, 0.01)",
        prec_besag_prior = "pc.prec(1, 0.01)"
    )
)

# Append input data so the Julia-side oracle test can re-run the same
# fit without SpatialEpi / spdep dependencies.
fixture <- jsonlite::fromJSON(out_path, simplifyVector = FALSE)
fixture$input <- list(
    cases    = as.integer(dat$cases),
    expected = as.numeric(dat$expected),
    x        = as.numeric(dat$x),
    W        = sparse_to_triplet(Matrix::Matrix(W, sparse = TRUE))
)
jsonlite::write_json(
    fixture, out_path,
    auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
)

cat("wrote ", out_path, "\n")

# Reference: zero-inflated Poisson (type 1, standard mixture) regression
# via R-INLA on a synthetic dataset. Smallest possible oracle for the
# `family = "zeroinflatedpoisson1"` pathway — intercept + one covariate,
# no latent random effect, n = 200 observations.
#
# Model:
#   y_i = 0           with prob π + (1 - π) · exp(-μ_i)
#   y_i = k  (k ≥ 1)  with prob (1 - π) · μ_i^k exp(-μ_i) / k!
#   μ_i = exp(α + β x_i)
#   logit(π) ~ N(0, 1)              # R-INLA default for zeroinflatedpoisson1
#
# The fixture stores the input vectors so the Julia-side oracle test
# can re-fit the same data without requiring R at test time.

here <- tryCatch(
    normalizePath(dirname(sys.frame(1)$ofile), mustWork = FALSE),
    error = function(e) getwd()
)
if (!nzchar(here)) here <- getwd()
source(file.path(here, "..", "_helpers.R"))

suppressPackageStartupMessages({
    library(INLA)
})

set.seed(20260501)

n <- 200L
pi_true <- 0.3
alpha_true <- 0.4
beta_true <- 0.8
x <- rnorm(n)
mu <- exp(alpha_true + beta_true * x)

# Sample from the ZIP1 mixture: with prob π emit a structural zero,
# otherwise emit a Poisson(μ) draw.
z <- rbinom(n, size = 1L, prob = pi_true)
y <- ifelse(z == 1L, 0L, rpois(n, lambda = mu))

df <- data.frame(y = y, x = x)

formula <- y ~ 1 + x

fit <- INLA::inla(
    formula,
    family = "zeroinflatedpoisson1",
    data = df,
    control.predictor = list(compute = TRUE),
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm", "synthetic_zip1.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "synthetic_zip1",
    component_names = character(0),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        n = n,
        family = "zeroinflatedpoisson1",
        pi_true = pi_true,
        alpha_true = alpha_true,
        beta_true = beta_true,
        pi_prior = "logit(pi) ~ N(0, 1)"
    )
)

# Append input data so the Julia-side oracle test can re-run the fit.
fixture <- jsonlite::fromJSON(out_path, simplifyVector = FALSE)
fixture$input <- list(
    y = as.integer(df$y),
    x = as.numeric(df$x)
)
jsonlite::write_json(
    fixture, out_path,
    auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
)

cat("wrote ", out_path, "\n")

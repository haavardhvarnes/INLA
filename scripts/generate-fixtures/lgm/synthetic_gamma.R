# Reference: Gamma regression via R-INLA on a synthetic dataset.
# Smallest possible oracle for the `family = "gamma"` pathway —
# intercept + one covariate, no latent random effect, 200 observations.
#
# Model (R-INLA's mean-precision parameterisation):
#   y_i | η_i, φ ~ Gamma(μ_i, φ),  μ_i = exp(η_i),  Var = μ_i² / φ
#   η_i = α + β x_i
#   log(φ) ~ logGamma(1, 5e-5)            # R-INLA default for gamma
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

set.seed(20260427)

n <- 200L
phi_true <- 4.0
alpha_true <- 0.3
beta_true <- 0.6
x <- rnorm(n)
mu <- exp(alpha_true + beta_true * x)
# rgamma uses (shape, rate). Match α = φ, rate = φ/μ → mean = μ, var = μ²/φ.
y <- rgamma(n, shape = phi_true, rate = phi_true / mu)

df <- data.frame(y = y, x = x)

formula <- y ~ 1 + x

fit <- INLA::inla(
    formula,
    family = "gamma",
    data = df,
    control.predictor = list(compute = TRUE),
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm", "synthetic_gamma.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "synthetic_gamma",
    component_names = character(0),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        n = n,
        family = "gamma",
        phi_true = phi_true,
        alpha_true = alpha_true,
        beta_true = beta_true,
        phi_prior = "loggamma(1, 5e-5)"
    )
)

# Append input data so the Julia-side oracle test can re-run the fit.
fixture <- jsonlite::fromJSON(out_path, simplifyVector = FALSE)
fixture$input <- list(
    y = as.numeric(df$y),
    x = as.numeric(df$x)
)
jsonlite::write_json(
    fixture, out_path,
    auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
)

cat("wrote ", out_path, "\n")

# Reference: NegativeBinomial regression via R-INLA on a synthetic
# dataset. Smallest possible oracle for the `family = "nbinomial"`
# pathway — intercept + one covariate, no latent random effect, 200
# observations.
#
# Model:
#   y_i | η_i, n ~ NegBinomial(μ_i, size = n),  μ_i = exp(η_i)
#   η_i = α + β x_i
#   log(n) ~ logGamma(1, 0.1)            # R-INLA default for nbinomial
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

set.seed(20260426)

n <- 200L
size_true <- 2.0
alpha_true <- 0.4
beta_true <- 0.8
x <- rnorm(n)
mu <- exp(alpha_true + beta_true * x)
# Simulate from NegBin(mu, size) using R's mu/size parameterisation.
y <- rnbinom(n, mu = mu, size = size_true)

df <- data.frame(y = y, x = x)

formula <- y ~ 1 + x

fit <- INLA::inla(
    formula,
    family = "nbinomial",
    data = df,
    control.predictor = list(compute = TRUE),
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm", "synthetic_nbinomial.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "synthetic_nbinomial",
    component_names = character(0),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        n = n,
        family = "nbinomial",
        size_true = size_true,
        alpha_true = alpha_true,
        beta_true = beta_true,
        size_prior = "loggamma(1, 0.1)"
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

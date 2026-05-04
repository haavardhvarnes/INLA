# Reference: Beta regression via R-INLA on a synthetic dataset.
# Smallest possible oracle for the `family = "beta"` pathway —
# intercept + one covariate, no latent random effect, 200 observations.
#
# Model (R-INLA's mean-dispersion parameterisation):
#   y_i | η_i, φ ~ Beta(μ_i φ, (1 - μ_i) φ),  μ_i = logit⁻¹(η_i),
#   Var = μ_i (1 - μ_i) / (φ + 1)
#   η_i = α + β x_i
#   log(φ) ~ logGamma(1, 0.01)             # R-INLA default for beta
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

set.seed(20260504)

n <- 200L
phi_true <- 8.0
alpha_true <- 0.2
beta_true <- 0.5
x <- rnorm(n)
mu <- 1 / (1 + exp(-(alpha_true + beta_true * x)))
y <- rbeta(n, shape1 = phi_true * mu, shape2 = phi_true * (1 - mu))

df <- data.frame(y = y, x = x)

formula <- y ~ 1 + x

# Pin the prior to loggamma(1, 0.01) on log(φ) so the R-side prior is
# bit-for-bit identical to Julia's `BetaLikelihood` default
# `GammaPrecision(1.0, 0.01)`. (R-INLA's documented default is the same
# but we set it explicitly to remove any silent-default risk.)
fit <- INLA::inla(
    formula,
    family = "beta",
    data = df,
    control.family = list(
        hyper = list(
            theta = list(prior = "loggamma", param = c(1.0, 0.01))
        )
    ),
    control.predictor = list(compute = TRUE),
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm", "synthetic_beta.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "synthetic_beta",
    component_names = character(0),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        n = n,
        family = "beta",
        phi_true = phi_true,
        alpha_true = alpha_true,
        beta_true = beta_true,
        phi_prior = "loggamma(1, 0.01)"
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

# Reference: Student-t regression via R-INLA on a synthetic dataset.
# Smallest possible oracle for the `family = "T"` pathway —
# intercept + one covariate, no latent random effect, 200 observations.
#
# Model (R-INLA's scaled-t parameterisation):
#   y_i = η_i + ε_i / √τ,  ε_i ~ Student-t(ν),
#   η_i = α + β x_i
#   log(τ) ~ logGamma(1, 1e-4)
#   log(ν − 2) ~ Gaussian(2.5, prec = 1)
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
tau_true <- 4.0          # σ = 0.5
nu_true <- 5.0           # heavier-than-Gaussian tails
alpha_true <- 0.4
beta_true <- 0.7

x <- rnorm(n)
eta <- alpha_true + beta_true * x
# Sample from scaled-t: y = η + (1/√τ) · t_ν
y <- eta + rt(n, df = nu_true) / sqrt(tau_true)

df <- data.frame(y = y, x = x)

formula <- y ~ 1 + x

# Pin both hyperparameter priors so the R-side prior is bit-for-bit
# identical to Julia's `StudentTLikelihood` defaults
# (`GammaPrecision(1, 1e-4)` on `log τ`, `GaussianPrior(2.5, 1)` on
# `log(ν − 2)`).
fit <- INLA::inla(
    formula,
    family = "T",
    data = df,
    control.family = list(
        hyper = list(
            theta1 = list(prior = "loggamma", param = c(1.0, 1.0e-4)),
            theta2 = list(prior = "gaussian", param = c(2.5, 1.0))
        )
    ),
    control.predictor = list(compute = TRUE),
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm", "synthetic_studentt.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "synthetic_studentt",
    component_names = character(0),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        n = n,
        family = "T",
        tau_true = tau_true,
        nu_true = nu_true,
        alpha_true = alpha_true,
        beta_true = beta_true,
        tau_prior = "loggamma(1, 1e-4)",
        nu_prior = "gaussian(2.5, prec=1) on log(ν − 2)"
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

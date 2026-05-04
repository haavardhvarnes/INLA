# Reference: GEV regression via R-INLA on a synthetic dataset.
# Smallest possible oracle for the `family = "gev"` pathway —
# intercept + one covariate, no latent random effect, 200 observations.
#
# R-INLA's `gev` family is marked **disabled** in current releases in
# favour of `bgev`. We re-enable it via the documented escape hatch
# (`enable.model.likelihood.gev <- TRUE` in `inla.get.inlaEnv()`) for
# the body-of-distribution oracle. The disabled tag warns "usage is
# either not recommended and/or unsupported" — we test bit-for-bit
# parameterisation parity, which the disabled status does not
# invalidate.
#
# Model (R-INLA's GEV parameterisation):
#   F(y; η, τ, ξ) = exp(− [1 + ξ √τ (y − η)]^(−1/ξ))
#   η_i = α + β x_i
#   log(τ) ~ logGamma(1, 5e-5)              (R-INLA default)
#   ξ / 0.1 = θ2 ~ Gaussian(0, prec = 25)   (R-INLA default; gev.scale.xi=0.1)
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
    library(evd)
})

# Re-enable R-INLA's "gev" family (disabled in current releases per
# `inla.doc("gev")$status`). bgev is the supported body+tail variant;
# we test the body parameterisation here.
assign("enable.model.likelihood.gev", TRUE, envir = inla.get.inlaEnv())

set.seed(20260504)

n <- 200L
tau_true <- 16.0          # σ = 1/√τ = 0.25
xi_true <- 0.1            # moderate positive shape
alpha_true <- 0.3
beta_true <- 0.6

x <- rnorm(n)
eta <- alpha_true + beta_true * x
# y = η + (1/√τ) · z, z ~ standard GEV(loc=0, scale=1, shape=ξ).
z <- evd::rgev(n, loc = 0, scale = 1, shape = xi_true)
y <- eta + z / sqrt(tau_true)

df <- data.frame(y = y, x = x)

formula <- y ~ 1 + x

# Pin both hyperparameter priors so the R-side prior is bit-for-bit
# identical to Julia's `GEVLikelihood` defaults
# (`GammaPrecision(1, 5e-5)` on `log τ`, `GaussianPrior(0, σ=0.2)` =
#  `gaussian(0, prec=25)` on `θ2 = ξ / 0.1`).
fit <- INLA::inla(
    formula,
    family = "gev",
    data = df,
    control.family = list(
        gev.scale.xi = 0.1,
        hyper = list(
            prec = list(prior = "loggamma", param = c(1.0, 5.0e-5)),
            tail = list(prior = "gaussian", param = c(0.0, 25.0))
        )
    ),
    control.predictor = list(compute = TRUE),
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm", "synthetic_gev.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "synthetic_gev",
    component_names = character(0),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        n = n,
        family = "gev",
        tau_true = tau_true,
        xi_true = xi_true,
        alpha_true = alpha_true,
        beta_true = beta_true,
        xi_scale = 0.1,
        tau_prior = "loggamma(1, 5e-5)",
        xi_prior = "gaussian(0, prec=25) on θ2 = ξ/0.1",
        notes = paste(
            "R-INLA `family = \"gev\"` is marked disabled in favour",
            "of `bgev`; re-enabled via",
            "enable.model.likelihood.gev <- TRUE for this oracle."
        )
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

# Reference: Skew-normal regression via R-INLA on a synthetic dataset.
# Smallest possible oracle for the `family = "sn"` pathway —
# intercept + one covariate, no latent random effect, 300 observations.
#
# Model (R-INLA's standardised SN parameterisation):
#   z_i = (y_i − η_i) √τ ~ standardised SN with skewness γ
#   η_i = α + β x_i
#   log(τ) ~ logGamma(1, 5e-5)             (R-INLA default)
#   logit-skew θ_2 ~ Gaussian(0, prec = 1) (overrides R-INLA's pc.sn)
#
# Julia's `SkewNormalLikelihood` defaults to a Gaussian on the internal
# `θ[2] = logit-skew` (R-INLA's pc.sn has no closed-form Julia
# equivalent yet); we pin both sides via `control.family$hyper$skew$
# prior = "gaussian"` to keep the prior bit-for-bit identical.
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
    library(sn)
})

set.seed(20260505)

n <- 300L
tau_true <- 16.0           # σ = 0.25
gamma_true <- 0.5          # moderate positive skewness
alpha_true <- 0.3
beta_true <- 0.6

x <- rnorm(n)
eta <- alpha_true + beta_true * x

# Sample from R-INLA's "sn" generative model: y_i = η_i + (1/√τ) · z_i,
# where z_i is standardised SN with mean 0, variance 1, skewness γ.
# `INLA:::inla.sn.reparam` solves for (xi, omega, alpha) given moments
# (mean, variance, skewness).
y <- numeric(n)
for (i in seq_len(n)) {
    p <- INLA:::inla.sn.reparam(
        moments = c(eta[i], 1.0 / tau_true, gamma_true)
    )
    y[i] <- sn::rsn(1L, xi = p$xi, omega = p$omega, alpha = p$alpha)
}

df <- data.frame(y = y, x = x)

formula <- y ~ 1 + x

# Pin both hyperparameter priors to match Julia's SkewNormalLikelihood
# defaults: `GammaPrecision(1, 5e-5)` on `log τ`, `GaussianPrior(0, 1)`
# on the logit-skew internal scale `θ[2]`.
fit <- INLA::inla(
    formula,
    family = "sn",
    data = df,
    control.family = list(
        hyper = list(
            prec = list(prior = "loggamma", param = c(1.0, 5.0e-5)),
            skew = list(prior = "gaussian", param = c(0.0, 1.0))
        )
    ),
    control.predictor = list(compute = TRUE),
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm", "synthetic_skewnormal.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "synthetic_skewnormal",
    component_names = character(0),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        n = n,
        family = "sn",
        tau_true = tau_true,
        gamma_true = gamma_true,
        alpha_true = alpha_true,
        beta_true = beta_true,
        tau_prior = "loggamma(1, 5e-5)",
        skew_prior = "gaussian(0, prec=1) on logit-skew"
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

# Reference: Gamma-survival regression via R-INLA on a synthetic
# right-censored dataset. Smallest oracle for the `gammasurv` pathway —
# intercept + one covariate, no latent random effect, mixture of
# uncensored and right-censored rows.
#
# Model (R-INLA's `family = "gammasurv"`, log-link on the linear
# predictor; mean parameterisation):
#   T_i ~ Gamma(shape = φ, rate = φ / μ_i)     ⇒ E[T_i] = μ_i
#   μ_i = exp(α + β x_i)
#   y_i = min(T_i, C_i),   event_i = 1{T_i ≤ C_i}
#
# Generative parameters: α = 0.3, β = 0.6, φ_true = 2.0, n = 200.
# Censoring time C_i ~ Uniform(0.5, 8.0), giving ~14% right-censored.
#
# Hyperparameter prior: loggamma(1, 5e-5) on log φ. Matches our
# `GammaSurvLikelihood` default (`GammaPrecision(1.0, 5.0e-5)`), which
# is also the R-INLA gammasurv default — so no `control.family` override
# is needed for parity.

here <- tryCatch(
    normalizePath(dirname(sys.frame(1)$ofile), mustWork = FALSE),
    error = function(e) getwd()
)
if (!nzchar(here)) here <- getwd()
source(file.path(here, "..", "_helpers.R"))

suppressPackageStartupMessages({
    library(INLA)
})

set.seed(20260502)

n <- 200L
alpha_true <- 0.3
beta_true  <- 0.6
phi_true   <- 2.0          # gamma shape (= R-INLA's "precision" hyperpar)

x <- rnorm(n)
mu_true <- exp(alpha_true + beta_true * x)
T_true  <- rgamma(n, shape = phi_true, rate = phi_true / mu_true)
C_i     <- runif(n, 0.5, 8.0)
event   <- as.integer(T_true <= C_i)
y_obs   <- pmin(T_true, C_i)

df <- data.frame(time = y_obs, event = event, x = x)

formula <- inla.surv(time = time, event = event) ~ 1 + x

fit <- INLA::inla(
    formula,
    family = "gammasurv",
    data = df,
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm",
                     "synthetic_gamma_survival.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "synthetic_gamma_survival",
    component_names = character(0),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        n = n,
        family = "gammasurv",
        alpha_true = alpha_true,
        beta_true = beta_true,
        phi_true = phi_true,
        n_events = sum(event),
        n_censored = sum(1 - event)
    )
)

# Append input data so the Julia-side oracle test can re-run the fit.
fixture <- jsonlite::fromJSON(out_path, simplifyVector = FALSE)
fixture$input <- list(
    time  = as.numeric(df$time),
    event = as.integer(df$event),
    x     = as.numeric(df$x)
)
jsonlite::write_json(
    fixture, out_path,
    auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
)

cat("wrote ", out_path, "\n")

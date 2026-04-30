# Reference: Lognormal-AFT survival regression via R-INLA on a
# synthetic right-censored dataset. Smallest oracle for the
# `lognormalsurv` pathway — intercept + one covariate, no latent random
# effect, mixture of uncensored and right-censored rows.
#
# Model (R-INLA's `family = "lognormalsurv"`, log-link on the linear
# predictor; η is the mean of log T):
#   log T_i    ~ N(η_i, σ²),    σ² = 1/τ
#   η_i        = α₀ + β x_i
#   y_i        = min(T_i, C_i),  event_i = 1{T_i ≤ C_i}
#
# Generative parameters: α (linear-predictor intercept) = 0.3, β = 0.6,
# σ_true = 0.6 (τ_true = 1/σ² ≈ 2.78), n = 200. Censoring time
# C_i ~ Uniform(0.5, 8.0). ~25% right-censored.
#
# Hyperparameter prior: PC prior on σ with `P(σ > 1) = 0.01`. Matches
# our `LognormalSurvLikelihood` default (`PCPrecision(1.0, 0.01)`).

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
alpha_lp_true <- 0.3       # linear-predictor intercept (mean of log T)
beta_true     <- 0.6
sigma_true    <- 0.6       # std of log T

x <- rnorm(n)
mu_true <- alpha_lp_true + beta_true * x
T_true  <- exp(rnorm(n, mean = mu_true, sd = sigma_true))
C_i     <- runif(n, 0.5, 8.0)
event   <- as.integer(T_true <= C_i)
y_obs   <- pmin(T_true, C_i)

df <- data.frame(time = y_obs, event = event, x = x)

formula <- inla.surv(time = time, event = event) ~ 1 + x

# PC prior on σ matching our default: P(σ > 1) = 0.01 → λ = -log(0.01)/1
fit <- INLA::inla(
    formula,
    family = "lognormalsurv",
    data = df,
    control.family = list(
        hyper = list(
            prec = list(prior = "pc.prec", param = c(1, 0.01))
        )
    ),
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm",
                     "synthetic_lognormal_survival.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "synthetic_lognormal_survival",
    component_names = character(0),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        n = n,
        family = "lognormalsurv",
        alpha_lp_true = alpha_lp_true,
        beta_true = beta_true,
        sigma_true = sigma_true,
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

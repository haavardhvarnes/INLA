# Reference: Weibull survival regression via R-INLA on a synthetic
# right-censored dataset. Smallest oracle for the Weibull-PH survival
# pathway — intercept + one covariate, no latent random effect, mixture
# of uncensored and right-censored rows. Adds a single hyperparameter
# (shape α) on top of the exponential-survival fixture.
#
# Model (R-INLA's `family = "weibullsurv"`, variant 0 / PH, log link):
#   h(t | η_i) = α λ_i t^(α-1),  λ_i = exp(η_i)
#   T_i        ~ Weibull(shape = α, scale = λ_i^{-1/α})
#   η_i        = α₀ + β x_i
#   y_i        = min(T_i, C_i),  event_i = 1{T_i ≤ C_i}
#
# Generative parameters: α (linear-predictor intercept) = -0.5, β = 0.7,
# shape α_w = 1.5, n = 200. Censoring time C_i ~ Uniform(0.5, 4.0).
# ~28% right-censored.
#
# Hyperparameter prior: R-INLA's `weibullsurv` default — `loggamma(1, 0.001)`
# on `log α_w`. This matches our `WeibullLikelihood` placeholder default
# (`GammaPrecision(1.0, 0.001)`) until PCAlphaW lands in PR6 (ADR-018).

here <- tryCatch(
    normalizePath(dirname(sys.frame(1)$ofile), mustWork = FALSE),
    error = function(e) getwd()
)
if (!nzchar(here)) here <- getwd()
source(file.path(here, "..", "_helpers.R"))

suppressPackageStartupMessages({
    library(INLA)
})

set.seed(20260430)

n <- 200L
alpha_lp_true <- -0.5      # linear-predictor intercept
beta_true     <- 0.7
shape_true    <- 1.5       # Weibull shape α_w

x <- rnorm(n)
lambda_true <- exp(alpha_lp_true + beta_true * x)
# Inverse CDF sample: T = (-log U / λ)^(1/α_w)
U <- runif(n)
T_true <- (-log(U) / lambda_true) ^ (1.0 / shape_true)
C_i    <- runif(n, 0.5, 4.0)
event  <- as.integer(T_true <= C_i)
y_obs  <- pmin(T_true, C_i)

df <- data.frame(time = y_obs, event = event, x = x)

formula <- inla.surv(time = time, event = event) ~ 1 + x

fit <- INLA::inla(
    formula,
    family = "weibullsurv",
    data = df,
    control.family = list(variant = 0),   # PH parameterisation
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm",
                     "synthetic_weibull_survival.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "synthetic_weibull_survival",
    component_names = character(0),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        n = n,
        family = "weibullsurv",
        variant = 0,
        alpha_lp_true = alpha_lp_true,
        beta_true = beta_true,
        shape_true = shape_true,
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

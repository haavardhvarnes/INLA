# Reference: Exponential survival regression via R-INLA on a synthetic
# right-censored dataset. Smallest oracle for the exponential survival
# pathway — intercept + one covariate, no latent random effect, mixture
# of uncensored and right-censored rows.
#
# Model (R-INLA's `family = "exponentialsurv"`, log link):
#   T_i | η_i ~ Exponential(rate = exp(η_i))
#   η_i      = α + β x_i
#   y_i      = min(T_i, C_i),  event_i = 1{T_i ≤ C_i}
#
# Generative parameters: α = -0.5, β = 0.7, n = 200, censoring time
# C_i ~ Uniform(0.5, 4.0). With these settings ~30% of rows are
# right-censored.
#
# The likelihood has zero hyperparameters (the rate is fully determined
# by η), so this fixture exercises the dim(θ) = 0 path.

here <- tryCatch(
    normalizePath(dirname(sys.frame(1)$ofile), mustWork = FALSE),
    error = function(e) getwd()
)
if (!nzchar(here)) here <- getwd()
source(file.path(here, "..", "_helpers.R"))

suppressPackageStartupMessages({
    library(INLA)
})

set.seed(20260429)

n <- 200L
alpha_true <- -0.5
beta_true  <- 0.7
x <- rnorm(n)
lambda_true <- exp(alpha_true + beta_true * x)
T_true <- rexp(n, rate = lambda_true)
C_i    <- runif(n, 0.5, 4.0)
event  <- as.integer(T_true <= C_i)
y_obs  <- pmin(T_true, C_i)

df <- data.frame(time = y_obs, event = event, x = x)

formula <- inla.surv(time = time, event = event) ~ 1 + x

fit <- INLA::inla(
    formula,
    family = "exponentialsurv",
    data = df,
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm",
                     "synthetic_exponential_survival.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "synthetic_exponential_survival",
    component_names = character(0),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        n = n,
        family = "exponentialsurv",
        alpha_true = alpha_true,
        beta_true = beta_true,
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

# Reference: BetaBinomial regression via R-INLA on a synthetic dataset.
# Smallest possible oracle for the `family = "betabinomial"` pathway —
# intercept + one covariate, no latent random effect, 200 observations.
#
# Model (R-INLA's mean-overdispersion parameterisation):
#   y_i | n_i, η_i, ρ ~ BetaBinomial(n_i, μ_i s, (1 - μ_i) s),
#     μ_i = logit⁻¹(η_i),  s = (1 - ρ) / ρ,
#     E[y_i] = n_i μ_i,
#     Var[y_i] = n_i μ_i (1 - μ_i) (1 + (n_i − 1) ρ)
#   η_i = α + β x_i
#   logit(ρ) ~ Gaussian(0, prec = 0.5)      # R-INLA default for betabinomial
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

n_obs <- 200L
rho_true <- 0.20
s_true <- (1 - rho_true) / rho_true        # ≈ 4
alpha_true <- 0.3
beta_true <- 0.6

x <- rnorm(n_obs)
mu <- 1 / (1 + exp(-(alpha_true + beta_true * x)))

# Per-observation trial counts; vary so the BetaBinomial vs Binomial
# distinction is visible.
n_trials <- sample(5L:25L, n_obs, replace = TRUE)

# Sample y_i ~ BetaBinomial(n_i, μ_i s, (1 − μ_i) s).
y <- integer(n_obs)
for (i in seq_len(n_obs)) {
    a <- mu[i] * s_true
    b <- (1 - mu[i]) * s_true
    p_i <- rbeta(1, a, b)
    y[i] <- rbinom(1, n_trials[i], p_i)
}

df <- data.frame(y = y, x = x, Ntrials = n_trials)

formula <- y ~ 1 + x

# Pin the prior to gaussian(0, prec = 0.5) on logit(ρ) so the R-side
# prior is bit-for-bit identical to Julia's `BetaBinomialLikelihood`
# default `GaussianPrior(0, √2)`.
fit <- INLA::inla(
    formula,
    family = "betabinomial",
    data = df,
    Ntrials = n_trials,
    control.family = list(
        hyper = list(
            theta = list(prior = "gaussian", param = c(0.0, 0.5))
        )
    ),
    control.predictor = list(compute = TRUE),
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm", "synthetic_betabinomial.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "synthetic_betabinomial",
    component_names = character(0),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        n = n_obs,
        family = "betabinomial",
        rho_true = rho_true,
        alpha_true = alpha_true,
        beta_true = beta_true,
        rho_prior = "gaussian(0, prec=0.5) on logit(ρ)"
    )
)

# Append input data so the Julia-side oracle test can re-run the fit.
fixture <- jsonlite::fromJSON(out_path, simplifyVector = FALSE)
fixture$input <- list(
    y = as.integer(df$y),
    x = as.numeric(df$x),
    n_trials = as.integer(df$Ntrials)
)
jsonlite::write_json(
    fixture, out_path,
    auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
)

cat("wrote ", out_path, "\n")

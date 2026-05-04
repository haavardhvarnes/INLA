# Reference: Proportional-odds ordinal regression via R-INLA on a
# synthetic dataset. Smallest possible oracle for the
# `family = "pom"` pathway — one covariate, no intercept (cut points
# absorb it), K = 4 ordered classes, 400 observations.
#
# Model (R-INLA's POM parameterisation):
#   P(y_i ≤ k | η_i, α) = F(α_k − η_i),  k = 1, …, K − 1
#   η_i = β x_i               (no intercept; cut points absorb it)
#   θ_1 = α_1
#   θ_k = log(α_k − α_{k−1}),  k = 2, …, K − 1
#
# Prior on the cut points: a single Dirichlet(γ, …, γ) on the implied
# class probabilities π_k(α) = F(α_k) − F(α_{k−1}) at η = 0. The
# concentration γ is the *only* free hyper-parameter on the prior side
# (theta1 carries it as `param`; theta2..theta_{K−1} are tagged
# `prior = "none"` and are read-only — they share the joint Dirichlet
# with theta1). Default γ = 3 matches `inla.models()$likelihood$pom`.
#
# F is the standard logistic CDF. The fixture stores the input
# vectors so the Julia-side oracle test can re-fit the same data
# without requiring R at test time.

here <- tryCatch(
    normalizePath(dirname(sys.frame(1)$ofile), mustWork = FALSE),
    error = function(e) getwd()
)
if (!nzchar(here)) here <- getwd()
source(file.path(here, "..", "_helpers.R"))

suppressPackageStartupMessages({
    library(INLA)
})

set.seed(20260505)

n <- 400L
K <- 4L
beta_true <- 0.8
# Cut points α = (-1.0, 0.0, 1.2): three distinct gaps and balanced
# class probabilities under η around 0.
alpha_true <- c(-1.0, 0.0, 1.2)
theta_true <- c(alpha_true[1],
                log(alpha_true[2] - alpha_true[1]),
                log(alpha_true[3] - alpha_true[2]))

x <- rnorm(n)
eta <- beta_true * x

# Sample y_i ∈ {1, …, K} via the cumulative-logit construction:
# y_i = 1 + sum_k 1{α_k − η_i < logit(U_i)}, U_i ~ Uniform(0, 1).
y <- integer(n)
for (i in seq_len(n)) {
    u <- runif(1)
    t <- log(u / (1 - u))
    k <- K
    for (j in seq_len(K - 1)) {
        if (alpha_true[j] - eta[i] >= t) {
            k <- j
            break
        }
    }
    y[i] <- k
}

df <- data.frame(y = y, x = x)

# No intercept: cut points absorb it. R-INLA's `pom` enforces this.
formula <- y ~ -1 + x

# R-INLA's `pom` family hard-wires a single Dirichlet prior on the
# implied class probabilities; the concentration `γ` lives on `theta1`
# as the `param` slot. theta2..theta_{K-1} are read-only ("prior = none")
# placeholders — they share the joint Dirichlet with theta1. We pass the
# default γ = 3 explicitly so the prior is bit-for-bit identical to
# Julia's POMLikelihood default (`dirichlet_concentration = 3.0`).
gamma_concentration <- 3.0

fit <- INLA::inla(
    formula,
    family = "pom",
    data = df,
    control.family = list(
        hyper = list(theta1 = list(param = gamma_concentration))
    ),
    control.predictor = list(compute = TRUE),
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm", "synthetic_pom.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "synthetic_pom",
    component_names = character(0),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        n = n,
        family = "pom",
        n_classes = K,
        beta_true = beta_true,
        alpha_true = alpha_true,
        theta_true = theta_true,
        cumulative_link = "logit",
        dirichlet_concentration = gamma_concentration,
        theta_prior = "Dirichlet(γ, ..., γ) on implied class probabilities",
        notes = paste(
            "POM cumulative-logit with K = 4 ordered classes;",
            "cut points absorb the intercept (formula = y ~ -1 + x);",
            "Dirichlet concentration γ = 3 (R-INLA default)."
        )
    )
)

# Append input data so the Julia-side oracle test can re-run the fit.
fixture <- jsonlite::fromJSON(out_path, simplifyVector = FALSE)
fixture$input <- list(
    y = as.integer(df$y),
    x = as.numeric(df$x),
    n_classes = as.integer(K)
)
jsonlite::write_json(
    fixture, out_path,
    auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
)

cat("wrote ", out_path, "\n")

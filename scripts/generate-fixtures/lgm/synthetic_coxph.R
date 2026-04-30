# Reference: Cox proportional-hazards regression via R-INLA on a
# synthetic right-censored dataset. Smallest oracle for the Cox-PH
# pathway.
#
# Model (R-INLA's `family = "coxph"`, equivalent to the Holford /
# Laird-Olivier piecewise-exponential augmentation):
#   T_i | x_i, γ ~ proportional hazards with hazard
#       λ_i(t) = exp(γ_{k(t)} + xᵀ β),
#     γ_k = piecewise-constant baseline log-hazard with RW1 smoothing
#           prior (15 quantile-based breakpoints).
#   y_i = min(T_i, C_i),  event_i = 1{T_i ≤ C_i}.
#
# The R-INLA fit uses `family = "coxph"` which performs the augmentation
# internally. Our `inla_coxph` mirrors that augmentation explicitly, so
# the resulting Poisson-RW1 fit is algebraically identical and posterior
# summaries should match within numerical tolerance.
#
# Generative parameters: β = (0.5, -0.3), n = 400, censoring time
# C_i ~ Uniform(2.5, 6.5). With these settings ~50-60% of rows are events.

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

# --- Truth --------------------------------------------------------------
n         <- 400L
beta_true <- c(0.5, -0.3)

# Piecewise-constant baseline log-hazard on a fixed coarse grid for
# simulation. The R-INLA fit picks its own breakpoints internally; we
# only need the resulting time-to-event distribution.
sim_breakpoints <- seq(0, 5, length.out = 11L)   # 10 intervals
gamma_true <- c(-1.0, -0.7, -0.4, -0.1, 0.1,
                 0.0, -0.2, -0.5, -0.8, -1.1)

# Inverse-CDF sampler from the piecewise-exponential survival distribution
# with covariate shift `xᵀβ`.
sample_pwexp <- function(x, beta, gamma, bp) {
    R    <- -log(runif(1L))
    cum  <- 0.0
    sh   <- as.numeric(crossprod(x, beta))
    K    <- length(bp) - 1L
    for (k in 1:K) {
        lambda <- exp(gamma[k] + sh)
        delta  <- bp[k + 1L] - bp[k]
        if (cum + lambda * delta >= R) {
            return(bp[k] + (R - cum) / lambda)
        }
        cum <- cum + lambda * delta
    }
    return(bp[length(bp)])
}

# --- Simulate -----------------------------------------------------------
X <- matrix(rnorm(n * 2L), nrow = n, ncol = 2L)
T_true <- vapply(seq_len(n), function(i)
    sample_pwexp(X[i, ], beta_true, gamma_true, sim_breakpoints),
    numeric(1))
C_i    <- runif(n, 2.5, 6.5)
event  <- as.integer(T_true <= C_i)
y_obs  <- pmin(T_true, C_i)
# Numerical safety: avoid t = 0 exactly.
y_obs  <- pmax(y_obs, 1e-6)

df <- data.frame(time = y_obs, event = event,
                 x1 = X[, 1L], x2 = X[, 2L])

# --- Fit with R-INLA's coxph family -------------------------------------
# `family = "coxph"` triggers internal augmentation; the baseline hazard
# component is auto-named `baseline.hazard` in the fit object.
formula <- inla.surv(time = time, event = event) ~ -1 + x1 + x2

fit <- INLA::inla(
    formula,
    family = "coxph",
    data   = df,
    control.hazard  = list(
        model       = "rw1",
        n.intervals = 15L,
        scale.model = FALSE,
        hyper       = list(prec = list(prior = "pc.prec",
                                        param = c(1.0, 0.01)))
    ),
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm",
                      "synthetic_coxph.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "synthetic_coxph",
    component_names = c("baseline.hazard"),
    include_marginals = TRUE,
    meta = list(
        dataset    = "synthetic",
        n          = n,
        family     = "coxph",
        beta_true  = beta_true,
        n_events   = sum(event),
        n_censored = sum(1L - event)
    )
)

# Append input data + the breakpoints R-INLA actually used so the
# Julia-side oracle can mirror them exactly.
fixture <- jsonlite::fromJSON(out_path, simplifyVector = FALSE)
# `summary.random$baseline.hazard$ID` is the midpoint of each interval;
# the boundaries are stored on `fit$.args$control.hazard$cutpoints` if
# accessible, otherwise we extract from the augmented data structure.
breakpoints <- if (!is.null(fit$.args$control.hazard$cutpoints)) {
    as.numeric(fit$.args$control.hazard$cutpoints)
} else {
    NULL
}

fixture$input <- list(
    time        = as.numeric(df$time),
    event       = as.integer(df$event),
    X           = lapply(seq_len(nrow(X)), function(i) as.numeric(X[i, ])),
    breakpoints = breakpoints
)
jsonlite::write_json(
    fixture, out_path,
    auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
)

cat("wrote ", out_path, "\n")

# Reference: joint longitudinal (Gaussian) + survival (Weibull-PH)
# regression via R-INLA on a synthetic dataset, mirroring the Baghfalaki
# et al. (2024)-style joint-model regression test in
# `test/regression/test_inla_joint_baghfalaki.jl`.
#
# Model (per subject i = 1..N, j = 1..K longitudinal observations):
#   b_i           ~ N(0, σ_b²)
#   y_{i,j}       ~ N(α_long + b_i, σ_g²)
#   T_i           ~ Weibull(α_w, exp(α_surv + φ · b_i))     (PH form)
# with right-censoring at T_admin.
#
# The R-INLA equivalent uses two intercepts (`intercept_long`,
# `intercept_surv`), one IID(N) random effect named `b_long` carrying
# the subject offsets, and a `copy = "b_long", fixed = FALSE` block on
# the survival arm that introduces a single `beta` hyperparameter `φ`
# to scale the shared subject effect.
#
# Parameters and the seed match the Julia regression test exactly so
# both sides see identical synthetic data:
#   N = 80, K = 5, α_long = 0.4, α_surv = -0.2, σ_b = 0.7, σ_g = 0.5,
#   shape α_w = 1.5, φ = 0.8, T_admin = 3.5.

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

# --- ground-truth parameters ----------------------------------------
N           <- 80L
K           <- 5L
alpha_long  <- 0.4
alpha_surv  <- -0.2
sigma_b     <- 0.7
sigma_g     <- 0.5
shape_w     <- 1.5
phi_true    <- 0.8
T_admin     <- 3.5

# --- simulate -------------------------------------------------------
b_true <- rnorm(N, sd = sigma_b)

y_long <- numeric(N * K)
subj_long <- integer(N * K)
for (i in seq_len(N)) {
    for (j in seq_len(K)) {
        idx <- (i - 1L) * K + j
        y_long[idx] <- alpha_long + b_true[i] + rnorm(1L, sd = sigma_g)
        subj_long[idx] <- i
    }
}
n_long <- length(y_long)

T_event <- numeric(N)
ev_indicator <- integer(N)
for (i in seq_len(N)) {
    lambda_i <- exp(alpha_surv + phi_true * b_true[i])
    u <- runif(1L)
    t_i <- (-log(u) / lambda_i)^(1.0 / shape_w)
    if (t_i > T_admin) {
        T_event[i] <- T_admin
        ev_indicator[i] <- 0L
    } else {
        T_event[i] <- t_i
        ev_indicator[i] <- 1L
    }
}

n_total <- n_long + N

# --- build joint response -------------------------------------------
# Two-arm response: column 1 is Gaussian, column 2 is a surv object
# applied to the survival rows. Off-block rows get NA so R-INLA knows
# which family to use per row.
y_gauss <- c(y_long, rep(NA_real_, N))
surv_time <- c(rep(NA_real_, n_long), T_event)
surv_event <- c(rep(NA_integer_, n_long), ev_indicator)
y_resp <- list(y_gauss, inla.surv(time = surv_time, event = surv_event))

# --- design columns -------------------------------------------------
intercept_long <- c(rep(1.0, n_long), rep(NA_real_, N))
intercept_surv <- c(rep(NA_real_, n_long), rep(1.0, N))
b_long_idx     <- c(subj_long,           rep(NA_integer_, N))
b_surv_idx     <- c(rep(NA_integer_, n_long), seq_len(N))

# --- formula ---------------------------------------------------------
# `b_long` is the IID(N) shared subject effect. The survival arm uses
# `copy = "b_long", fixed = FALSE` to scale it by the learnable β = φ.
# Hyperprior parity with the Julia regression test:
#   - Gaussian likelihood precision τ_g    : pc.prec(u = 1, alpha = 0.01)
#   - IID(N) random-effect precision τ_b   : pc.prec(u = 1, alpha = 0.01)
#   - Weibull shape α_w                    : R-INLA loggamma(1, 0.001) default
#   - Copy β (= φ)                         : R-INLA's wide N(0, 1e3⁻¹) default
formula <- y_resp ~ -1 + intercept_long + intercept_surv +
    f(b_long_idx, model = "iid",
      hyper = list(prec = list(prior = "pc.prec", param = c(1, 0.01)))) +
    f(b_surv_idx, copy = "b_long_idx", fixed = FALSE)

fit <- INLA::inla(
    formula,
    family = c("gaussian", "weibullsurv"),
    data = list(
        y_resp = y_resp,
        intercept_long = intercept_long,
        intercept_surv = intercept_surv,
        b_long_idx = b_long_idx,
        b_surv_idx = b_surv_idx
    ),
    control.family = list(
        list(hyper = list(prec = list(prior = "pc.prec", param = c(1, 0.01)))),
        list(variant = 0L)
    ),
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm", "synthetic_baghfalaki.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "synthetic_baghfalaki",
    component_names = c("b_long_idx", "b_surv_idx"),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        N = N,
        K = K,
        alpha_long_true = alpha_long,
        alpha_surv_true = alpha_surv,
        sigma_b_true = sigma_b,
        sigma_g_true = sigma_g,
        shape_true = shape_w,
        phi_true = phi_true,
        T_admin = T_admin
    )
)

# Append input data so the Julia-side oracle can re-fit the same
# dataset without re-running R.
fixture <- jsonlite::fromJSON(out_path, simplifyVector = FALSE)
fixture$input <- list(
    y_long       = as.numeric(y_long),
    subj_long    = as.integer(subj_long),
    T_event      = as.numeric(T_event),
    ev_indicator = as.integer(ev_indicator),
    b_true       = as.numeric(b_true)
)
jsonlite::write_json(
    fixture, out_path,
    auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
)

cat("wrote ", out_path, "\n")

# Reference: bivariate IID random effects (R-INLA's `2diid`) on synthetic
# data. Smallest oracle for the IIDND_Sep{2}/IID2D component introduced
# in Phase I-A PR-1a (ADR-022).
#
# Model (per group i = 1..n_groups, replicate k = 1..m):
#   y_1_{i,k} = β_1 + b_1_i + ε_1_{i,k}
#   y_2_{i,k} = β_2 + b_2_i + ε_2_{i,k}
#   (b_1_i, b_2_i) ~ N₂(0, Σ),  Σ = inv(Λ)
#   Λ = (τ_1   -ρ √(τ_1 τ_2);  -ρ √(τ_1 τ_2)   τ_2) / (1 - ρ²)
#   ε ~ N(0, τ_y⁻¹)
#   τ_1, τ_2 ~ pc.prec(u = 1, α = 0.01)        # match Julia PCPrecision()
#   ρ        ~ pc.cor0(U = 0.5, α = 0.5)       # match Julia PCCor0()
#   τ_y      ~ pc.prec(u = 1, α = 0.01)        # match Julia GaussianLikelihood()
#   β_1, β_2 ~ N(0, 1e-3⁻¹)                    # R-INLA default fixed-effect
#
# m > 1 replication per (group, dimension) is required to identify τ_y
# separately from (τ_1, τ_2). With m = 1 each ε_{i,k} is perfectly
# confounded with b_d_i and only the sum-of-variances `1/τ_d + 1/τ_y` is
# identifiable, leaving τ_y with a near-prior posterior.
#
# R-INLA `2diid` internal layout: INTERLEAVED by group, not contiguous —
#   slot 2i - 1 → b_1[i]  (dim 1, group i)
#   slot 2i     → b_2[i]  (dim 2, group i)
# Verified by inspecting `fit$misc$configs$config[[1]]$Q` on a fixed-θ
# fit: the 2×2 diagonal Λ-blocks couple consecutive slots (1↔2, 3↔4, …).
# Our Julia `IIDND_Sep{2}` uses the contiguous layout `[b_1[1..n], b_2[1..n]]`;
# this is a R-INLA-side implementation detail that does not affect the
# user-scale posteriors of the hyperparameters or the fixed effects.
#
# Therefore the R-side data is laid out in INTERLEAVED order:
#   row 2i - 1 → y_1[i]  (dim 1, group i, picks slot 2i-1)
#   row 2i     → y_2[i]  (dim 2, group i, picks slot 2i)
# The Julia oracle test reorders the input vectors back to the contiguous
# `[y_1; y_2]` layout that the Julia model expects.
#
# Hyperparameter naming for `2diid`: `theta1` = log τ_1, `theta2` = log τ_2,
# `theta3` = log((1+ρ)/(1-ρ)) = 2·atanh(ρ). Our Julia `IIDND_Sep{2}` uses
# `θ_3 = atanh(ρ)`; the factor of 2 is purely an internal-scale convention
# and does not affect user-scale posteriors (which are reported on `ρ` in
# both implementations). The PC prior on `ρ` is identical regardless.

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

n_groups <- 30L
m_reps   <- 5L
beta_1_true <- 1.0
beta_2_true <- -0.5
tau_1_true  <- 2.0
tau_2_true  <- 4.0
rho_true    <- 0.5
tau_y_true  <- 25.0   # observation noise sd = 0.2

# Marginal covariance of (b_1, b_2) — closed-form inverse of Λ.
Sigma <- matrix(
    c(1 / tau_1_true,                      rho_true / sqrt(tau_1_true * tau_2_true),
      rho_true / sqrt(tau_1_true * tau_2_true), 1 / tau_2_true),
    nrow = 2L, ncol = 2L
)

# Sample b: n_groups × 2; b[, 1] is dimension 1, b[, 2] is dimension 2.
L_chol <- chol(Sigma)
Z <- matrix(rnorm(n_groups * 2L), nrow = n_groups, ncol = 2L)
b <- Z %*% L_chol

# Observations: m replicates per (group, dim), all dim-1 first then dim-2.
# Stored as length-(n_groups * m_reps) vectors; each element is keyed by
# (group, replicate). y_1_long[(i-1)*m + k] = y_1[i, k].
y_1_long <- rep(beta_1_true + b[, 1L], each = m_reps) +
    rnorm(n_groups * m_reps, sd = 1 / sqrt(tau_y_true))
y_2_long <- rep(beta_2_true + b[, 2L], each = m_reps) +
    rnorm(n_groups * m_reps, sd = 1 / sqrt(tau_y_true))

# Long-format data: 2 * n_groups * m_reps observations. Layout: all
# dim-1 observations first (in group-major order), then all dim-2.
# `2diid`'s INTERLEAVED slot layout means dim-1 obs maps to slot
# `2*i - 1` and dim-2 obs to slot `2*i` for group i.
y <- c(y_1_long, y_2_long)
N_obs <- length(y)
intercept_1 <- c(rep(1.0, n_groups * m_reps), rep(0.0, n_groups * m_reps))
intercept_2 <- c(rep(0.0, n_groups * m_reps), rep(1.0, n_groups * m_reps))
group_idx <- rep(seq_len(n_groups), each = m_reps)   # group index per row, dim-1 block
idx <- c(2L * group_idx - 1L,    # dim-1 rows pick slot 2i-1 (b_1[i])
         2L * group_idx)         # dim-2 rows pick slot 2i   (b_2[i])

prec_pc <- list(prior = "pc.prec", param = c(1, 0.01))
cor_pc  <- list(prior = "pc.cor0", param = c(0.5, 0.5))

formula <- y ~ -1 + intercept_1 + intercept_2 +
    f(idx, model = "2diid", n = 2L * n_groups,
      hyper = list(theta1 = prec_pc, theta2 = prec_pc, theta3 = cor_pc))

fit <- INLA::inla(
    formula,
    family = "gaussian",
    data = list(y = y, intercept_1 = intercept_1,
                intercept_2 = intercept_2, idx = idx),
    control.family = list(
        hyper = list(prec = list(prior = "pc.prec", param = c(1, 0.01)))
    ),
    control.predictor = list(compute = TRUE),
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm", "synthetic_iid2d.json")

write_inla_fixture(
    fit = fit, path = out_path, name = "synthetic_iid2d",
    component_names = c("idx"),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        n_groups = n_groups,
        m_reps   = m_reps,
        beta_1_true = beta_1_true,
        beta_2_true = beta_2_true,
        tau_1_true  = tau_1_true,
        tau_2_true  = tau_2_true,
        rho_true    = rho_true,
        tau_y_true  = tau_y_true
    )
)

# Append input data so the Julia oracle can re-fit without R.
fixture <- jsonlite::fromJSON(out_path, simplifyVector = FALSE)
fixture$input <- list(
    y_1 = as.numeric(y_1_long),
    y_2 = as.numeric(y_2_long),
    n_groups = n_groups,
    m_reps   = m_reps
)
jsonlite::write_json(
    fixture, out_path,
    auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
)

cat("wrote ", out_path, "\n")

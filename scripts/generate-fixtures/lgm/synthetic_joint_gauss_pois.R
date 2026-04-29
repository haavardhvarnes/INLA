# Reference: joint Gaussian + Poisson regression via R-INLA on a
# synthetic dataset. Smallest oracle for the multi-likelihood pathway
# introduced in Phase G PR2 (ADR-017): two observation blocks share one
# `IID(n)` random effect plus a shared intercept.
#
# Model (per site i = 1..n):
#   y_g_i | u_i ~ Normal(α + u_i, τ_g⁻¹)
#   y_p_i | u_i ~ Poisson(exp(α + u_i))
#   u_i ~ N(0, τ_u⁻¹)
#   τ_g ~ pc.prec(u = 1, α = 0.01)  # matches Julia's GaussianLikelihood default
#   τ_u ~ pc.prec(u = 1, α = 0.01)  # matches Julia's IID default
#   α   ~ N(0, 1e-3⁻¹)              # R-INLA default for intercept
#
# Block layout (matches Julia-side StackedMapping):
#   rows 1..n         — Gaussian block, mapping = [1 I_n]
#   rows (n+1)..(2n)  — Poisson block,  mapping = [1 I_n]
#
# The fixture stores the inputs so the Julia oracle can re-fit the same
# data without requiring R at test time.

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

n <- 50L
alpha_true <- 0.3
tau_g_true <- 4.0          # Gaussian precision (sd = 0.5)
tau_u_true <- 9.0          # IID precision (sd ≈ 0.33)
u_true <- rnorm(n, sd = 1 / sqrt(tau_u_true))

eta <- alpha_true + u_true
y_g <- rnorm(n, mean = eta, sd = 1 / sqrt(tau_g_true))
y_p <- rpois(n, lambda = exp(eta))

# Two-column response: row k of column 1 carries Gaussian y_g_k for
# k=1..n; row k of column 2 carries Poisson y_p_{k-n} for k=(n+1)..(2n).
# NA on the off-block column tells R-INLA which family applies.
Y <- matrix(NA, nrow = 2L * n, ncol = 2L)
Y[1:n, 1L] <- y_g
Y[(n + 1L):(2L * n), 2L] <- y_p

idx <- c(seq_len(n), seq_len(n))           # site index, shared
intercept <- rep(1.0, 2L * n)              # explicit intercept column

# Match Julia defaults: `GaussianLikelihood()` and `IID(n)` both default
# to `PCPrecision(1.0, 0.01)` — the PC prior with `P(σ > 1) = 0.01`. The
# R-INLA equivalent is `prior = "pc.prec", param = c(u = 1, alpha = 0.01)`.
formula <- Y ~ -1 + intercept +
    f(idx, model = "iid",
      hyper = list(prec = list(prior = "pc.prec", param = c(1, 0.01))))

fit <- INLA::inla(
    formula,
    family = c("gaussian", "poisson"),
    data = list(Y = Y, idx = idx, intercept = intercept),
    control.family = list(
        list(hyper = list(prec = list(prior = "pc.prec", param = c(1, 0.01)))),
        list()
    ),
    control.predictor = list(compute = TRUE),
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm", "synthetic_joint_gauss_pois.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "synthetic_joint_gauss_pois",
    component_names = c("idx"),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        n = n,
        families = c("gaussian", "poisson"),
        alpha_true = alpha_true,
        tau_g_true = tau_g_true,
        tau_u_true = tau_u_true
    )
)

# Append input data so the Julia-side oracle test can re-run the fit.
fixture <- jsonlite::fromJSON(out_path, simplifyVector = FALSE)
fixture$input <- list(
    y_g = as.numeric(y_g),
    y_p = as.integer(y_p)
)
jsonlite::write_json(
    fixture, out_path,
    auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
)

cat("wrote ", out_path, "\n")

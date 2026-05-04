# Reference: Poisson + Intercept + Generic0 (RW1 + sum-to-zero) with
# unit-scale latent variance via R-INLA. Targets the "Brunei pathology"
# of sharply non-Gaussian latents (`plans/replan-2026-04-28.md` §Phase L
# Acceptance, ADR-026): a regime with low expected counts E_i where
# `b_i | y, θ` is strongly non-Gaussian and requires the per-`x_i`
# refitted Laplace (R-INLA `strategy = "laplace"`) — what we ship as
# `LatentGaussianModels.FullLaplace`.
#
# Modern R-INLA (25.10.19) routes all `control.inla(strategy = …)` paths
# through a unified VB-corrected pipeline, so the per-strategy
# `summary.random` outputs converge to a single canonical answer. The
# fixture therefore stores one fit (strategy = "laplace") as the
# reference. The Julia oracle test (`test_synthetic_brunei.jl`) asserts
# Julia `FullLaplace` matches it within Phase F tolerances and that
# Julia `SimplifiedLaplace` produces a measurable gap on the same
# coordinates — i.e. the per-`x_i` strategy is doing real work, not a
# pass-through.
#
# Model:
#   y_i | η_i ~ Poisson(E_i · exp(η_i))
#   η_i = α + b_i
#   b ~ N(0, (τ R)⁻¹), R = RW1 precision (n × n, rank n - 1)
#   1·b = 0                          # sum-to-zero constraint (extraconstr)
#   τ ~ logGamma(100, 100)           # tight prior pinning τ ≈ 1.0 so the
#                                    # latent has unit-scale variance and
#                                    # the conditional `b_i | y, θ` is
#                                    # measurably non-Gaussian.

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

n <- 24L

# RW1 precision matrix (rank-deficient, sum-to-zero null space).
R_rw1 <- matrix(0, n, n)
diag(R_rw1) <- c(1, rep(2, n - 2), 1)
for (k in seq_len(n - 1)) {
    R_rw1[k, k + 1] <- -1
    R_rw1[k + 1, k] <- -1
}
Rsp <- Matrix::Matrix(R_rw1, sparse = TRUE)

# Deterministic oscillatory latent with sum-to-zero. Strong variation
# (peak-to-trough ~3) plus low expected counts E_i drive a meaningful
# fraction of `y_i = 0` and `y_i ≥ 4` — the regime where the
# conditional `b_i | y, θ` posterior is sharply non-Gaussian.
alpha_true <- -0.4
b_true <- 1.5 * sin(2 * pi * seq_len(n) / n)
b_true <- b_true - mean(b_true)
stopifnot(abs(sum(b_true)) < 1e-12)

E_vec <- rep(1.0, n)
mu <- E_vec * exp(alpha_true + b_true)
y <- rpois(n, mu)

# Sanity: keep going only if the data exhibit the low-count regime that
# motivates the FL fit (most y in {0, 1, 2, 3}). Aborting here lets
# future regenerations notice if the seed lands on a non-pathological
# draw.
stopifnot(mean(y <= 3) >= 0.7)

dat <- data.frame(y = y, idx = seq_len(n))

formula <- y ~ 1 +
    f(idx, model = "generic0", Cmatrix = Rsp,
      rankdef = 1L,
      extraconstr = list(A = matrix(1, 1, n), e = 0),
      hyper = list(prec = list(prior = "loggamma",
                                param = c(100, 100))))

fit_fl <- INLA::inla(
    formula,
    family = "poisson",
    data = dat,
    E = E_vec,
    control.predictor = list(compute = TRUE),
    control.compute = list(return.marginals = TRUE),
    control.inla = list(strategy = "laplace")
)

out_path <- file.path(here, "..", "fixtures", "lgm", "synthetic_brunei.json")

write_inla_fixture(
    fit = fit_fl,
    path = out_path,
    name = "synthetic_brunei",
    component_names = c("idx"),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        n = n,
        family = "poisson",
        model = "generic0 (RW1 + sum-to-zero)",
        scale_E = 1.0,
        alpha_true = alpha_true,
        prec_prior = "loggamma(100, 100)",
        strategy = "laplace"
    )
)

# Append input data so the Julia-side oracle test can re-run the fit
# without an R-side dependency.
fixture <- jsonlite::fromJSON(out_path, simplifyVector = FALSE)
fixture$input <- list(
    y = as.integer(dat$y),
    E = as.numeric(E_vec),
    R = sparse_to_triplet(Rsp),
    n = n
)
jsonlite::write_json(
    fixture, out_path,
    auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
)

cat("wrote ", out_path, "\n")
cat("y =", paste(y, collapse = " "), "\n")
cat("FL τ mean =", fit_fl$summary.hyperpar["Precision for idx", "mean"], "\n")
cat("FL idx[1..3] mean =",
    paste(round(fit_fl$summary.random$idx$mean[1:3], 4), collapse = " "), "\n")
cat("FL idx[1..3] sd =",
    paste(round(fit_fl$summary.random$idx$sd[1:3], 4), collapse = " "), "\n")

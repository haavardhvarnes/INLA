# Reference: replicated AR1 random effect (R-INLA's `f(t, model="ar1",
# replicate=id)`) on synthetic data. Smallest oracle for the `Replicate`
# component introduced in Phase I-C PR-3a.
#
# Model (per replicate r = 1..R, time t = 1..n):
#   x^{r}_t = ρ x^{r}_{t-1} + ε,   ε ~ N(0, (1-ρ²)/τ)
#   marginal:  x^{r}_t ~ N(0, 1/τ)
#   y^{r}_t   = x^{r}_t + ν,        ν ~ N(0, 1/τ_y)
#   τ, τ_y    ~ pc.prec(u=1, α=0.01)            # Julia PCPrecision()
#   ρ         ~ Normal(0, σ=1) on atanh(ρ)      # Julia AR1 default
#
# Prior matching: Julia's AR1 default is `_NormalAR1ρ(0, σ=1)` on the
# Fisher-z scale `atanh(ρ)`. R-INLA's internal `theta2` is
# `logit(ρ) = 2·atanh(ρ)`, so a `Normal(0, prec=p)` on R-INLA's internal
# scale becomes `Normal(0, prec=4p)` on `atanh(ρ)`. To match Julia's
# σ=1 (prec=1) on `atanh(ρ)`, we set R's `prior="normal", param=c(0, 0.25)`
# on `logit(ρ)`. (R-INLA's *built-in* AR1 default is prec=0.15 on logit;
# we override here so the two priors are bit-for-bit identical.)
#
# `replicate=id` shares (τ, ρ) across the R independent chains; the
# stacked latent has length R·n with precision blockdiag(Q_inner, …, Q_inner).

here <- tryCatch(
    normalizePath(dirname(sys.frame(1)$ofile), mustWork = FALSE),
    error = function(e) getwd()
)
if (!nzchar(here)) here <- getwd()
source(file.path(here, "..", "_helpers.R"))

suppressPackageStartupMessages({
    library(INLA)
})

set.seed(20260508)

# n_replicates=30 panels of length n=20 chosen to sharpen τ_y
# identifiability: at smaller (R, n) the τ_y posterior is heavy-tailed
# (the AR1 latent absorbs most of the residual variance). With 600
# observations all three hyperparameters have well-defined posteriors.
n <- 20L
R <- 30L
N <- R * n
tau_true   <- 2.0
rho_true   <- 0.5
sigma_y    <- 0.4
sigma_x    <- 1 / sqrt(tau_true)

# Per-replicate AR1 chain in the marginal-variance Rue-Held parameterisation.
x_true <- numeric(N)
for (r in seq_len(R)) {
    x_true[(r - 1L) * n + 1L] <- sigma_x * rnorm(1)
    for (t in 2:n) {
        i <- (r - 1L) * n + t
        x_true[i] <- rho_true * x_true[i - 1L] +
            sqrt(1 - rho_true^2) * sigma_x * rnorm(1)
    }
}
y <- x_true + rnorm(N, sd = sigma_y)

# R-INLA layout: t_idx is the within-chain time index (1..n) and
# repl_idx is the chain id (1..R). The replicate machinery stacks the
# inner AR1 across the R chains automatically.
t_idx    <- rep(seq_len(n), times = R)
repl_idx <- rep(seq_len(R), each = n)

prec_pc <- list(prior = "pc.prec", param = c(1, 0.01))
rho_pr  <- list(prior = "normal",  param = c(0, 0.25))

formula <- y ~ -1 + f(t_idx, model = "ar1", replicate = repl_idx,
                       hyper = list(prec = prec_pc, rho = rho_pr))

fit <- INLA::inla(
    formula,
    family = "gaussian",
    data = list(y = y, t_idx = t_idx, repl_idx = repl_idx),
    control.family = list(hyper = list(prec = prec_pc)),
    control.predictor = list(compute = TRUE),
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm", "synthetic_replicate_ar1.json")

write_inla_fixture(
    fit = fit, path = out_path, name = "synthetic_replicate_ar1",
    component_names = c("t_idx"),
    include_marginals = TRUE,
    meta = list(
        dataset      = "synthetic",
        n            = n,
        R            = R,
        tau_true     = tau_true,
        rho_true     = rho_true,
        sigma_y_true = sigma_y
    )
)

# Append input data so the Julia oracle can re-fit without R.
fixture <- jsonlite::fromJSON(out_path, simplifyVector = FALSE)
fixture$input <- list(
    y = as.numeric(y),
    n = n,
    R = R
)
jsonlite::write_json(
    fixture, out_path,
    auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
)

cat("wrote ", out_path, "\n")

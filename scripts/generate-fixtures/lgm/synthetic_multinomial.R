# Reference: multinomial-logit regression via R-INLA's
# Multinomial-INLA (Baker 1994; Chen 1985) reformulation. See Phase J
# PR-7 / ADR-024.
#
# Model:
#   For row i = 1, ..., n with covariate x_i ~ N(0, 1):
#     Y_i ~ Multinomial(N_trials, π_i)
#     π_ik = exp(x_i β_k) / Σ_{k'} exp(x_i β_{k'}),  k = 1, ..., K
#     β_K = 0  (reference class identifiability)
#
# Independent-Poisson reformulation (R-INLA recipe):
#   Y_ik ~ Poisson(λ_ik)
#   λ_ik = exp(α_i + x_i β_k)
#   α_i ~ IID with prec = list(initial = -10, fixed = TRUE)
#         (per-row nuisance intercept; very small fixed precision so
#         α_i absorbs the row-sum information without contributing to
#         the β posterior).
#   β_k ~ N(0, 1e-3⁻¹), k < K  (R-INLA default fixed-effect prior).
#
# Long-format layout (matches `multinomial_to_poisson` in
# LatentGaussianModels.jl):
#   idx = (i - 1) * K + k       i ∈ 1..n, k ∈ 1..K
#   y[idx]        = Y[i, k]
#   row_id[idx]   = i
#   class_id[idx] = k
#
# Fixture stores y, row_id, class_id, x, n, K, N_trials, β_true so the
# Julia oracle can re-fit and compare β posterior moments.

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

n_rows   <- 200L
K        <- 3L
p        <- 1L                 # one covariate per class block
N_trials <- 5L
beta_true <- c(0.7, -0.4)      # β_1, β_2; β_3 = 0 (reference)

x <- rnorm(n_rows)

# Sample multinomial counts per row.
Y <- matrix(0L, nrow = n_rows, ncol = K)
for (i in seq_len(n_rows)) {
    eta <- c(beta_true * x[i], 0)
    p_i <- exp(eta - max(eta))
    p_i <- p_i / sum(p_i)
    Y[i, ] <- as.integer(rmultinom(1, size = N_trials, prob = p_i))
}

# Long-format expansion: row-major. idx = (i - 1) * K + k.
n_long <- n_rows * K
y <- integer(n_long)
row_id <- integer(n_long)
class_id <- integer(n_long)
for (i in seq_len(n_rows)) {
    for (k in seq_len(K)) {
        idx <- (i - 1L) * K + k
        y[idx] <- Y[i, k]
        row_id[idx] <- i
        class_id[idx] <- k
    }
}

# Class-specific covariate columns. Reference class K -> all-zero
# entry (β_K identification constraint).
x_class1 <- ifelse(class_id == 1L, x[row_id], 0)
x_class2 <- ifelse(class_id == 2L, x[row_id], 0)

formula <- y ~ -1 + x_class1 + x_class2 +
    f(row_id, model = "iid", n = n_rows,
      hyper = list(prec = list(initial = -10, fixed = TRUE)))

fit <- INLA::inla(
    formula,
    family = "poisson",
    data = list(
        y = y,
        x_class1 = x_class1,
        x_class2 = x_class2,
        row_id = row_id
    ),
    control.predictor = list(compute = TRUE),
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm", "synthetic_multinomial.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "synthetic_multinomial",
    component_names = c("row_id"),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        n_rows = n_rows,
        K = K,
        p = p,
        N_trials = N_trials,
        beta_true = beta_true,
        reference_class = K,
        nuisance_alpha_prec_initial = -10,
        nuisance_alpha_prec_fixed = TRUE,
        notes = paste(
            "Multinomial-logit reformulated as N*K independent",
            "Poissons (Baker 1994; Chen 1985 / ADR-024). Per-row",
            "α_i is a fixed-precision IID nuisance intercept.",
            "Reference class K has β = 0."
        )
    )
)

# Append the input long-format layout so the Julia oracle can
# re-build the design matrix without R.
fixture <- jsonlite::fromJSON(out_path, simplifyVector = FALSE)
fixture$input <- list(
    y        = as.integer(y),
    row_id   = as.integer(row_id),
    class_id = as.integer(class_id),
    x        = as.numeric(x),
    n_rows   = as.integer(n_rows),
    K        = as.integer(K),
    p        = as.integer(p),
    N_trials = as.integer(N_trials)
)
jsonlite::write_json(
    fixture, out_path,
    auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
)

cat("wrote ", out_path, "\n")

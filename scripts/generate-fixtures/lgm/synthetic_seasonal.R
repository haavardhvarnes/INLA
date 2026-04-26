# Reference: Gaussian time series with an intercept + R-INLA's
# `model = "seasonal"` random effect. Validates the Seasonal LGM
# component against R-INLA's seasonal model end-to-end.
#
# Setup:
#   period s = 6, n = 6 * 8 = 48 observations
#   y_t = α + b_t + ε_t,  ε_t ~ N(0, σ²)
#   b ~ Seasonal(s = 6)             # null space: period-6 zero-sum sequences
#   τ ~ logGamma(1, 5e-5)           # R-INLA default for `seasonal`
#   1/σ² ~ logGamma(1, 5e-5)        # R-INLA default likelihood prec

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

s <- 6L
n <- 48L
stopifnot(n %% s == 0)

# True period-s pattern, zero-mean within one period.
pattern <- rnorm(s)
pattern <- pattern - mean(pattern)
b_true <- rep(pattern, n / s)

alpha_true <- 1.4
sigma_true <- 0.4
y <- alpha_true + b_true + rnorm(n, sd = sigma_true)

dat <- list(y = y, t = seq_len(n))

formula <- y ~ 1 + f(t, model = "seasonal", season.length = s,
                     hyper = list(prec = list(prior = "loggamma",
                                              param = c(1, 5e-5))))

fit <- INLA::inla(
    formula,
    family = "gaussian",
    data = dat,
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "fixtures", "lgm", "synthetic_seasonal.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "synthetic_seasonal",
    component_names = c("t"),
    include_marginals = TRUE,
    meta = list(
        dataset = "synthetic",
        n = n,
        period = s,
        family = "gaussian",
        model = "seasonal",
        prec_prior = "loggamma(1, 5e-5)"
    )
)

# Append input data for the Julia oracle re-fit.
fixture <- jsonlite::fromJSON(out_path, simplifyVector = FALSE)
fixture$input <- list(
    y = as.numeric(y),
    n = n,
    period = s
)
jsonlite::write_json(
    fixture, out_path,
    auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
)

cat("wrote ", out_path, "\n")

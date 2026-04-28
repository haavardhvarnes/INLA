# Reference: BYM2 on Pennsylvania lung cancer via R-INLA. Second
# Poisson-BYM2 oracle alongside Scotland — 67 counties with a proper
# covariate (smoking rate) and expected counts derived from
# indirect standardisation.
#
# Model:
#   y_i | η_i ~ Poisson(E_i · exp(η_i))
#   η_i = β_0 + x_i β_x + b_i          # b = (1/√τ)(√(1-φ) v + √φ u*)
#   v ~ N(0, I)
#   u* ~ scaled-Besag(W)               # Sørbye-Rue scaled, sum-to-zero
#   1/τ ~ PC(U = 1, α = 0.01)
#   φ   ~ PC-BYM2-φ(U = 0.5, α = 2/3)
#
# Dataset: SpatialEpi::pennLC. Strata are (race × gender × age);
# `SpatialEpi::expected` performs indirect standardisation by strata.
#
# Output: fixtures/lgm/pennsylvania_bym2.json (converted via
# convert_to_jld2.jl to packages/LatentGaussianModels.jl/test/oracle/
# fixtures/pennsylvania_bym2.jld2).

here <- tryCatch(
    normalizePath(dirname(sys.frame(1)$ofile), mustWork = FALSE),
    error = function(e) getwd()
)
if (!nzchar(here)) here <- getwd()
source(file.path(here, "..", "_helpers.R"))

# SpatialEpi is installed to the user library; ensure it is on libPaths.
user_lib <- Sys.getenv("R_LIBS_USER")
if (nzchar(user_lib) && dir.exists(user_lib)) {
    .libPaths(c(user_lib, .libPaths()))
}

suppressPackageStartupMessages({
    library(INLA)
    library(SpatialEpi)
    library(spdep)
})

set.seed(20260423)

data(pennLC)

# Aggregate by county. The per-stratum layout is (county, race, gender, age).
# We preserve strata order (16 per county) to feed into
# SpatialEpi::expected for indirect standardisation.
dat_raw <- pennLC$data
counties <- levels(dat_raw$county)
n <- length(counties)

cases_per_county <- tapply(dat_raw$cases, dat_raw$county, sum)
cases <- as.integer(cases_per_county[counties])

# Indirect standardisation: reference rates computed across strata, then
# applied per-county to get expected counts. n.strata = 16 for pennLC
# (2 race × 2 gender × 4 age).
n_strata <- 2L * 2L * 4L
E <- SpatialEpi::expected(
    population = dat_raw$population,
    cases      = dat_raw$cases,
    n.strata   = n_strata
)
# `expected` returns a vector of length n in county order.
stopifnot(length(E) == n)

# Covariate: smoking rate, matched to counties (the smoking frame is
# also per-county, aligned to the `county` factor levels).
smk <- pennLC$smoking
stopifnot(all(smk$county == counties))
x <- scale(smk$smoking)[, 1]

# Neighbourhood graph from the county polygons.
nb <- spdep::poly2nb(pennLC$spatial.polygon)
W <- spdep::nb2mat(nb, style = "B", zero.policy = TRUE)
stopifnot(nrow(W) == n)

graph_file <- tempfile(fileext = ".graph")
INLA::inla.write.graph(W, file = graph_file)

df <- data.frame(
    cases    = cases,
    expected = as.numeric(E),
    x        = as.numeric(x),
    region   = seq_len(n)
)

formula <- cases ~ 1 + x +
    f(region, model = "bym2", graph = graph_file,
      scale.model = TRUE,
      hyper = list(
          prec = list(prior = "pc.prec", param = c(1, 0.01)),
          phi  = list(prior = "pc",      param = c(0.5, 2 / 3))
      ))

fit <- INLA::inla(
    formula,
    family = "poisson",
    data = df,
    E = df$expected,
    control.predictor = list(compute = TRUE),
    control.compute = list(return.marginals = TRUE)
)

# Second fit under R-INLA's `strategy = "simplified.laplace"` to provide
# an oracle for INLA.jl's `latent_strategy = :simplified_laplace` (ADR-016).
# We only need the BYM2 latent posterior mean here; marginals and DIC
# are skipped for speed.
fit_sla <- INLA::inla(
    formula,
    family = "poisson",
    data = df,
    E = df$expected,
    control.predictor = list(compute = FALSE),
    control.compute = list(return.marginals = FALSE),
    control.inla = list(strategy = "simplified.laplace")
)

out_path <- file.path(here, "..", "fixtures", "lgm", "pennsylvania_bym2.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "pennsylvania_bym2",
    component_names = c("region"),
    include_marginals = TRUE,
    meta = list(
        dataset = "SpatialEpi::pennLC",
        n = n,
        family = "poisson",
        scale_model = TRUE,
        covariates = "smoking rate (standardized)",
        standardization = paste0("indirect, n.strata=", n_strata),
        prec_prior = "pc.prec(1, 0.01)",
        phi_prior = "pc(0.5, 2/3)"
    )
)

# Append input data so the Julia-side oracle test can re-run the same
# fit without SpatialEpi / spdep dependencies, plus the SLA-strategy
# BYM mean for ADR-016 oracle coverage. Length is 2n: first n are the
# joint b = (1/√τ)(√(1-φ) v + √φ u*), next n are the unstructured u*.
fixture <- jsonlite::fromJSON(out_path, simplifyVector = FALSE)
fixture$input <- list(
    cases    = as.integer(df$cases),
    expected = as.numeric(df$expected),
    x        = as.numeric(df$x),
    W        = sparse_to_triplet(Matrix::Matrix(W, sparse = TRUE))
)
fixture$bym_mean_sla <- as.numeric(fit_sla$summary.random$region$mean)
jsonlite::write_json(
    fixture, out_path,
    auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
)

cat("wrote ", out_path, "\n")

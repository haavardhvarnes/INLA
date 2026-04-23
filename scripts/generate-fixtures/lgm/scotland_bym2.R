# Reference: BYM2 on Scotland lip cancer via R-INLA. This is the MVP
# go/no-go oracle for the Julia-native LGM stack
# (LatentGaussianModels.jl + GMRFs.jl).
#
# Model:
#   y_i | η_i ~ Poisson(E_i · exp(η_i))
#   η_i = β_0 + x_i β_x + b_i          # b = (1/√τ)(√(1-φ) v + √φ u*)
#   v ~ N(0, I)
#   u* ~ scaled-Besag(W)               # Sørbye-Rue scaled, sum-to-zero
#   1/τ ~ PC(U = 1, α = 0.01)
#   φ   ~ PC-BYM2-φ(U = 0.5, α = 2/3)
#
# Dataset: SpatialEpi::scotland.
#
# Output: fixtures/lgm/scotland_bym2.json (converted via
# convert_to_jld2.jl to packages/LatentGaussianModels.jl/test/oracle/
# fixtures/scotland_bym2.jld2).

here <- tryCatch(
    normalizePath(dirname(sys.frame(1)$ofile), mustWork = FALSE),
    error = function(e) getwd()
)
if (!nzchar(here)) here <- getwd()
source(file.path(here, "..", "_helpers.R"))

suppressPackageStartupMessages({
    library(INLA)
    library(SpatialEpi)
    library(spdep)
})

set.seed(20260423)

data(scotland)
dat <- scotland$data
nb <- spdep::poly2nb(scotland$spatial.polygon)
W <- spdep::nb2mat(nb, style = "B", zero.policy = TRUE)
n <- nrow(W)
graph_file <- tempfile(fileext = ".graph")
INLA::inla.write.graph(W, file = graph_file)

dat$region <- seq_len(n)

# Standardise AFF covariate to match the common published preprocessing.
dat$x <- scale(dat$AFF)[, 1]

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
    data = dat,
    E = dat$expected,
    control.predictor = list(compute = TRUE),
    control.compute = list(return.marginals = TRUE)
)

out_path <- file.path(here, "..", "..", "..",
                      "packages", "LatentGaussianModels.jl", "test", "oracle",
                      "fixtures", "scotland_bym2.json")

write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "scotland_bym2",
    component_names = c("region"),
    include_marginals = TRUE,
    meta = list(
        dataset = "SpatialEpi::scotland",
        n = n,
        family = "poisson",
        scale_model = TRUE,
        covariates = "AFF (standardized)",
        prec_prior = "pc.prec(1, 0.01)",
        phi_prior = "pc(0.5, 2/3)"
    )
)

cat("wrote ", out_path, "\n")

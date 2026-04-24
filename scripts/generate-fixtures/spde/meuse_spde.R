# Reference R-INLA SPDE fit on the Meuse zinc dataset. The canonical
# geostatistical end-to-end test for INLASPDE.jl + LatentGaussianModels.jl.
#
# Model:
#   log(zinc_i) = β_0 + β_dist · dist_i + u(s_i) + ε_i
#   u(s)  ~ SPDE-Matérn, α = 2  (ν = 1)
#   ε_i   ~ N(0, 1/τ)
#   (β_0, β_dist)     ~ N(0, 1000)                 — matches Julia defaults
#   1/τ               ~ PC-prec(U = 1, α = 0.01)    — matches Julia default
#   range             ~ PC(U = 0.5, α = 0.5)
#   σ                 ~ PC(U = 1,   α = 0.5)
#
# Coordinates are in km (Meuse is in metres; we divide by 1000 so the
# Matérn range is on a meaningful unit).
#
# Output: fixtures/spde/meuse_spde.json
# (→ packages/INLASPDE.jl/test/oracle/fixtures/meuse_spde.jld2)

here <- tryCatch(
    normalizePath(dirname(sys.frame(1)$ofile), mustWork = FALSE),
    error = function(e) getwd()
)
if (!nzchar(here)) here <- getwd()
source(file.path(here, "..", "_helpers.R"))

suppressPackageStartupMessages({
    library(INLA)
    library(fmesher)
    library(sp)
})

set.seed(20260424)

data(meuse)
coords <- as.matrix(meuse[, c("x", "y")]) / 1000.0
y <- log(meuse$zinc)
dist_cov <- meuse$dist

mesh <- fmesher::fm_mesh_2d_inla(
    loc       = coords,
    max.edge  = c(0.2, 0.5),
    offset    = c(0.3, 1.0),
    cutoff    = 0.05,
    min.angle = 25
)

spde <- inla.spde2.pcmatern(
    mesh = mesh, alpha = 2,
    prior.range = c(0.5, 0.5),    # P(range < 0.5 km) = 0.5
    prior.sigma = c(1, 0.5)       # P(σ > 1) = 0.5
)

A <- inla.spde.make.A(mesh = mesh, loc = coords)

stk <- inla.stack(
    data = list(y = y),
    A = list(A, 1),
    effects = list(
        field = seq_len(spde$n.spde),
        list(intercept = rep(1, nrow(coords)), dist = dist_cov)
    ),
    tag = "est"
)

form <- y ~ 0 + intercept + dist + f(field, model = spde)

fit <- INLA::inla(
    form, family = "gaussian",
    data = inla.stack.data(stk),
    control.predictor = list(A = inla.stack.A(stk), compute = FALSE),
    control.fixed = list(prec.intercept = 1e-3, prec = 1e-3),
    control.family = list(hyper = list(prec = list(
        prior = "pc.prec", param = c(1, 0.01)
    ))),
    control.compute = list(return.marginals = FALSE)
)

out_path <- file.path(here, "..", "fixtures", "spde", "meuse_spde.json")

# Emit summary frames (no marginals to keep the JSON small — the
# oracle test compares posterior means/SDs and mlik only).
write_inla_fixture(
    fit = fit,
    path = out_path,
    name = "meuse_spde",
    component_names = character(0),   # spatial component summary lives in fit$summary.random, but we compare on hyperpar scale instead
    include_marginals = FALSE,
    meta = list(
        dataset = "sp::meuse",
        n = nrow(coords),
        family = "gaussian",
        formula = "log(zinc) ~ intercept + dist + f(field, model=spde)",
        coordinate_unit = "km",
        prior_range = c(0.5, 0.5),
        prior_sigma = c(1, 0.5),
        prec_prior = "pc.prec(1, 0.01)",
        fixed_prec = 1e-3
    )
)

# Append the input data + mesh + spatial projector so the Julia-side
# oracle test can rebuild the LGM without any R dependency at test
# time. Re-read the JSON, splice in `input` and `mesh`, write back.
fixture <- jsonlite::fromJSON(out_path, simplifyVector = FALSE)
fixture$input <- list(
    y         = as.numeric(y),
    dist      = as.numeric(dist_cov),
    locations = lapply(seq_len(nrow(coords)), function(i) as.numeric(coords[i, ]))
)
fixture$mesh <- mesh_to_list(mesh)
fixture$A_field <- sparse_to_triplet(A)

jsonlite::write_json(
    fixture, out_path,
    auto_unbox = TRUE, digits = 16, pretty = FALSE, na = "null"
)

cat("wrote", out_path, "\n")

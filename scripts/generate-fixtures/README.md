# R-INLA fixture generation

R scripts that produce reference fits from R-INLA, serialized as JLD2
files for Julia oracle tests.

## Layout

```
generate-fixtures/
├── README.md                         # this file
├── renv.lock                         # pinned R + R-INLA version
├── Makefile                          # make all → regenerate everything
├── gmrfs/                            # fixtures for GMRFs.jl
│   ├── qsample_besag_germany.R
│   ├── qinv_rw2.R
│   └── ...
├── lgm/                              # fixtures for LatentGaussianModels.jl
│   ├── scotland_bym2.R
│   ├── penn_bym2.R
│   └── ...
├── spde/                             # fixtures for INLASPDE.jl
│   ├── meuse_spde.R
│   └── ...
└── _helpers.R                        # shared R utilities (jld2 writer, etc.)
```

## Conventions

- Each script is self-contained: sets seed, loads data, fits model, writes
  JLD2 to the corresponding package's `test/oracle/fixtures/`.
- Seeds are hard-coded and documented at the top of each script.
- R package versions are pinned via `renv.lock`; update with explicit
  commits recording the diff.
- Output files are named `<dataset>_<model>.jld2`.

## Writing a fixture

```r
source("_helpers.R")

set.seed(12345)  # reproducibility

library(INLA)
library(SpatialEpi)
data(scotland)

# ... fit model ...
fit <- inla(formula, family = "poisson", data = scotland$data, E = E)

# Write Julia-readable fixture
write_inla_fixture(
    fit,
    path = "../../packages/LatentGaussianModels.jl/test/oracle/fixtures/scotland_bym2.jld2",
    name = "scotland_bym2"
)
```

## When R-INLA updates

1. Update `renv.lock`.
2. Run `make all`.
3. `git diff` the binary fixtures (you'll see size changes; use
   `scripts/diff_fixtures.jl` to see numerical diffs).
4. If changes are explained by R-INLA release notes, commit the new
   fixtures along with the lock update.
5. If unexplained, investigate before committing.

## Running without R

Fixtures are committed to the repo, so Julia tests do not require R.
Contributors without R skip this directory entirely.

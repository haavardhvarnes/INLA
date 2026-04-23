# Benchmarks

Cross-package validation and performance runs. Unlike per-package tests,
these are:
- Slow (minutes to hours).
- End-to-end (use the full ecosystem).
- Compared against external references (R-INLA, Stan, NIMBLE).
- Run on tagged releases and nightly CI, not every PR.

## Layout

```
benchmarks/
├── README.md                      # this file
├── performance/                   # timing-oriented
│   ├── gmrf_sampling.jl
│   ├── inla_bym2.jl
│   └── spde_meuse.jl
├── triangulation/                 # correctness vs Stan/NIMBLE
│   ├── scotland_bym2_vs_stan.jl
│   └── spain_breastcancer_vs_nimble.jl
└── books/                         # textbook reproduction
    ├── moraga_ch05_penn.jl
    ├── moraga_ch06_scotland.jl
    ├── moraga_ch07_ohio.jl
    └── blangiardo_cameletti_ch06.jl
```

## Running

```
julia --project=. benchmarks/performance/gmrf_sampling.jl
```

Each benchmark is a standalone script. Results are printed to stdout and
optionally written to `benchmarks/results/<date>.json` for tracking.

## External dependencies

Benchmarks in `triangulation/` require R + R-INLA + Stan. They are opt-in
via a `BENCHMARKS_EXTERNAL=1` environment variable. Contributors without
these tools simply skip them.

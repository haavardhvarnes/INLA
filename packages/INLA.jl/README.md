# INLA.jl

Umbrella package for the Julia-native INLA stack. A single
`using INLA` brings the inference stack into scope:

- [`GMRFs`](../GMRFs.jl/) — sparse Gaussian Markov random fields.
- [`LatentGaussianModels`](../LatentGaussianModels.jl/) — LGM
  abstraction, latent components, likelihoods, INLA inference.
- [`INLASPDE`](../INLASPDE.jl/) — SPDE–Matérn finite-element
  components on triangulated meshes.

This package contains no numerical code of its own; it `@reexport`s
the three core packages above.

[`INLASPDERasters`](../INLASPDERasters.jl/) is intentionally **not**
bundled because its `Rasters` dependency pulls heavy geospatial
binaries (GDAL_jll, Proj_jll, NetCDF_jll). Users who need raster
glue should add it separately.

## Installation

Not yet on the General registry. Registered in a personal Julia
registry — add it once, then `Pkg.add` as usual:

```julia
using Pkg
Pkg.Registry.add(RegistrySpec(url = "https://github.com/haavardhvarnes/JuliaRegistry"))
Pkg.add("INLA")
```

## Quick example

```julia
using INLA  # GMRFs + LatentGaussianModels + INLASPDE in one import

c_int = Intercept()
c_iid = IID(n; hyperprior = PCPrecision(1.0, 0.01))
A     = sparse([ones(n) Matrix{Float64}(I, n, n)])
ℓ     = PoissonLikelihood()
model = LatentGaussianModel(ℓ, (c_int, c_iid), A)
res   = inla(model, y)
```

For end-to-end vignettes (Scotland BYM2, Tokyo rainfall, Meuse SPDE)
see the ecosystem [README](../../README.md) and the
[Documenter site](../../docs/).

## See also

- Ecosystem [README](../../README.md) and [CHANGELOG](../../CHANGELOG.md).
- [`INLASPDERasters.jl`](../INLASPDERasters.jl/) — raster ↔ SPDE glue
  (separate install).
- Optional sub-packages — [`LGMTuring.jl`](../LGMTuring.jl/),
  [`LGMFormula.jl`](../LGMFormula.jl/),
  [`GMRFsPardiso.jl`](../GMRFsPardiso.jl/) — not yet registered;
  `Pkg.develop` from this repo.

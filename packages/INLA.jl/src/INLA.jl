"""
    INLA

Umbrella package for the Julia-native INLA stack. Re-exports the public API
of [`GMRFs`](@ref), [`LatentGaussianModels`](@ref), and [`INLASPDE`](@ref) so
that a single `using INLA` brings the inference stack into scope.

`INLA` does not contain numerical code of its own. For raster IO bridges,
add `INLASPDERasters` separately — it is intentionally not bundled because
its `Rasters` dependency pulls in heavy geospatial binaries.

# Example

```julia
using INLA  # GMRFs + LatentGaussianModels + INLASPDE in one import
```
"""
module INLA

using Reexport

@reexport using GMRFs
@reexport using LatentGaussianModels
@reexport using INLASPDE

end # module INLA

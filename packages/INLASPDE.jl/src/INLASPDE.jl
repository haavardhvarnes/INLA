"""
    INLASPDE

SPDE–Matérn machinery for [`LatentGaussianModels`](@ref). Provides
FEM-assembled sparse precision matrices `Q(τ, κ)` on triangulated
meshes and exposes `SPDE2 <: AbstractLatentComponent` for use in
`LatentGaussianModel` stacks.

See `plans/plan.md` for the package roadmap and
`/plans/decisions.md` ADR-001, ADR-007, ADR-013 for load-bearing
design decisions.

### Module layout (populated incrementally per milestone)

- `assembly/` — per-element FEM matrices and `Q(α, τ, κ)` (M1).
- `components/` — `SPDE2 <: AbstractLatentComponent` (M2).
- `priors/` — PC-Matérn prior on `(range, σ)` (M2).
- `mesh/` — `inla_mesh_2d` and refinement helpers (M3).
- `projector.jl` — `MeshProjector` for observation-to-vertex maps (M4).
"""
module INLASPDE

using LinearAlgebra
using SparseArrays

using LatentGaussianModels
using LatentGaussianModels: AbstractLatentComponent, AbstractHyperPrior

using Meshes: Meshes
using DelaunayTriangulation: DelaunayTriangulation
using CoordRefSystems: CoordRefSystems
using SciMLOperators: SciMLOperators

# --- milestone includes (uncomment as each milestone lands) -----------
# M1 — FEM assembly
include("assembly/fem.jl")
include("assembly/lumping.jl")
include("assembly/precision.jl")

export assemble_fem_matrices, lumped_mass, stiffness_squared
export FEMMatrices, spde_precision

# M2 — SPDE2 component + PC-Matérn prior
# include("priors/pc_matern.jl")
# include("components/spde2.jl")
#
# M3 — Mesh generation
# include("mesh/boundary.jl")
# include("mesh/refinement.jl")
# include("mesh/inla_mesh.jl")
#
# M4 — Projector
# include("projector.jl")

end # module

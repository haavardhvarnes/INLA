"""
    SPDE2{α}(points, triangles; pc = PCMatern())

Integer-α SPDE–Matérn component on a 2D triangular mesh, implementing
the `AbstractLatentComponent` contract from `LatentGaussianModels`.

The latent field has length `n_vertices(mesh)` and follows a GMRF with
sparse precision `Q(τ, κ)` assembled from the FEM matrices `(C, G₁,
C̃, G₂)` (milestone M1). Internal hyperparameters are
`θ = [log τ, log κ]` per ADR-013; the user-facing scale is the Matérn
`(ρ, σ)` pair with mapping

    ρ = √(8ν) / κ,        σ² = 1 / (4π · ν · κ^(2ν) · τ²)         (d = 2)

where `ν = α - d/2`. In v0.1 only `α = 2` (⇒ `ν = 1`) is supported
alongside the PC-Matern prior; `α = 1` is deferred (ν = 0 in 2D is
degenerate for the PC-Matern density).

# Fields

- `fem::FEMMatrices` — the M1 FEM matrices `(C, G₁, C̃, G₂)`, assembled
  once at construction.
- `graph::GMRFGraph` — mesh adjacency, derived from the off-diagonal
  pattern of `C`. Used for diagnostics, visualisation and compatibility
  with the LGM component machinery.
- `pc::PCMatern` — joint PC prior on `(ρ, σ)`.

# Example

```julia
points, triangles = ...                  # some 2D triangular mesh
spde = SPDE2(points, triangles;
             pc = PCMatern(range_U=0.5, range_α=0.05,
                           sigma_U=1.0, sigma_α=0.01))
Q = precision_matrix(spde, [0.0, 0.0])   # (log τ, log κ) = (0, 0)
```
"""
struct SPDE2{α, T, FE <: FEMMatrices{T}, G <: GMRFs.AbstractGMRFGraph, PR <: PCMatern} <:
       AbstractLatentComponent
    fem::FE
    graph::G
    pc::PR
end

function SPDE2(
        points::AbstractMatrix{<:Real},
        triangles::AbstractMatrix{<:Integer};
        α::Integer=2,
        pc::PCMatern=PCMatern()
)
    α == 2 ||
        throw(ArgumentError("SPDE2: only α = 2 is supported in v0.1; got α=$α. " *
                            "α = 1 (ν = 0 in 2D) is invalid for the PC-Matern prior."))
    fem = FEMMatrices(points, triangles)
    graph = _mesh_graph_from_C(fem.C)
    T = eltype(fem.C)
    return SPDE2{Int(α), T, typeof(fem), typeof(graph), typeof(pc)}(fem, graph, pc)
end

"""
    _mesh_graph_from_C(C) -> GMRFGraph

Build the mesh adjacency graph from the sparsity pattern of the FEM
mass matrix. Two vertices are adjacent iff they share a triangle —
equivalently, iff the corresponding off-diagonal entry of `C` is
nonzero.
"""
function _mesh_graph_from_C(C::AbstractSparseMatrix)
    n = size(C, 1)
    rvs = rowvals(C)
    vals = nonzeros(C)
    Is = Int[]
    Js = Int[]
    for j in 1:n, k in nzrange(C, j)
        i = rvs[k]
        if i != j && !iszero(vals[k])
            push!(Is, i)
            push!(Js, j)
        end
    end
    W = sparse(Is, Js, trues(length(Is)), n, n)
    return GMRFs.GMRFGraph(W)
end

# --- AbstractLatentComponent contract ---------------------------------

Base.length(c::SPDE2) = size(c.fem.C, 1)
LatentGaussianModels.nhyperparameters(::SPDE2) = 2
LatentGaussianModels.initial_hyperparameters(::SPDE2) = [0.0, 0.0]   # (log τ, log κ) = (0, 0)

"""
    precision_matrix(c::SPDE2{α}, θ) -> SparseMatrixCSC

Sparse SPDE precision at `θ = [log τ, log κ]`, delegating to the M1
assembly `spde_precision(fem, α, τ, κ)`.
"""
function LatentGaussianModels.precision_matrix(c::SPDE2{α}, θ) where {α}
    τ = exp(θ[1])
    κ = exp(θ[2])
    return spde_precision(c.fem, α, τ, κ)
end

"""
    log_hyperprior(c::SPDE2, θ) -> Real

PC-Matern log-prior density evaluated at `θ = [log τ, log κ]` on the
internal scale. Maps `(log τ, log κ) → (log ρ, log σ)` and calls
`pc_matern_log_density`. The (log τ, log κ) ↔ (log ρ, log σ) Jacobian
has absolute determinant 1 and contributes no extra term (ADR-013).
"""
function LatentGaussianModels.log_hyperprior(c::SPDE2{2}, θ)
    log_τ, log_κ = θ[1], θ[2]
    # α = 2, d = 2 ⇒ ν = 1
    #   log ρ = 0.5 · log(8) - log κ
    #   log σ = -0.5 · log(4π) - log κ - log τ
    log_ρ = 0.5 * log(8.0) - log_κ
    log_σ = -0.5 * log(4π) - log_κ - log_τ
    return pc_matern_log_density(c.pc, log_ρ, log_σ)
end

"""
    GMRFs.constraints(::SPDE2) -> NoConstraint

The SPDE precision is strictly positive-definite for all `(τ, κ) > 0`
(κ² C + G₁ and κ⁴ C̃ + 2κ² G₁ + G₂ are SPD), so no hard linear
constraint is required.
"""
GMRFs.constraints(::SPDE2) = GMRFs.NoConstraint()

"""
    log_normalizing_constant(c::SPDE2, θ) -> Real

R-INLA-style log normalizing constant for the proper Gaussian SPDE
prior, evaluated at internal `θ = [log τ, log κ]`:

    log NC = -½ d log(2π) + ½ log|Q(θ)|

where `d = length(c)` is the number of mesh vertices and `Q(θ)` is the
SPDE precision (κ⁴ C̃ + 2κ² G₁ + G₂ scaled by τ², for α = 2). Required
by the marginal-likelihood formula in `LatentGaussianModels.jl`'s
Laplace inference (commit 635cbb9 / ADR-style change).
"""
function LatentGaussianModels.log_normalizing_constant(c::SPDE2, θ)
    Q = LatentGaussianModels.precision_matrix(c, θ)
    F = cholesky(Symmetric(Q))
    return -0.5 * length(c) * log(2π) + 0.5 * logdet(F)
end

"""
    spde_user_scale(c::SPDE2{α}, θ) -> (ρ, σ)

Convert the internal `θ = [log τ, log κ]` to the user-facing Matérn
pair `(ρ, σ)` using the fixed mapping for `α = 2` in 2D.
"""
function spde_user_scale(::SPDE2{2}, θ)
    log_τ, log_κ = θ[1], θ[2]
    log_ρ = 0.5 * log(8.0) - log_κ
    log_σ = -0.5 * log(4π) - log_κ - log_τ
    return exp(log_ρ), exp(log_σ)
end

"""
    spde_internal_scale(c::SPDE2{α}, ρ, σ) -> (log τ, log κ)

Inverse of [`spde_user_scale`](@ref). Returns the internal
`(log τ, log κ)` pair corresponding to user-scale Matérn `(ρ, σ)`.
"""
function spde_internal_scale(::SPDE2{2}, ρ::Real, σ::Real)
    ρ > 0 || throw(ArgumentError("ρ must be positive; got $ρ"))
    σ > 0 || throw(ArgumentError("σ must be positive; got $σ"))
    log_κ = 0.5 * log(8.0) - log(ρ)
    log_τ = -0.5 * log(4π) - log_κ - log(σ)
    return log_τ, log_κ
end

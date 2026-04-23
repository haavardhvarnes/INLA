"""
    FEMMatrices{T, SC, SG}

Container for the precomputed FEM matrices `(C, G₁, C̃, G₂)` on a fixed
triangular mesh. These matrices depend only on mesh geometry, not on the
SPDE hyperparameters `(τ, κ)`, so they are assembled once and reused for
every precision evaluation.

# Fields
- `C::SC`        — full P1 mass matrix.
- `G1::SG`       — P1 stiffness matrix (`∫ ∇φ_i · ∇φ_j`).
- `C_lumped::SC` — diagonal mass matrix from `lumped_mass(C)`.
- `G2::SG`       — `G₁ · C̃⁻¹ · G₁`, used for α = 2.

Construct via `FEMMatrices(points, triangles)`.
"""
struct FEMMatrices{T, SC <: AbstractSparseMatrix{T}, SG <: AbstractSparseMatrix{T}}
    C::SC
    G1::SG
    C_lumped::SC
    G2::SG
end

"""
    FEMMatrices(points, triangles)

Assemble `C`, `G₁`, `C̃`, and `G₂` from a 2D triangular mesh given as raw
arrays. See [`assemble_fem_matrices`](@ref) for the argument conventions.
"""
function FEMMatrices(
        points::AbstractMatrix{<:Real},
        triangles::AbstractMatrix{<:Integer},
    )
    C, G1 = assemble_fem_matrices(points, triangles)
    C_lumped = lumped_mass(C)
    G2 = stiffness_squared(G1, C_lumped)
    return FEMMatrices(C, G1, C_lumped, G2)
end

"""
    stiffness_squared(G1, C_lumped) -> G2

Construct `G₂ = G₁ · C̃⁻¹ · G₁`, where `C̃` is a lumped (diagonal) mass
matrix. This is the sparse approximation of `G₁ · C⁻¹ · G₁` used for α = 2
SPDE precision.

Throws `ArgumentError` if any diagonal entry of `C_lumped` is zero — this
indicates a vertex with zero associated area, i.e. a degenerate mesh.
"""
function stiffness_squared(G1::AbstractSparseMatrix, C_lumped::AbstractSparseMatrix)
    d = diag(C_lumped)
    any(iszero, d) &&
        throw(ArgumentError("C_lumped has a zero diagonal entry; mesh is degenerate"))
    D_inv = Diagonal(inv.(d))
    return G1 * D_inv * G1
end

"""
    spde_precision(fem::FEMMatrices, α, τ, κ) -> Q

Assemble the SPDE-Matérn precision matrix on user-scale parameters
`(τ, κ)`. Supported orders are `α ∈ {1, 2}`.

- α = 1: `Q = τ² · (κ² C + G₁)` (Matérn smoothness ν = 0).
- α = 2: `Q = τ² · (κ⁴ C̃ + 2κ² G₁ + G₂)` (ν = 1), using the lumped mass
  matrix `C̃` — this matches R-INLA's implementation per
  Lindgren-Rue-Lindström (2011, Appendix C).

Fractional α is deferred to v0.3 (Bolin–Kirchner 2020 rational
approximation).
"""
function spde_precision(fem::FEMMatrices, α::Integer, τ::Real, κ::Real)
    τ > 0 ||
        throw(ArgumentError("τ must be positive; got τ=$τ"))
    κ > 0 ||
        throw(ArgumentError("κ must be positive; got κ=$κ"))
    if α == 1
        return τ^2 * (κ^2 * fem.C + fem.G1)
    elseif α == 2
        return τ^2 * (κ^4 * fem.C_lumped + 2 * κ^2 * fem.G1 + fem.G2)
    end
    throw(ArgumentError("α must be 1 or 2; got α=$α. Fractional α deferred to v0.3."))
end

"""
    spde_precision(α, τ, κ, C, G1[, C_lumped, G2]) -> Q

Stateless form: assemble `Q(α, τ, κ)` directly from the raw FEM matrices.
Missing `C_lumped` and `G2` are derived on the fly. Prefer
[`spde_precision(::FEMMatrices, ...)`](@ref) in hot loops — the
`FEMMatrices` constructor precomputes `C̃` and `G₂` once.
"""
function spde_precision(
        α::Integer, τ::Real, κ::Real,
        C::AbstractSparseMatrix, G1::AbstractSparseMatrix,
        C_lumped::Union{Nothing, AbstractSparseMatrix} = nothing,
        G2::Union{Nothing, AbstractSparseMatrix} = nothing,
    )
    if α == 1
        τ > 0 && κ > 0 ||
            throw(ArgumentError("τ and κ must be positive; got τ=$τ, κ=$κ"))
        return τ^2 * (κ^2 * C + G1)
    end
    Cl = C_lumped === nothing ? lumped_mass(C) : C_lumped
    G2_ = G2 === nothing ? stiffness_squared(G1, Cl) : G2
    fem = FEMMatrices(C, G1, Cl, G2_)
    return spde_precision(fem, α, τ, κ)
end

"""
    AbstractObservationMapping

Abstract type for the linear map from the stacked latent vector `x`
into the linear predictor `η = mapping * x`. The seam introduced by
ADR-017 to support joint-likelihood / multi-response models without
inheriting `inla.stack`'s ergonomic baggage.

# Concrete subtypes

- [`IdentityMapping`](@ref)    — `η = x`. Areal models with `A = I`.
- [`LinearProjector`](@ref)    — wraps an `A::AbstractMatrix`. The
  v0.1 default; constructed automatically when `LatentGaussianModel`
  is called with an `AbstractMatrix`.
- [`StackedMapping`](@ref)     — block-row stack of per-likelihood
  mappings. Lands fully in Phase G PR2 (multi-likelihood); the struct
  ships now so the type hierarchy is complete.
- [`KroneckerMapping`](@ref)   — separable `A_space ⊗ A_time`. Stub
  for Phase M; struct ships now, `apply!` errors until Phase M lands.

# Required interface

Every concrete subtype implements:

```julia
apply!(η, mapping, x)            # η .= mapping * x, in place
apply_adjoint!(g, mapping, r)    # g .= mappingᵀ * r, in place
nrows(mapping) -> Int            # row count of the implicit matrix
ncols(mapping) -> Int            # column count = length(x)
```

Optional (default returns `1` for single-block mappings):

```julia
likelihood_for(mapping, i) -> Int   # which likelihood-block owns row i
```

# Convenience

Defined here for any subtype satisfying the required interface:

- `Base.:*(mapping, x)` — allocating forward multiply.
- `Base.size(mapping)`, `Base.size(mapping, dim)`.
- [`as_matrix`](@ref) — concrete materialization for the ~3 inference
  sites that need a sparse/dense `A` (multi-RHS solves; `A' D A`
  Hessian quadratic form).

# References

- ADR-017 in `plans/decisions.md`.
"""
abstract type AbstractObservationMapping end

# ---------------------------------------------------------------------
# Required-interface stubs (so ambiguous calls error informatively).
# ---------------------------------------------------------------------

"""
    apply!(η, mapping, x) -> η

In-place forward multiply: `η .= mapping * x`. Concrete subtypes must
specialise.
"""
function apply!(::AbstractVector, m::AbstractObservationMapping, ::AbstractVector)
    throw(MethodError(apply!, (m,)))
end

"""
    apply_adjoint!(g, mapping, r) -> g

In-place adjoint multiply: `g .= mappingᵀ * r`. Concrete subtypes must
specialise.
"""
function apply_adjoint!(::AbstractVector, m::AbstractObservationMapping, ::AbstractVector)
    throw(MethodError(apply_adjoint!, (m,)))
end

"""
    nrows(mapping) -> Int

Row count of the implicit matrix; equal to `length(η)`.
"""
function nrows end

"""
    ncols(mapping) -> Int

Column count of the implicit matrix; equal to `length(x)`.
"""
function ncols end

"""
    likelihood_for(mapping, i) -> Int

Block index of the likelihood applied to observation row `i`. Defaults
to `1` for any single-block mapping; multi-block mappings (e.g.
`StackedMapping`) override.
"""
likelihood_for(::AbstractObservationMapping, ::Integer) = 1

"""
    as_matrix(mapping) -> AbstractMatrix

Concrete materialization of the mapping as an `AbstractMatrix`. Used
by inference sites that need to multi-RHS solve through the projector
or build the `Aᵀ D A` Hessian quadratic form. Sparse where the
underlying mapping is sparse.
"""
function as_matrix end

# ---------------------------------------------------------------------
# IdentityMapping
# ---------------------------------------------------------------------

"""
    IdentityMapping(n)

`η = x`; algebraically `A = I_n`. Used for areal models where the
observation index equals the latent index. The dimension `n` is
carried so shape assertions at construction are cheap.
"""
struct IdentityMapping <: AbstractObservationMapping
    n::Int
end

nrows(m::IdentityMapping) = m.n
ncols(m::IdentityMapping) = m.n

function apply!(η::AbstractVector, m::IdentityMapping, x::AbstractVector)
    length(η) == m.n ||
        throw(DimensionMismatch("η has length $(length(η)); IdentityMapping expects $(m.n)"))
    length(x) == m.n ||
        throw(DimensionMismatch("x has length $(length(x)); IdentityMapping expects $(m.n)"))
    copyto!(η, x)
    return η
end

function apply_adjoint!(g::AbstractVector, m::IdentityMapping, r::AbstractVector)
    length(g) == m.n ||
        throw(DimensionMismatch("g has length $(length(g)); IdentityMapping expects $(m.n)"))
    length(r) == m.n ||
        throw(DimensionMismatch("r has length $(length(r)); IdentityMapping expects $(m.n)"))
    copyto!(g, r)
    return g
end

as_matrix(m::IdentityMapping) = sparse(1.0 * I, m.n, m.n)

# ---------------------------------------------------------------------
# LinearProjector
# ---------------------------------------------------------------------

"""
    LinearProjector(A::AbstractMatrix)

Wraps an explicit row-to-latent matrix `A`. The v0.1 default — every
existing `LatentGaussianModel(...)` call with a matrix `A` becomes
`LinearProjector(A)` internally via the compatibility constructor.

Forward / adjoint multiplies dispatch directly onto `LinearAlgebra.mul!`
on the wrapped matrix; no additional allocation.
"""
struct LinearProjector{A <: AbstractMatrix} <: AbstractObservationMapping
    A::A
end

nrows(m::LinearProjector) = size(m.A, 1)
ncols(m::LinearProjector) = size(m.A, 2)

function apply!(η::AbstractVector, m::LinearProjector, x::AbstractVector)
    mul!(η, m.A, x)
    return η
end

function apply_adjoint!(g::AbstractVector, m::LinearProjector, r::AbstractVector)
    mul!(g, m.A', r)
    return g
end

as_matrix(m::LinearProjector) = m.A

# ---------------------------------------------------------------------
# StackedMapping  (struct ships now; full use lands in Phase G PR2)
# ---------------------------------------------------------------------

"""
    StackedMapping(blocks::Tuple, rows::Vector{UnitRange{Int}})

Block-row stack of per-likelihood mappings. Each block shares the
*same* latent vector `x` (shared `ncols`); each block contributes a
contiguous slice of observation rows defined by `rows[k]`.

# Invariants

- `length(blocks) == length(rows)`.
- All blocks have the same `ncols`.
- `rows[k]` are disjoint, contiguous, and sorted; together they cover
  `1:nrows(mapping)`.

The contiguous-and-sorted invariant is documented as the v0.2 design
point in ADR-017's "Open questions" — interleaved row indices are
deferred unless a real user needs them.
"""
struct StackedMapping{T <: Tuple} <: AbstractObservationMapping
    blocks::T
    rows::Vector{UnitRange{Int}}

    function StackedMapping{T}(blocks::T, rows::Vector{UnitRange{Int}}) where {T}
        length(blocks) == length(rows) ||
            throw(ArgumentError("blocks ($(length(blocks))) and rows " *
                                "($(length(rows))) must have equal length"))
        # Shared ncols across blocks.
        n_x = ncols(first(blocks))
        for (k, b) in enumerate(blocks)
            ncols(b) == n_x ||
                throw(ArgumentError("block $k has ncols=$(ncols(b)); " *
                                    "expected $n_x (all blocks share the latent)"))
        end
        # Contiguous sorted rows summing to nrows.
        cursor = 0
        for (k, rng) in enumerate(rows)
            first(rng) == cursor + 1 ||
                throw(ArgumentError("rows[$k] = $rng is not contiguous; " *
                                    "expected to start at $(cursor + 1)"))
            length(rng) == nrows(blocks[k]) ||
                throw(ArgumentError("rows[$k] has length $(length(rng)); " *
                                    "block has $(nrows(blocks[k])) rows"))
            cursor = last(rng)
        end
        return new{T}(blocks, rows)
    end
end

StackedMapping(blocks::Tuple, rows::Vector{UnitRange{Int}}) =
    StackedMapping{typeof(blocks)}(blocks, rows)

function nrows(m::StackedMapping)
    isempty(m.rows) && return 0
    return last(last(m.rows))
end

ncols(m::StackedMapping) = ncols(first(m.blocks))

function apply!(η::AbstractVector, m::StackedMapping, x::AbstractVector)
    length(η) == nrows(m) ||
        throw(DimensionMismatch("η has length $(length(η)); StackedMapping expects $(nrows(m))"))
    for (block, rng) in zip(m.blocks, m.rows)
        @views apply!(η[rng], block, x)
    end
    return η
end

function apply_adjoint!(g::AbstractVector, m::StackedMapping, r::AbstractVector)
    length(r) == nrows(m) ||
        throw(DimensionMismatch("r has length $(length(r)); StackedMapping expects $(nrows(m))"))
    fill!(g, zero(eltype(g)))
    tmp = similar(g)
    for (block, rng) in zip(m.blocks, m.rows)
        @views apply_adjoint!(tmp, block, r[rng])
        g .+= tmp
    end
    return g
end

function likelihood_for(m::StackedMapping, i::Integer)
    for (k, rng) in enumerate(m.rows)
        i in rng && return k
    end
    throw(BoundsError(m, i))
end

as_matrix(m::StackedMapping) = vcat(map(as_matrix, m.blocks)...)

# ---------------------------------------------------------------------
# KroneckerMapping  (struct ships now; implementation in Phase M)
# ---------------------------------------------------------------------

"""
    KroneckerMapping(A_space, A_time)

Separable space-time projector `A = A_space ⊗ A_time`. The struct
exists so the type hierarchy is closed; `apply!` / `apply_adjoint!`
implementations land in Phase M (non-stationary SPDE expansion).
"""
struct KroneckerMapping{S <: AbstractObservationMapping,
                        T <: AbstractObservationMapping} <: AbstractObservationMapping
    A_space::S
    A_time::T
end

nrows(m::KroneckerMapping) = nrows(m.A_space) * nrows(m.A_time)
ncols(m::KroneckerMapping) = ncols(m.A_space) * ncols(m.A_time)

function apply!(::AbstractVector, ::KroneckerMapping, ::AbstractVector)
    error("KroneckerMapping.apply! is a Phase M task (replan-2026-04-28.md " *
          "Phase M — SPDE expansion); the struct ships in Phase G for " *
          "type-hierarchy completeness only.")
end

function apply_adjoint!(::AbstractVector, ::KroneckerMapping, ::AbstractVector)
    error("KroneckerMapping.apply_adjoint! is a Phase M task; see apply! for context.")
end

as_matrix(m::KroneckerMapping) = kron(as_matrix(m.A_space), as_matrix(m.A_time))

# ---------------------------------------------------------------------
# Base method conveniences (work for any subtype with the required interface)
# ---------------------------------------------------------------------

Base.size(m::AbstractObservationMapping) = (nrows(m), ncols(m))
function Base.size(m::AbstractObservationMapping, d::Integer)
    if d == 1
        return nrows(m)
    elseif d == 2
        return ncols(m)
    elseif d > 2
        return 1
    else
        throw(ArgumentError("dimension out of range: $d"))
    end
end

function Base.:*(m::AbstractObservationMapping, x::AbstractVector)
    η = similar(x, nrows(m))
    return apply!(η, m, x)
end

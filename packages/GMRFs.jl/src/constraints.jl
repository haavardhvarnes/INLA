"""
    AbstractConstraint

Base type for linear constraints on a GMRF. Currently only
`LinearConstraint` (the hard constraint form `A x = e`) and
`NoConstraint` are implemented.

Sampling and inference on constrained GMRFs uses the standard
conditioning-by-kriging correction (Rue & Held 2005, §2.3.3); that
lives in the sampling module, not here — this file provides only the
data types.
"""
abstract type AbstractConstraint end

"""
    NoConstraint <: AbstractConstraint

Singleton for "no constraint applied". Returned by default from
`constraints(::AbstractGMRF)` for proper GMRFs.
"""
struct NoConstraint <: AbstractConstraint end

nconstraints(::NoConstraint) = 0
constraint_matrix(::NoConstraint) = zeros(Float64, 0, 0)
constraint_rhs(::NoConstraint) = Float64[]

"""
    LinearConstraint{T,M<:AbstractMatrix{T},V<:AbstractVector{T}} <: AbstractConstraint

The hard linear constraint `A x = e`, with `A` of shape `k × n` and
`e` of length `k`. For sum-to-zero constraints on an intrinsic Besag
with `r` connected components, `A` has `r` rows — one indicator row
per component.
"""
struct LinearConstraint{T <: Real, M <: AbstractMatrix{T}, V <: AbstractVector{T}} <: AbstractConstraint
    A::M
    e::V

    function LinearConstraint(A::M, e::V) where {T, M <: AbstractMatrix{T}, V <: AbstractVector{T}}
        size(A, 1) == length(e) ||
            throw(DimensionMismatch("constraint rows ($(size(A, 1))) must match rhs length ($(length(e)))"))
        return new{T, M, V}(A, e)
    end
end

"""
    LinearConstraint(A::AbstractMatrix)

Homogeneous constraint `A x = 0`. RHS defaults to `zeros(size(A,1))` in
the element type of `A`.
"""
LinearConstraint(A::AbstractMatrix{T}) where {T} = LinearConstraint(A, zeros(T, size(A, 1)))

nconstraints(c::LinearConstraint) = size(c.A, 1)
constraint_matrix(c::LinearConstraint) = c.A
constraint_rhs(c::LinearConstraint) = c.e

"""
    constraints(g::AbstractGMRF) -> AbstractConstraint

The default constraint attached to a GMRF. Proper GMRFs return
`NoConstraint()`; intrinsic ones (RW1, RW2, Besag) return a
`LinearConstraint` with one sum-to-zero row per connected component
(Freni-Sterrantino et al. 2018).
"""
constraints(::AbstractGMRF) = NoConstraint()

# Default constraints for intrinsic models.

constraints(::IIDGMRF) = NoConstraint()
constraints(::AR1GMRF) = NoConstraint()

# RW1 open or cyclic: single sum-to-zero row (the null space is the
# constant vector).
function constraints(g::RW1GMRF)
    n = num_nodes(g)
    A = reshape(ones(eltype(g), n), 1, n)
    return LinearConstraint(A, zeros(eltype(g), 1))
end

# RW2: open has 2D null space spanned by {1, 1:n}; cyclic has 1D null
# space {1}.
function constraints(g::RW2GMRF)
    n = num_nodes(g)
    T = eltype(g)
    if g.cyclic
        A = reshape(ones(T, n), 1, n)
        return LinearConstraint(A, zeros(T, 1))
    else
        A = zeros(T, 2, n)
        @views A[1, :] .= one(T)
        @views A[2, :] .= T.(1:n)
        return LinearConstraint(A, zeros(T, 2))
    end
end

# Besag: one sum-to-zero row per connected component.
function constraints(g::BesagGMRF)
    return sum_to_zero_constraints(g.g; T = eltype(g))
end

# Seasonal: s-1 constraints spanning the null space (period-s sequences
# summing to zero within one period). Row k (k = 1..s-1) is the pattern
# ε_k(i) = δ((i-1) mod s == k-1) - δ((i-1) mod s == s-1) repeated with
# period s; so range(C^T) = null(Q) exactly, as required by the Laplace
# contract.
function constraints(g::SeasonalGMRF)
    n = num_nodes(g)
    s = g.period
    T = eltype(g)
    A = zeros(T, s - 1, n)
    for i in 1:n
        r = mod1(i, s)       # 1..s
        if r < s
            A[r, i] = one(T)
        else
            A[:, i] .= -one(T)
        end
    end
    return LinearConstraint(A, zeros(T, s - 1))
end

"""
    sum_to_zero_constraints(g::AbstractGMRFGraph; T = Float64) -> LinearConstraint

Build the per-component sum-to-zero constraint for a graph with
`r = nconnected_components(g)` components. The resulting `A` has `r`
rows, one for each component, with `1`s in that component's nodes and
`0`s elsewhere. RHS is `zeros(r)`.

This is the Freni-Sterrantino et al. (2018) correction for intrinsic
CAR on disconnected graphs.
"""
function sum_to_zero_constraints(g::AbstractGMRFGraph; T::Type{<:Real} = Float64)
    labels = connected_component_labels(g)
    n = num_nodes(g)
    r = maximum(labels)
    A = zeros(T, r, n)
    for i in 1:n
        A[labels[i], i] = one(T)
    end
    return LinearConstraint(A, zeros(T, r))
end

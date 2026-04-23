# Sampling from a GMRF.
#
# For a proper GMRF with precision Q = LLᵀ (Cholesky),
#   x = μ + L⁻ᵀ z,   z ∼ N(0, I)
# yields x ∼ N(μ, Q⁻¹).
#
# For an intrinsic GMRF (rankdef r ≥ 1) with null-space basis V
# (orthonormal), we draw from the projected distribution N(0, Q⁺) on
# the non-null subspace {x : V'x = 0}:
#   Qp = Q + V Vᵀ   (rank-r bump; strictly positive-definite)
#   x₀ = L⁻ᵀ z
#   x  = x₀ - V V' x₀
# The projection kills the random null-space component; the bump does
# not affect the non-null distribution.

"""
    Base.rand(rng::AbstractRNG, g::AbstractGMRF) -> Vector{Float64}

Draw one sample `x ∼ N(μ, Q⁺)` from the GMRF `g`. For intrinsic
models the sample satisfies the canonical per-component sum-to-zero
constraint (Freni-Sterrantino et al. 2018). Allocates.
"""
function Base.rand(rng::Random.AbstractRNG, g::AbstractGMRF)
    x = Vector{Float64}(undef, num_nodes(g))
    return rand!(rng, x, g)
end

Base.rand(g::AbstractGMRF) = rand(Random.default_rng(), g)

"""
    Random.rand!(rng::AbstractRNG, x::AbstractVector, g::AbstractGMRF)

In-place draw. `x` must have length `num_nodes(g)`. Returns `x`.
"""
function Random.rand!(rng::Random.AbstractRNG, x::AbstractVector, g::AbstractGMRF)
    n = num_nodes(g)
    length(x) == n ||
        throw(DimensionMismatch("rand!: destination length $(length(x)) ≠ num_nodes $n"))
    Q = precision_matrix(g)
    r = rankdef(g)
    if r == 0
        F = cholesky(Symmetric(Q))
        z = randn(rng, n)
        # Solve L' y = z  (y = L'^{-1} z has covariance Q^{-1})
        # cholesky(..).U is the upper factor; (U'U = Q since Symmetric uses
        # upper triangle by default). Solve U' y = z then apply permutation inverse.
        # In Julia's SparseSuite cholesky, F \ z returns Q^{-1} z.
        # For a one-sided triangular solve, use UpperTriangular(F.U) \ z after permutation.
        y = F.UP \ z
        x .= y .+ prior_mean(g)
    else
        V = null_space_basis(g)
        # Rank-r bump on the null space; V is orthonormal so the bump equals VVᵀ.
        Qp = Q + SparseMatrixCSC(V * V')
        F = cholesky(Symmetric(Qp))
        z = randn(rng, n)
        x₀ = F.UP \ z
        # Remove null-space component: x = x₀ - V V' x₀
        x .= x₀ .- V * (V' * x₀) .+ prior_mean(g)
    end
    return x
end

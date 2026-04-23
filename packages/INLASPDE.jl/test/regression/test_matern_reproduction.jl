# SPDE–Matérn covariance reproduction on a fine regular mesh.
#
# Theory: for the α = 2 SPDE in 2D (ν = 1), the precision `Q(τ, κ)`
# reproduces the Matérn covariance
#     C(r; κ, σ) = σ² · (κr) · K_1(κr),             (r > 0)
# with marginal variance σ² = 1 / (4π κ² τ²) and range ρ = √8 / κ.
# (Lindgren, Rue and Lindström, 2011, §3.2.)
#
# This test builds a fine regular triangulation, computes a few columns
# of Q⁻¹ via sparse Cholesky, and checks them against the closed-form
# Matérn covariance at the mesh vertices. Tolerances are generous — the
# finite-element approximation and the finite domain both introduce bias.

using LinearAlgebra
using SparseArrays
using SpecialFunctions: besselk

"""
Build a regular triangulation of `[-L/2, L/2]²` with `n` cells per side
(each split into two right triangles along the SW–NE diagonal).

Returns `(points, triangles, indexer)` where `indexer(i, j)` gives the
1-based global vertex index for the `(i, j)` lattice point.
"""
function _regular_triangulation(L::Real, n::Integer)
    h = L / n
    nvert_side = n + 1
    points = Matrix{Float64}(undef, nvert_side^2, 2)
    indexer(i, j) = (j - 1) * nvert_side + i     # 1-based on both axes
    for j in 1:nvert_side, i in 1:nvert_side
        points[indexer(i, j), 1] = -L / 2 + (i - 1) * h
        points[indexer(i, j), 2] = -L / 2 + (j - 1) * h
    end
    tris = Int[]
    for j in 1:n, i in 1:n
        a = indexer(i, j)
        b = indexer(i + 1, j)
        c = indexer(i + 1, j + 1)
        d = indexer(i, j + 1)
        push!(tris, a, b, c)
        push!(tris, a, c, d)
    end
    triangles = permutedims(reshape(tris, 3, :))
    return points, triangles, indexer
end

"2D Matérn covariance with ν = 1 (α = 2 in 2D): C(r) = σ² κr K_1(κr)."
function _matern_nu1(r::Real, κ::Real, σ²::Real)
    r == 0 && return σ²
    κr = κ * r
    return σ² * κr * besselk(1, κr)
end

@testset "Matérn reproduction — α = 2, ν = 1 on a fine regular grid" begin
    L = 10.0
    n = 40                    # 41 × 41 = 1681 vertices, 3200 triangles
    κ = 1.0
    σ² = 1.0
    τ = inv(sqrt(4π * κ^2 * σ²))

    points, triangles, indexer = _regular_triangulation(L, n)
    fem = FEMMatrices(points, triangles)
    Q = spde_precision(fem, 2, τ, κ)

    # Sparse Cholesky — uses CHOLMOD via SparseArrays.
    F = cholesky(Symmetric(Q))

    nvert_side = n + 1
    c_i = cld(nvert_side, 2)
    c_j = cld(nvert_side, 2)
    c = indexer(c_i, c_j)                     # central vertex
    e_c = zeros(size(Q, 1)); e_c[c] = 1.0
    q_inv_col = F \ e_c                        # Q⁻¹[:, c]

    # --- marginal variance at the centre ---
    var_center = q_inv_col[c]
    @test var_center ≈ σ² rtol = 0.05           # ~5% — FE + boundary error

    # --- covariance at a handful of radii ---
    # Sample along the +x axis where exact distances are available.
    for di in (1, 2, 4, 6)
        j = indexer(c_i + di, c_j)
        r = di * (L / n)
        theoretical = _matern_nu1(r, κ, σ²)
        empirical = q_inv_col[j]
        @test isapprox(empirical, theoretical; atol = 0.05 * σ², rtol = 0.1)
    end

    # --- monotone decay along the axis ---
    radial_values = [q_inv_col[indexer(c_i + di, c_j)] for di in 0:8]
    @test all(diff(radial_values) .< 0)

    # --- isotropy check: variance at vertices equidistant from centre
    # should be close. ---
    v_east = q_inv_col[indexer(c_i + 4, c_j)]
    v_north = q_inv_col[indexer(c_i, c_j + 4)]
    v_west = q_inv_col[indexer(c_i - 4, c_j)]
    v_south = q_inv_col[indexer(c_i, c_j - 4)]
    mean_axial = (v_east + v_north + v_west + v_south) / 4
    @test maximum(abs, (v_east, v_north, v_west, v_south) .- mean_axial) <
        0.02 * σ²
end

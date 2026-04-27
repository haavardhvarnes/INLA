# Null-space bases for intrinsic GMRFs.
#
# An intrinsic GMRF has precision `Q` with a nontrivial null space; any
# vector in that null space can be added to `x` without changing the
# density. Sampling, log-density, and scaling all need an explicit
# orthonormal basis `V ∈ ℝ^{n×r}` (V'V = I_r) for the null space.

"""
    null_space_basis(g::AbstractGMRF) -> Matrix{Float64}

Return an orthonormal basis `V` of the null space of
`precision_matrix(g)`, with shape `(n, rankdef(g))`. `V'V = I`. For
proper GMRFs the result is `zeros(n, 0)`.

This is a `Float64` matrix by default. If a different element type is
needed, convert at the call site.
"""
function null_space_basis end

null_space_basis(g::IIDGMRF) = zeros(Float64, num_nodes(g), 0)
null_space_basis(g::AR1GMRF) = zeros(Float64, num_nodes(g), 0)

# RW1 — null space is span{1}, orthonormal = 1/√n
function null_space_basis(g::RW1GMRF)
    n = num_nodes(g)
    V = fill(1.0 / sqrt(n), n, 1)
    return V
end

# RW2 — cyclic has null space span{1}; open has span{1, 1:n}
function null_space_basis(g::RW2GMRF)
    n = num_nodes(g)
    if g.cyclic
        return fill(1.0 / sqrt(n), n, 1)
    else
        # Gram-Schmidt on (1_n, (1:n))
        V = zeros(Float64, n, 2)
        V[:, 1] .= 1.0 / sqrt(n)
        v2 = Float64.(collect(1:n))
        v2 .-= dot(V[:, 1], v2) .* V[:, 1]
        v2 ./= sqrt(sum(abs2, v2))
        V[:, 2] .= v2
        return V
    end
end

# Besag — per-component indicators, L2-normalised
function null_space_basis(g::BesagGMRF)
    return _component_indicator_basis(g.g)
end

function _component_indicator_basis(gr::AbstractGMRFGraph)
    labels = connected_component_labels(gr)
    n = num_nodes(gr)
    r = maximum(labels)
    V = zeros(Float64, n, r)
    for i in 1:n
        V[i, labels[i]] = 1.0
    end
    # column-normalise
    for k in 1:r
        nk = sqrt(sum(abs2, view(V, :, k)))
        V[:, k] ./= nk
    end
    return V
end

# Seasonal — period-s zero-sum patterns, basis ε_k = e_k − e_s repeated
# with period s for k = 1..s−1, then QR-orthonormalised.
function null_space_basis(g::SeasonalGMRF)
    n = g.n
    s = g.period
    V = zeros(Float64, n, s - 1)
    for k in 1:(s - 1), i in 1:n
        r = mod1(i, s)
        if r == k
            V[i, k] = 1.0
        elseif r == s
            V[i, k] = -1.0
        end
    end
    F = qr(V)
    return Matrix(F.Q)[:, 1:(s - 1)]
end

# Generic0 — fall back to dense eigendecomposition on R
function null_space_basis(g::Generic0GMRF)
    r = rankdef(g)
    if r == 0
        return zeros(Float64, num_nodes(g), 0)
    end
    F = eigen(Symmetric(Matrix(g.R)))
    idx = sortperm(F.values)
    V = F.vectors[:, idx[1:r]]
    # Sign convention: make the largest-magnitude entry in each column
    # positive, so basis is deterministic across runs.
    for k in 1:r
        i_max = argmax(abs.(view(V, :, k)))
        if V[i_max, k] < 0
            V[:, k] .= -V[:, k]
        end
    end
    return Matrix{Float64}(V)
end

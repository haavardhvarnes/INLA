"""
    AbstractGMRF

A Gaussian Markov random field on a graph. Concrete subtypes are
distributions of vectors `x ∈ ℝⁿ` with precision matrix `Q` and mean
`μ` (possibly zero).

Required methods:

- `num_nodes(g)` — dimension of the field.
- `precision_matrix(g)` — the full symmetric precision matrix `Q` as a
  `SparseMatrixCSC{<:Real,Int}`.
- `prior_mean(g)` — the mean vector `μ`. Defaults to `zeros(num_nodes(g))`.
- `rankdef(g)` — rank deficiency of `Q` (size of the null space). `0`
  for proper GMRFs, `≥ 1` for intrinsic ones.

Intrinsic GMRFs (e.g. Besag, RW1, RW2) additionally carry constraint
metadata; see `constraints(::AbstractGMRF)`.
"""
abstract type AbstractGMRF end

Base.length(g::AbstractGMRF) = num_nodes(g)
Base.eltype(g::AbstractGMRF) = eltype(precision_matrix(g))

"""
    precision_matrix(g::AbstractGMRF)

Full symmetric precision matrix `Q` as a `SparseMatrixCSC{<:Real,Int}`.
Must include both upper and lower triangles so that slicing and linear
solves work without additional wrapping.
"""
function precision_matrix end

"""
    prior_mean(g::AbstractGMRF)

Prior mean vector. Defaults to zero for all currently implemented
GMRFs; carried as a function so that future mean-shifted types can
override.
"""
prior_mean(g::AbstractGMRF) = zeros(eltype(g), num_nodes(g))

"""
    rankdef(g::AbstractGMRF) -> Int

Rank deficiency of `Q`. `0` for proper GMRFs (IID, AR1), positive for
intrinsic ones (`1` for connected RW1/Besag, `2` for RW2, equal to the
number of connected components for disconnected Besag).
"""
rankdef(g::AbstractGMRF) = 0

num_nodes(g::AbstractGMRF) = size(precision_matrix(g), 1)

# ============================================================
# IID
# ============================================================

"""
    IIDGMRF{T}(n; τ = one(T))

`x ∼ N(0, τ⁻¹ I_n)`. The precision matrix is `τ · I`.
"""
struct IIDGMRF{T <: Real} <: AbstractGMRF
    n::Int
    τ::T
end
IIDGMRF(n::Integer; τ::Real = 1.0) = IIDGMRF{typeof(float(τ))}(Int(n), float(τ))

num_nodes(g::IIDGMRF) = g.n

function precision_matrix(g::IIDGMRF{T}) where {T}
    return spdiagm(0 => fill(g.τ, g.n))
end

# ============================================================
# Random-walks (RW1, RW2) — cyclic or open
# ============================================================

"""
    RW1GMRF{T}(n; τ = one(T), cyclic = false)

First-order random walk precision on `n` nodes. Open form has rank
deficiency 1 (constant shift), cyclic form has rank deficiency 1 as
well (the cyclic sum-to-zero null space).
"""
struct RW1GMRF{T <: Real} <: AbstractGMRF
    n::Int
    τ::T
    cyclic::Bool
end
function RW1GMRF(n::Integer; τ::Real = 1.0, cyclic::Bool = false)
    n ≥ 2 || throw(ArgumentError("RW1GMRF requires n ≥ 2, got $n"))
    return RW1GMRF{typeof(float(τ))}(Int(n), float(τ), cyclic)
end

num_nodes(g::RW1GMRF) = g.n
rankdef(g::RW1GMRF) = 1

function precision_matrix(g::RW1GMRF{T}) where {T}
    n = g.n
    τ = g.τ
    # structure matrix R of RW1: tridiagonal with 2 on diag (1 at
    # endpoints, open case) and -1 off-diagonal
    diag_vals = fill(T(2), n)
    if !g.cyclic
        diag_vals[1] = one(T)
        diag_vals[end] = one(T)
    end
    Is = Int[]; Js = Int[]; Vs = T[]
    for i in 1:n
        push!(Is, i); push!(Js, i); push!(Vs, τ * diag_vals[i])
    end
    for i in 1:(n - 1)
        push!(Is, i); push!(Js, i + 1); push!(Vs, -τ)
        push!(Is, i + 1); push!(Js, i); push!(Vs, -τ)
    end
    if g.cyclic
        push!(Is, 1); push!(Js, n); push!(Vs, -τ)
        push!(Is, n); push!(Js, 1); push!(Vs, -τ)
    end
    return sparse(Is, Js, Vs, n, n)
end

"""
    RW2GMRF{T}(n; τ = one(T), cyclic = false)

Second-order random walk precision on `n` nodes. `R = D' D` where `D`
is the second-difference operator. Rank deficiency 2 (linear null
space) in the open case, 1 in the cyclic case.
"""
struct RW2GMRF{T <: Real} <: AbstractGMRF
    n::Int
    τ::T
    cyclic::Bool
end
function RW2GMRF(n::Integer; τ::Real = 1.0, cyclic::Bool = false)
    n ≥ 3 || throw(ArgumentError("RW2GMRF requires n ≥ 3, got $n"))
    return RW2GMRF{typeof(float(τ))}(Int(n), float(τ), cyclic)
end

num_nodes(g::RW2GMRF) = g.n
rankdef(g::RW2GMRF) = g.cyclic ? 1 : 2

function precision_matrix(g::RW2GMRF{T}) where {T}
    n = g.n
    τ = g.τ
    # Build D (second differences), then R = D' D.
    if g.cyclic
        # cyclic: all n rows
        Id = Int[]; Jd = Int[]; Vd = T[]
        for k in 1:n
            im1 = mod1(k - 1, n)
            ip1 = mod1(k + 1, n)
            push!(Id, k); push!(Jd, im1); push!(Vd, one(T))
            push!(Id, k); push!(Jd, k);   push!(Vd, -T(2))
            push!(Id, k); push!(Jd, ip1); push!(Vd, one(T))
        end
        D = sparse(Id, Jd, Vd, n, n)
    else
        # open: n-2 rows for second differences i=2..n-1 on a length-n signal
        Id = Int[]; Jd = Int[]; Vd = T[]
        for (row, k) in enumerate(2:(n - 1))
            push!(Id, row); push!(Jd, k - 1); push!(Vd, one(T))
            push!(Id, row); push!(Jd, k);     push!(Vd, -T(2))
            push!(Id, row); push!(Jd, k + 1); push!(Vd, one(T))
        end
        D = sparse(Id, Jd, Vd, n - 2, n)
    end
    R = D' * D
    return τ .* SparseMatrixCSC{T, Int}(R)
end

# ============================================================
# AR1 — stationary first-order autoregressive (proper)
# ============================================================

"""
    AR1GMRF{T}(n; ρ, τ)

Stationary AR(1) with correlation `ρ ∈ (-1, 1)` and marginal precision
`τ > 0`. The joint precision is tridiagonal, full-rank; this is a
*proper* GMRF with `Var(x_i) = 1/τ` for every `i`.

    Q = (τ / (1 - ρ²)) · S
    S_11 = S_nn = 1
    S_ii = 1 + ρ²           for interior i
    S_{i,i+1} = -ρ

This matches Rue & Held (2005, Eq. 1.39) and R-INLA's `ar1` convention,
where `τ` is the marginal precision of the field.

Note: R-INLA parameterizes via `log((1+ρ)/(1−ρ))` internally. This
constructor takes `ρ` directly; the logit-Fisher mapping is applied at
the `LatentGaussianModels.jl` component layer.
"""
struct AR1GMRF{T <: Real} <: AbstractGMRF
    n::Int
    ρ::T
    τ::T
end
function AR1GMRF(n::Integer; ρ::Real, τ::Real = 1.0)
    n ≥ 2 || throw(ArgumentError("AR1GMRF requires n ≥ 2, got $n"))
    -1 < ρ < 1 || throw(ArgumentError("AR1GMRF requires ρ ∈ (-1, 1), got ρ=$ρ"))
    τ > 0 || throw(ArgumentError("AR1GMRF requires τ > 0, got τ=$τ"))
    T = typeof(float(ρ * τ))
    return AR1GMRF{T}(Int(n), T(ρ), T(τ))
end

num_nodes(g::AR1GMRF) = g.n
rankdef(::AR1GMRF) = 0

function precision_matrix(g::AR1GMRF{T}) where {T}
    n = g.n
    ρ = g.ρ
    τ = g.τ
    # Rue & Held (2005) Eq. 1.39: AR(1) with Var(x_i) = 1/τ.
    scale = τ / (one(T) - ρ^2)
    Is = Int[]; Js = Int[]; Vs = T[]
    for i in 1:n
        if i == 1 || i == n
            push!(Is, i); push!(Js, i); push!(Vs, scale * one(T))
        else
            push!(Is, i); push!(Js, i); push!(Vs, scale * (one(T) + ρ^2))
        end
    end
    for i in 1:(n - 1)
        push!(Is, i); push!(Js, i + 1); push!(Vs, -scale * ρ)
        push!(Is, i + 1); push!(Js, i); push!(Vs, -scale * ρ)
    end
    return sparse(Is, Js, Vs, n, n)
end

# ============================================================
# Seasonal — Rue & Held (2005, §3.4.3); R-INLA `model = "seasonal"`
# ============================================================

"""
    SeasonalGMRF{T}(n; period, τ = 1.0)

Intrinsic seasonal-variation model of length `n` and period `s =
period`. Precision

    Q = τ · B' B,     B ∈ ℝ^{(n-s+1) × n},
    B_{t,j} = 1 if j ∈ [t, t+s-1] else 0,

so that each `s`-consecutive sum of the field is penalised:

    π(x | τ) ∝ exp(-½ τ Σ_{t=1}^{n-s+1} (x_t + ⋯ + x_{t+s-1})²).

The null space has dimension `s - 1`: it is spanned by all `s`-periodic
sequences summing to zero within one period. Equivalently, a basis is
`ε_k = e_k - e_s` (repeated with period `s`) for `k = 1, …, s-1`, where
`e_k` is the `k`-th canonical basis vector of ℝ^s; so `rankdef = s-1`.

Matches Rue & Held (2005, §3.4.3) and R-INLA's `model = "seasonal"`.
"""
struct SeasonalGMRF{T <: Real} <: AbstractGMRF
    n::Int
    period::Int
    τ::T
end
function SeasonalGMRF(n::Integer; period::Integer, τ::Real = 1.0)
    period ≥ 2 ||
        throw(ArgumentError("SeasonalGMRF requires period ≥ 2, got period=$period"))
    n > period ||
        throw(ArgumentError("SeasonalGMRF requires n > period, got n=$n, period=$period"))
    return SeasonalGMRF{typeof(float(τ))}(Int(n), Int(period), float(τ))
end

num_nodes(g::SeasonalGMRF) = g.n
rankdef(g::SeasonalGMRF) = g.period - 1

function precision_matrix(g::SeasonalGMRF{T}) where {T}
    n = g.n
    s = g.period
    nr = n - s + 1
    Ib = Int[]; Jb = Int[]; Vb = T[]
    for t in 1:nr, j in t:(t + s - 1)
        push!(Ib, t); push!(Jb, j); push!(Vb, one(T))
    end
    B = sparse(Ib, Jb, Vb, nr, n)
    R = B' * B
    return g.τ .* SparseMatrixCSC{T, Int}(R)
end

# ============================================================
# Besag — intrinsic conditional autoregressive on a graph
# ============================================================

"""
    BesagGMRF(g::AbstractGMRFGraph; τ = 1.0, scale_model = true)

The intrinsic CAR / Besag GMRF on graph `g`. Precision
`Q = τ · (c · L)` where `L` is the graph Laplacian and `c` is the
Sørbye-Rue scaling constant if `scale_model = true` (default, matching
R-INLA ≥ 17.06). If `scale_model = false`, `c = 1`.

`rankdef` equals the number of connected components of `g`.
"""
struct BesagGMRF{T <: Real, G <: AbstractGMRFGraph} <: AbstractGMRF
    g::G
    τ::T
    c::T                 # scaling constant (c=1 when scale_model=false)
    scale_model::Bool
    # Laplacian cached in the graph; the τ-independent structure τ·L·c
    # is not cached here because τ may vary freely.
end

function BesagGMRF(g::AbstractGMRFGraph; τ::Real = 1.0, scale_model::Bool = true)
    c = scale_model ? scale_factor(g) : 1.0
    T = typeof(float(τ * c))
    return BesagGMRF{T, typeof(g)}(g, T(τ), T(c), scale_model)
end

"""
    BesagGMRF(W::AbstractMatrix; τ = 1.0, scale_model = true)

Convenience constructor from an adjacency matrix.
"""
BesagGMRF(W::AbstractMatrix; kwargs...) = BesagGMRF(GMRFGraph(W); kwargs...)

num_nodes(g::BesagGMRF) = num_nodes(g.g)
rankdef(g::BesagGMRF) = nconnected_components(g.g)
is_scaled(g::BesagGMRF) = g.scale_model

function precision_matrix(g::BesagGMRF{T}) where {T}
    L = laplacian_matrix(g.g)
    Q = (g.τ * g.c) .* SparseMatrixCSC{T, Int}(L)
    return Q
end

"""
    scale_factor(g::AbstractGMRFGraph) -> Float64

The Sørbye & Rue (2014) geometric-mean scaling constant for the
intrinsic Besag GMRF on `g`. Applied so that under `Q_scaled = c · L`,
the geometric mean of marginal variances of `τ⁻¹ Q_scaled⁻¹` on the
non-null subspace equals `τ⁻¹`.

The reference implementation is dense. For production use on large
graphs, the LGM Phase 3 selected-inversion implementation should be
used instead (see `plans/defaults-parity.md` and ADR-004).
"""
function scale_factor(g::AbstractGMRFGraph)
    L = laplacian_matrix(g)
    n = size(L, 1)
    # Per-component null-space basis V ∈ ℝ^{n×r}: each component's
    # indicator, L2-normalised.
    labels = connected_component_labels(g)
    r = maximum(labels)
    V = zeros(Float64, n, r)
    for i in 1:n
        V[i, labels[i]] = 1.0
    end
    # Normalise columns to unit L2 (so V'V = I_r).
    for k in 1:r
        nk = sqrt(sum(abs2, view(V, :, k)))
        V[:, k] ./= nk
    end
    # Dense generalised inverse on non-null subspace:
    #   Q_perp = L + V V'
    #   Σ = inv(Q_perp) - V V'
    Ld = Matrix{Float64}(L)
    Qperp = Ld + V * V'
    Σ = inv(Qperp) - V * V'
    # Scaling constant is geometric mean of positive diagonals.
    d = diag(Σ)
    # Guard against numerical zeros / slightly negative values at
    # machine precision — clamp to a small positive floor.
    pos = filter(x -> x > 0, d)
    isempty(pos) && throw(ArgumentError("could not compute scale factor: no positive diagonal entries of generalised inverse"))
    return exp(mean(log.(pos)))
end

"""
    scale_model(g::BesagGMRF)

Return a scaled copy. If `g.scale_model == true` this returns `g`
unchanged; otherwise it returns a new `BesagGMRF` with
`scale_model = true`. Pure function.
"""
function scale_model(g::BesagGMRF)
    g.scale_model && return g
    return BesagGMRF(g.g; τ = g.τ, scale_model = true)
end

# ============================================================
# Generic0: user-supplied structure matrix R, Q = τ R
# ============================================================

"""
    Generic0GMRF(R; τ = 1.0, rankdef = 0, scale_model = false)

User-supplied structure matrix `R` (symmetric, non-negative definite),
precision `Q = τ · R`. `rankdef` must be supplied since in general we
cannot infer it cheaply from `R`. If `scale_model = true`, the
Sørbye-Rue scaling is applied to `R` before multiplying by `τ`.

R-INLA's `generic1` is a special case of this with `R` rescaled so
its largest eigenvalue is 1; that transformation lives at the LGM
component layer.
"""
struct Generic0GMRF{T <: Real} <: AbstractGMRF
    R::SparseMatrixCSC{T, Int}   # structure matrix (after scaling, if any)
    τ::T
    rd::Int
    c::T                         # scaling constant actually applied
    scaled::Bool
end

function Generic0GMRF(R::AbstractMatrix; τ::Real = 1.0,
                      rankdef::Integer = 0, scale_model::Bool = false)
    n, m = size(R)
    n == m || throw(DimensionMismatch("structure matrix must be square, got $n×$m"))
    issymmetric(R) || throw(ArgumentError("structure matrix must be symmetric"))
    rankdef ≥ 0 || throw(ArgumentError("rankdef must be ≥ 0, got $rankdef"))
    T = typeof(float(τ))
    Rs = SparseMatrixCSC{T, Int}(R)
    c = one(T)
    if scale_model
        c = T(_generic_scale_factor(Rs, rankdef))
        Rs = c .* Rs
    end
    return Generic0GMRF{T}(Rs, T(τ), Int(rankdef), c, scale_model)
end

num_nodes(g::Generic0GMRF) = size(g.R, 1)
rankdef(g::Generic0GMRF) = g.rd
is_scaled(g::Generic0GMRF) = g.scaled

function precision_matrix(g::Generic0GMRF{T}) where {T}
    return g.τ .* g.R
end

"""
Generic dense reference scale-factor for an arbitrary structure matrix
R with known rank deficiency `rd`. For intrinsic Besag we have the
combinatorial-Laplacian null space explicitly (constant-per-component);
here we fall back to an eigendecomposition on the dense form.
"""
function _generic_scale_factor(R::SparseMatrixCSC, rd::Integer)
    n = size(R, 1)
    # Sort eigenvalues ascending; drop the `rd` smallest (assumed to be
    # the null space; we check they are numerically ~0).
    λ = eigvals(Symmetric(Matrix(R)))
    λ_sorted = sort(λ)
    null_λ = λ_sorted[1:rd]
    nz_λ = λ_sorted[(rd + 1):end]
    tol = sqrt(eps()) * (isempty(nz_λ) ? one(eltype(nz_λ)) : maximum(abs, nz_λ))
    if !isempty(null_λ) && maximum(abs, null_λ) > tol
        throw(ArgumentError("claimed rankdef=$rd but the smallest $rd eigenvalues are not numerically zero (largest: $(maximum(abs, null_λ)))"))
    end
    # Reconstruct V (eigenvectors of the null space) for Σ correction.
    F = eigen(Symmetric(Matrix(R)))
    idx = sortperm(F.values)
    V = F.vectors[:, idx[1:rd]]
    # Σ = (R + V V')⁻¹ - V V'
    Σ = inv(Matrix(R) + V * V') - V * V'
    d = diag(Σ)
    pos = filter(x -> x > 0, d)
    isempty(pos) && throw(ArgumentError("could not compute scale factor: no positive diagonals of generalised inverse"))
    return exp(mean(log.(pos)))
end

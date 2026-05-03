"""
    AbstractIIDND <: AbstractLatentComponent

Umbrella type for the multivariate-IID family. Two concrete subtypes
(see ADR-022):

- `IIDND_Sep{N}` — separable parameterisation: `N` marginal precisions
  + `N(N-1)/2` correlations, each carrying its own scalar
  `AbstractHyperPrior`. This matches R-INLA's
  `f(., model = "2diid"/"iid3d", ...)` defaults.
- `IIDND_Joint{N}` — joint Λ parameterisation with a single
  `AbstractJointHyperPrior` (Wishart / InvWishart). Lands in PR-1c.

The latent vector has length `n · N` and stores the `N` blocks of length
`n` consecutively: `x = (x^{(1)}, x^{(2)}, …, x^{(N)})` where each
`x^{(j)} ∈ ℝ^n`. Joint precision is `Q = Λ ⊗ I_n` for both subtypes.
"""
abstract type AbstractIIDND <: AbstractLatentComponent end

"""
    IIDND_Sep{N, PP, PC} <: AbstractIIDND

Separable parameterisation of an `N`-variate IID component. `PP` is the
type of `precpriors` (an `NTuple{N, AbstractHyperPrior}`); `PC` is the
type of `corrpriors` (an `NTuple{N(N-1)/2, AbstractHyperPrior}`).

Hyperparameter vector layout (length `N + N(N-1)/2`):

    θ = (log τ_1, …, log τ_N,  atanh ρ_{1,2}, atanh ρ_{1,3}, …, atanh ρ_{N-1,N})

with the correlations enumerated in row-major order over the strictly
lower-triangular indices (i.e. `(i, j)` with `i < j`).

PR-1a ships only the `N = 2` case; `N ≥ 3` lands in PR-1b.
"""
struct IIDND_Sep{N, PP <: Tuple, PC <: Tuple} <: AbstractIIDND
    n::Int
    precpriors::PP
    corrpriors::PC
end

"""
    IIDND(n, N = 2; hyperprior_precs, hyperprior_corr, hyperprior_corrs)

Multivariate IID latent component of dimension `n × N`. See ADR-022 for
the parameterisation choice; in PR-1a only `N = 2` is implemented.

For `N = 2`, pass `hyperprior_corr` (a single `AbstractHyperPrior` on
`atanh ρ`). For `N ≥ 3`, pass `hyperprior_corrs` as an `NTuple` of
length `N(N-1)/2`.

Defaults: each marginal precision uses `PCPrecision()`; each correlation
uses `PCCor0()` (R-INLA's `pc.cor0(0.5, 0.5)` default).
"""
function IIDND(n::Integer, N::Integer=2;
        hyperprior_precs::Union{Nothing, NTuple}=nothing,
        hyperprior_corr::Union{Nothing, AbstractHyperPrior}=nothing,
        hyperprior_corrs::Union{Nothing, NTuple}=nothing)
    n > 0 || throw(ArgumentError("IIDND: n must be positive"))
    N ≥ 2 || throw(ArgumentError("IIDND: N must be ≥ 2"))
    N == 2 || throw(ArgumentError("IIDND: N ≥ 3 lands in PR-1b; PR-1a supports N = 2 only"))

    if hyperprior_corr !== nothing && hyperprior_corrs !== nothing
        throw(ArgumentError("IIDND: pass exactly one of `hyperprior_corr` (N=2) or `hyperprior_corrs` (N≥3)"))
    end

    pps = hyperprior_precs === nothing ?
        ntuple(_ -> PCPrecision(), N) :
        hyperprior_precs
    length(pps) == N ||
        throw(ArgumentError("IIDND: hyperprior_precs must have length $N, got $(length(pps))"))
    all(p -> p isa AbstractHyperPrior, pps) ||
        throw(ArgumentError("IIDND: every entry of hyperprior_precs must be an AbstractHyperPrior"))

    K = N * (N - 1) ÷ 2
    cps = if hyperprior_corr !== nothing
        N == 2 ||
            throw(ArgumentError("IIDND: `hyperprior_corr` is for N = 2; use `hyperprior_corrs` for N ≥ 3"))
        (hyperprior_corr,)
    elseif hyperprior_corrs !== nothing
        length(hyperprior_corrs) == K ||
            throw(ArgumentError("IIDND: hyperprior_corrs must have length $K, got $(length(hyperprior_corrs))"))
        all(p -> p isa AbstractHyperPrior, hyperprior_corrs) ||
            throw(ArgumentError("IIDND: every entry of hyperprior_corrs must be an AbstractHyperPrior"))
        hyperprior_corrs
    else
        ntuple(_ -> PCCor0(), K)
    end

    return IIDND_Sep{N, typeof(pps), typeof(cps)}(Int(n), pps, cps)
end

"""
    IID2D(n; hyperprior_precs = (PCPrecision(), PCPrecision()),
              hyperprior_corr  = PCCor0())

Bivariate IID component — ergonomic alias for `IIDND(n, 2; …)`. Mirrors
R-INLA's `f(idx, model = "2diid", …)` defaults.
"""
function IID2D(n::Integer;
        hyperprior_precs::NTuple{2, AbstractHyperPrior}=(PCPrecision(), PCPrecision()),
        hyperprior_corr::AbstractHyperPrior=PCCor0())
    return IIDND(n, 2;
        hyperprior_precs=hyperprior_precs,
        hyperprior_corr=hyperprior_corr)
end

Base.length(c::IIDND_Sep{N}) where {N} = c.n * N

nhyperparameters(::IIDND_Sep{N}) where {N} = N + N * (N - 1) ÷ 2

initial_hyperparameters(c::IIDND_Sep{N}) where {N} =
    zeros(Float64, nhyperparameters(c))

GMRFs.constraints(::IIDND_Sep) = NoConstraint()

# ---------------------------------------------------------------------
# N = 2 specialisations (PR-1a). N ≥ 3 lifted in PR-1b.
# ---------------------------------------------------------------------

# Internal: bivariate precision-matrix entries `(Λ_11, Λ_12, Λ_22)` from
# `(τ_1, τ_2, t)` with `t = atanh(ρ)`. The closed form
#
#     Λ = (1/(1 - ρ²)) · [τ_1            -ρ √(τ_1 τ_2);
#                          -ρ √(τ_1 τ_2)   τ_2         ]
#
# is rewritten via `1 - ρ² = sech²(t)` and `ρ/(1 - ρ²) = sinh(t) cosh(t)`
# to stay finite where `1 - tanh(t)²` underflows to `0` in float64
# (|t| ≳ 19) — `precision_matrix(IIDND_Sep{2}, θ)` is called inside the
# outer-θ LBFGS line search, which can probe |θ[3]| ≫ 19 before the
# objective is evaluated to discover the region is bad.
#
# Returned as a 3-tuple so `precision_matrix` can build the sparse
# `Λ ⊗ I_n` directly without allocating a 2×2 matrix.
function _iid2d_lambda(τ1, τ2, t)
    cosh_t = cosh(t)
    sinh_t = sinh(t)
    cosh²_t = cosh_t * cosh_t
    Λ11 = τ1 * cosh²_t
    Λ22 = τ2 * cosh²_t
    Λ12 = -sqrt(τ1 * τ2) * sinh_t * cosh_t
    return Λ11, Λ12, Λ22
end

function precision_matrix(c::IIDND_Sep{2}, θ)
    n = c.n
    τ1 = exp(θ[1])
    τ2 = exp(θ[2])
    Λ11, Λ12, Λ22 = _iid2d_lambda(τ1, τ2, θ[3])
    diag_main = vcat(fill(Λ11, n), fill(Λ22, n))
    off_diag  = fill(Λ12, n)
    return spdiagm(0 => diag_main, n => off_diag, -n => off_diag)
end

function log_hyperprior(c::IIDND_Sep{2}, θ)
    return log_prior_density(c.precpriors[1], θ[1]) +
           log_prior_density(c.precpriors[2], θ[2]) +
           log_prior_density(c.corrpriors[1], θ[3])
end

# Proper, full-rank Q = Λ ⊗ I_n with `det(Q) = det(Λ)^n` and
# `det(Λ) = τ_1 τ_2 / (1 - ρ²)`. Total dimension is `2n`. The
# `-log(1 - ρ²)` term is computed as `2·logcosh(θ[3])` to remain finite
# at saturation; see `_iid2d_lambda` and `_logcosh` for context.
#
#     log NC = -½ · 2n · log(2π) + ½ · n · (θ_1 + θ_2 + 2·logcosh(θ_3)).
function log_normalizing_constant(c::IIDND_Sep{2}, θ)
    n = c.n
    return -n * log(2π) + 0.5 * n * (θ[1] + θ[2] + 2 * _logcosh(θ[3]))
end

function gmrf(c::IIDND_Sep{2}, θ)
    Q = precision_matrix(c, θ)
    return GMRFs.Generic0GMRF(Q; τ=1.0, rankdef=0, scale_model=false)
end

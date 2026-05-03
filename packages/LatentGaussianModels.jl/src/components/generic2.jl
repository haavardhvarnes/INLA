"""
    Generic2(C; rankdef = 0, scale_model = false,
            hyperprior_τv = PCPrecision(), hyperprior_τu = PCPrecision(),
            constraint = nothing)

R-INLA's `model = "generic2"` — hierarchical Gaussian model on a joint
latent vector `[u; v]` of length `2n` (where `n = size(C, 1)`):

    v ~ N(0, (τ_v C)⁻¹),
    u | v ~ N(v, (τ_u I_n)⁻¹).

The joint precision (eq. 1 of the R-INLA `generic2` doc) is the
`2n × 2n` block matrix

    Q = [[ τ_u I,         -τ_u I       ],
         [ -τ_u I,        τ_u I + τ_v C ]]

with two hyperparameters on the internal scale matching R-INLA's
parameterisation (`theta1`, `theta2`):

    θ[1] = log τ_v   (precision multiplying C),
    θ[2] = log τ_u   (precision of the conditional noise u | v).

Used for shared/specific decompositions — the `u` block carries the
linear-predictor-relevant signal, the `v` block carries the structured
prior mediated by `C`. With `C` the ICAR Laplacian this is a
joint-precision-parameterised cousin of BYM (which exposes
`u + v` directly via summation rather than the joint precision).

# Arguments
- `C`: symmetric, non-negative-definite "structure" matrix.
- `rankdef`: dimension of `null(C)`. The rank of `Q` is then
  `2n − rankdef`, and the caller is responsible for supplying a
  matching `LinearConstraint` when `rankdef > 0`.
- `scale_model`: if `true`, applies the Sørbye-Rue (2014)
  geometric-mean scaling to `C` once at construction (matches
  R-INLA's `scale.model = TRUE`).
- `hyperprior_τv`: scalar prior on `θ[1] = log τ_v`. Default
  `PCPrecision()`. R-INLA's default is `loggamma(1, 5e-5)` — pass
  `GammaPrecision(1.0, 5.0e-5)` to match.
- `hyperprior_τu`: scalar prior on `θ[2] = log τ_u`. Default
  `PCPrecision()`. R-INLA's default is `loggamma(1, 1e-3)` (looser
  shape than τ_v) — pass `GammaPrecision(1.0, 1.0e-3)` to match.
- `constraint`: optional `LinearConstraint` over the joint length-`2n`
  latent. For intrinsic `C` (e.g. ICAR Laplacian, `rankdef = 1`) the
  nullspace of `Q` is `span([1_n; 1_n])` — a single sum-to-zero on
  either the `u` or the `v` block alone breaks the ambiguity.
"""
struct Generic2{T <: Real, P1 <: AbstractHyperPrior, P2 <: AbstractHyperPrior,
    Con <: Union{NoConstraint, LinearConstraint}} <: AbstractLatentComponent
    C::SparseMatrixCSC{T, Int}    # structure matrix (after scaling, if any)
    rd::Int
    hyperprior_τv::P1
    hyperprior_τu::P2
    constraint::Con
end

function Generic2(Cmat::AbstractMatrix; rankdef::Integer=0,
        scale_model::Bool=false,
        hyperprior_τv::AbstractHyperPrior=PCPrecision(),
        hyperprior_τu::AbstractHyperPrior=PCPrecision(),
        constraint::Union{Nothing, LinearConstraint}=nothing)
    n, m = size(Cmat)
    n == m || throw(DimensionMismatch("Generic2: C must be square, got $n×$m"))
    issymmetric(Cmat) || throw(ArgumentError("Generic2: C must be symmetric"))
    rankdef ≥ 0 || throw(ArgumentError("Generic2: rankdef must be ≥ 0, got $rankdef"))
    rankdef ≤ n || throw(ArgumentError("Generic2: rankdef must be ≤ n, got $rankdef"))

    # Apply Sørbye-Rue scaling via the public Generic0GMRF API so we
    # store the (possibly scaled) C once. Downstream τ_v multiplies it.
    g = GMRFs.Generic0GMRF(Cmat; τ=1.0, rankdef=rankdef, scale_model=scale_model)
    Cs = SparseMatrixCSC{Float64, Int}(GMRFs.precision_matrix(g))
    con = constraint === nothing ? NoConstraint() : constraint
    return Generic2(Cs, Int(rankdef), hyperprior_τv, hyperprior_τu, con)
end

Base.length(c::Generic2) = 2 * size(c.C, 1)
nhyperparameters(::Generic2) = 2
initial_hyperparameters(::Generic2) = [0.0, 0.0]

function precision_matrix(c::Generic2, θ)
    n = size(c.C, 1)
    τv = exp(θ[1])
    τu = exp(θ[2])
    I_n = sparse(I, n, n)
    A = τu * I_n
    B = -τu * I_n            # B' = B (B is a scalar multiple of identity)
    D = τu * I_n + τv * c.C
    return [A B; B D]
end

function log_hyperprior(c::Generic2, θ)
    return log_prior_density(c.hyperprior_τv, θ[1]) +
           log_prior_density(c.hyperprior_τu, θ[2])
end

GMRFs.constraints(c::Generic2) = c.constraint

# Per-component log NC. The Schur complement on the (1,1) block (full
# rank) gives `|Q|_+ = τ_u^n · τ_v^(n−rd) · |C|_+`, so
#
#   ½ log|Q|_+ = ½ n log τ_u + ½ (n−rd) log τ_v + ½ log|C|_+.
#
# Following the F_GENERIC0 / F_BYM2 convention shared with the rest of
# the package, the user-independent `½ log|C|_+` term is dropped — this
# keeps Julia's `mlik` aligned with R-INLA's "up to a constant" form.
function log_normalizing_constant(c::Generic2, θ)
    n = size(c.C, 1)
    return -0.5 * (2n - c.rd) * log(2π) +
           0.5 * n * θ[2] +
           0.5 * (n - c.rd) * θ[1]
end

function gmrf(c::Generic2, θ)
    Q = precision_matrix(c, θ)
    return GMRFs.Generic0GMRF(Q; τ=1.0, rankdef=c.rd, scale_model=false)
end

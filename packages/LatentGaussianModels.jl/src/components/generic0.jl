"""
    Generic0(R; rankdef = 0, scale_model = false, hyperprior = PCPrecision(),
            constraint = nothing)

User-supplied structure-matrix component, R-INLA's `model = "generic0"`.
The latent vector has length `n = size(R, 1)` and precision `Q = τ · R`,
with one hyperparameter on the internal scale `θ = log(τ)`.

`R` must be symmetric and non-negative definite. `rankdef` is the
dimension of the null space (rank deficiency); supply this when known
because we cannot infer it cheaply for a general `R`. With
`scale_model = true` the Sørbye-Rue (2014) geometric-mean scaling is
applied to `R` once at construction (matching R-INLA's
`scale.model = TRUE`).

`constraint` is an optional `LinearConstraint` attached to the
component (e.g. sum-to-zero for an intrinsic `R`); if `nothing`, no
constraint is applied. For intrinsic components the user is
responsible for providing the appropriate constraint.

The default hyperprior is `PCPrecision()`. R-INLA's `generic0` default
is `loggamma(1, 5e-5)` — pass `hyperprior = GammaPrecision(1.0, 5.0e-5)`
to match.
"""
struct Generic0{T <: Real, P <: AbstractHyperPrior,
    C <: Union{NoConstraint, LinearConstraint}} <: AbstractLatentComponent
    R::SparseMatrixCSC{T, Int}    # structure matrix (after scaling, if any)
    rd::Int
    hyperprior::P
    constraint::C
end

function Generic0(R::AbstractMatrix; rankdef::Integer=0,
        scale_model::Bool=false,
        hyperprior::AbstractHyperPrior=PCPrecision(),
        constraint::Union{Nothing, LinearConstraint}=nothing)
    n, m = size(R)
    n == m || throw(DimensionMismatch("Generic0: R must be square, got $n×$m"))
    issymmetric(R) || throw(ArgumentError("Generic0: R must be symmetric"))
    rankdef ≥ 0 || throw(ArgumentError("Generic0: rankdef must be ≥ 0, got $rankdef"))
    # Apply Sørbye-Rue scaling via the public Generic0GMRF API so we
    # store the (possibly scaled) R once. Use τ = 1 and read back the
    # structure matrix; downstream we multiply by exp(θ[1]).
    g = GMRFs.Generic0GMRF(R; τ=1.0, rankdef=rankdef,
        scale_model=scale_model)
    Rs = SparseMatrixCSC{Float64, Int}(GMRFs.precision_matrix(g))
    con = constraint === nothing ? NoConstraint() : constraint
    return Generic0(Rs, Int(rankdef), hyperprior, con)
end

Base.length(c::Generic0) = size(c.R, 1)
nhyperparameters(::Generic0) = 1
initial_hyperparameters(::Generic0) = [0.0]   # log τ = 0 ⇒ τ = 1

precision_matrix(c::Generic0, θ) = exp(θ[1]) .* c.R

log_hyperprior(c::Generic0, θ) = log_prior_density(c.hyperprior, θ[1])

GMRFs.constraints(c::Generic0) = c.constraint

# Per-component log NC matching R-INLA's `extra()` for `F_GENERIC0`
# (`inla.c:2986-2987`, shared branch with F_IID/F_BESAG/F_RW1/F_RW2/
# F_SEASONAL): `LOG_NORMC_GAUSSIAN · (n - rd) + (n - rd)/2 · log τ`,
# i.e. `-½(n - rd) log(2π) + ½(n - rd) log τ`. The structural
# `½ log|R̃|_+` term is *dropped* — R-INLA reports `mlik` "up to a
# normalisation constant" and omits this θ-independent piece for
# `F_GENERIC0`. We follow the same convention so Julia's mlik matches
# R-INLA's mlik (rather than the exact Gaussian-Gaussian conjugate).
function log_normalizing_constant(c::Generic0, θ)
    n = size(c.R, 1)
    return -0.5 * (n - c.rd) * log(2π) + 0.5 * (n - c.rd) * θ[1]
end

function gmrf(c::Generic0, θ)
    # `c.R` is already (optionally) scaled at construction; pass it
    # through with `scale_model = false` to avoid double scaling.
    return GMRFs.Generic0GMRF(c.R; τ=exp(θ[1]), rankdef=c.rd,
        scale_model=false)
end

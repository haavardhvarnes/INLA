"""
    Generic1(R; rankdef = 0, hyperprior = PCPrecision(), constraint = nothing)

Like [`Generic0`](@ref) but rescales `R` at construction so that its
largest eigenvalue is `1`. Matches the structure-matrix normalisation
used by R-INLA's `generic1` before multiplying by `τ`. Precision is
`Q = τ · R̃` with `R̃ = R / λ_max(R)` and `τ = exp(θ[1])`.

Unlike R-INLA's full `generic1`, this component does **not** introduce
the mixing parameter `β ∈ (0, 1)` — only the eigenvalue rescaling. The
β-mixing flavour is deferred; see
[packages/GMRFs.jl/plans/plan.md M2 note](../../GMRFs.jl/plans/plan.md).

`R` must be symmetric and non-negative definite with a strictly
positive largest eigenvalue. `rankdef` is the dimension of the null
space; if non-zero the caller is responsible for supplying a matching
`constraint` (e.g. sum-to-zero).

The default hyperprior is `PCPrecision()`. R-INLA's `generic1` default
is `loggamma(1, 5e-5)` — pass `hyperprior = GammaPrecision(1.0, 5.0e-5)`
to match.
"""
struct Generic1{T <: Real, P <: AbstractHyperPrior,
                C <: Union{NoConstraint, LinearConstraint}} <: AbstractLatentComponent
    R::SparseMatrixCSC{T, Int}    # rescaled so λ_max(R) = 1
    rd::Int
    hyperprior::P
    constraint::C
    λ_max_original::T             # λ_max(R) before rescaling
end

function Generic1(R::AbstractMatrix; rankdef::Integer = 0,
                  hyperprior::AbstractHyperPrior = PCPrecision(),
                  constraint::Union{Nothing, LinearConstraint} = nothing)
    n, m = size(R)
    n == m || throw(DimensionMismatch("Generic1: R must be square, got $n×$m"))
    issymmetric(R) || throw(ArgumentError("Generic1: R must be symmetric"))
    rankdef ≥ 0 || throw(ArgumentError("Generic1: rankdef must be ≥ 0, got $rankdef"))

    # Dense eigendecomposition for λ_max — acceptable at v0.1 scale
    # (Generic1 is for small user-supplied structure matrices).
    λ_max = maximum(eigvals(Symmetric(Matrix(R))))
    λ_max > 0 ||
        throw(ArgumentError("Generic1: largest eigenvalue of R must be positive, got $λ_max"))

    Rs = SparseMatrixCSC{Float64, Int}(R) ./ Float64(λ_max)
    con = constraint === nothing ? NoConstraint() : constraint
    return Generic1(Rs, Int(rankdef), hyperprior, con, Float64(λ_max))
end

Base.length(c::Generic1) = size(c.R, 1)
nhyperparameters(::Generic1) = 1
initial_hyperparameters(::Generic1) = [0.0]

precision_matrix(c::Generic1, θ) = exp(θ[1]) .* c.R

log_hyperprior(c::Generic1, θ) = log_prior_density(c.hyperprior, θ[1])

GMRFs.constraints(c::Generic1) = c.constraint

# Same convention as Generic0; matches R-INLA's `extra()` for
# `F_GENERIC1` (`inla.c:3568-3570`). Without β-mixing the `logdet_Q`
# correction in R-INLA collapses to zero, so the formula reduces to
# the F_GENERIC0 branch: `-½(n - rd) log(2π) + ½(n - rd) log τ`. The
# structural `½ log|R̃|_+` is dropped (see the Generic0 note for the
# R-INLA `mlik`-up-to-a-constant convention).
function log_normalizing_constant(c::Generic1, θ)
    n = size(c.R, 1)
    return -0.5 * (n - c.rd) * log(2π) + 0.5 * (n - c.rd) * θ[1]
end

function gmrf(c::Generic1, θ)
    # `c.R` is already eigen-rescaled at construction; pass it through
    # with `scale_model = false` to avoid Sørbye-Rue double scaling.
    return GMRFs.Generic0GMRF(c.R; τ = exp(θ[1]), rankdef = c.rd,
                               scale_model = false)
end

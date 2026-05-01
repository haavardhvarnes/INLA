"""
    Leroux(graph; hyperprior_tau = PCPrecision(),
           hyperprior_rho = LogitBeta(1.0, 1.0))

Leroux (1999) CAR component. The precision is

    Q(τ, ρ) = τ · ((1 - ρ) · I_n + ρ · R)

where `R = D - W` is the combinatorial graph Laplacian. The mixing
parameter `ρ ∈ (0, 1)` interpolates between pure IID (`ρ = 0`) and a
pure ICAR (`ρ = 1`, rank-deficient). For `ρ` strictly inside `(0, 1)`
the precision is positive definite and no sum-to-zero constraint is
needed, distinguishing Leroux from [`Besag`](@ref) / [`BYM`](@ref).

Two hyperparameters on the internal scale: `θ = [log(τ), logit(ρ)]`.
Default priors: PC precision on `τ` (matching `PCPrecision()`),
`Beta(1, 1)` on `ρ` (uniform), the latter expressed on the logit scale
via [`LogitBeta`](@ref).

R-INLA equivalent: `model = "besagproper2"` with the convex Leroux
combination, or `model = "leroux"` in the proprietary `inla` build.
"""
struct Leroux{Pτ <: AbstractHyperPrior, Pρ <: AbstractHyperPrior,
    G <: GMRFs.AbstractGMRFGraph} <: AbstractLatentComponent
    graph::G
    R::SparseMatrixCSC{Float64, Int}      # combinatorial Laplacian D - W
    hyperprior_tau::Pτ
    hyperprior_rho::Pρ
end

function Leroux(graph::GMRFs.AbstractGMRFGraph;
        hyperprior_tau::AbstractHyperPrior=PCPrecision(),
        hyperprior_rho::AbstractHyperPrior=LogitBeta(1.0, 1.0))
    R = SparseMatrixCSC{Float64, Int}(GMRFs.laplacian_matrix(graph))
    return Leroux(graph, R, hyperprior_tau, hyperprior_rho)
end

Leroux(W::AbstractMatrix; kwargs...) = Leroux(GMRFs.GMRFGraph(W); kwargs...)

Base.length(c::Leroux) = size(c.R, 1)
nhyperparameters(::Leroux) = 2
# log τ = 0, logit ρ = 0 (ρ = 0.5).
initial_hyperparameters(::Leroux) = [0.0, 0.0]

function precision_matrix(c::Leroux, θ)
    τ = exp(θ[1])
    ρ = inv(one(eltype(θ)) + exp(-θ[2]))
    n = size(c.R, 1)
    I_n = sparse(one(Float64) * I, n, n)
    return τ .* ((1 - ρ) .* I_n .+ ρ .* c.R)
end

function log_hyperprior(c::Leroux, θ)
    return log_prior_density(c.hyperprior_tau, θ[1]) +
           log_prior_density(c.hyperprior_rho, θ[2])
end

# Leroux is proper for ρ ∈ (0, 1): full Gaussian log-NC,
# `-½ n log(2π) + ½ log|Q|`. We compute log|Q| directly from a sparse
# Cholesky at the current θ; this is exact and matches R-INLA's
# proper-component bookkeeping.
function log_normalizing_constant(c::Leroux, θ)
    Q = precision_matrix(c, θ)
    F = cholesky(Symmetric(Q))
    n = size(c.R, 1)
    return -0.5 * n * log(2π) + 0.5 * logdet(F)
end

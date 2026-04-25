"""
    BYM(graph; hyperprior_iid = PCPrecision(), hyperprior_besag = PCPrecision(),
        scale_model = true)

Classical Besag-York-Mollié (1991) convolution model. The latent field
stacks an unstructured and a spatial piece side-by-side:

    x = [v; u] ∈ ℝ^{2n},   v ∼ N(0, τ_v^{-1} I_n),   u ∼ BesagGMRF(graph, τ_u).

The linear predictor uses the sum `v + u`; the caller supplies a
projector that emits `v + u` on the observation side (e.g. `[1 … 1 ; I_n
I_n]` for a single-intercept Poisson BYM model).

Two hyperparameters on the internal scale: `θ = [log τ_v, log τ_u]`.
Constraints: per-component sum-to-zero on the spatial block `u`
(Freni-Sterrantino et al. 2018); `v` is left unconstrained.

Distinct from [`BYM2`](@ref), which reparameterises `(τ_v, τ_u)` to a
total-precision / mixing pair `(τ, φ)`; both are shipped by R-INLA.
"""
struct BYM{Pv <: AbstractHyperPrior, Pu <: AbstractHyperPrior,
           G <: GMRFs.AbstractGMRFGraph} <: AbstractLatentComponent
    graph::G
    hyperprior_iid::Pv
    hyperprior_besag::Pu
    scale_model::Bool
end

function BYM(graph::GMRFs.AbstractGMRFGraph;
             hyperprior_iid::AbstractHyperPrior = PCPrecision(),
             hyperprior_besag::AbstractHyperPrior = PCPrecision(),
             scale_model::Bool = true)
    return BYM(graph, hyperprior_iid, hyperprior_besag, scale_model)
end

BYM(W::AbstractMatrix; kwargs...) = BYM(GMRFs.GMRFGraph(W); kwargs...)

Base.length(c::BYM) = 2 * GMRFs.num_nodes(c.graph)
nhyperparameters(::BYM) = 2
initial_hyperparameters(::BYM) = [0.0, 0.0]

function precision_matrix(c::BYM, θ)
    τ_v = exp(θ[1])
    τ_u = exp(θ[2])
    n = GMRFs.num_nodes(c.graph)
    Q_v = spdiagm(0 => fill(τ_v, n))
    Q_u = GMRFs.precision_matrix(GMRFs.BesagGMRF(c.graph; τ = τ_u,
                                                  scale_model = c.scale_model))
    return blockdiag(Q_v, SparseMatrixCSC{Float64, Int}(Q_u))
end

function log_hyperprior(c::BYM, θ)
    return log_prior_density(c.hyperprior_iid, θ[1]) +
           log_prior_density(c.hyperprior_besag, θ[2])
end

# Sum of an IID block (proper, dim n) and a Besag block (intrinsic,
# rank n - K). The Besag block's structural `½ log|R̃|_+` is dropped
# per R-INLA convention.
function log_normalizing_constant(c::BYM, θ)
    n = GMRFs.num_nodes(c.graph)
    K = GMRFs.nconnected_components(c.graph)
    return -0.5 * n * log(2π) + 0.5 * n * θ[1] + 0.5 * (n - K) * θ[2]
end

"""
    GMRFs.constraints(c::BYM) -> LinearConstraint

Per-connected-component sum-to-zero on the spatial block `u`
(positions `n+1:2n`). The IID block `v` is unconstrained.
"""
function GMRFs.constraints(c::BYM)
    n = GMRFs.num_nodes(c.graph)
    base = GMRFs.sum_to_zero_constraints(c.graph)
    A_u = GMRFs.constraint_matrix(base)
    e = GMRFs.constraint_rhs(base)
    r = size(A_u, 1)
    A = zeros(eltype(A_u), r, 2n)
    @views A[:, (n + 1):(2n)] .= A_u
    return GMRFs.LinearConstraint(A, e)
end

"""
    BYM(graph; hyperprior_iid = PCPrecision(), hyperprior_besag = PCPrecision(),
        scale_model = true)

Classical Besag–York–Mollié (1991) decomposition of a spatial random
effect into an iid (unstructured) part and an intrinsic CAR (Besag)
part — R-INLA's `model = "bym"`. The latent vector has length `2n`
and is arranged as `x = [v; b]`:

- `v` — IID component with precision `τ_v`. Length `n`.
- `b` — intrinsic Besag component with precision `τ_b`. Length `n`,
  subject to a sum-to-zero constraint per connected component
  (Freni-Sterrantino et al. 2018).

The combined effect entering the linear predictor is `v + b`; the user
projector `A` is responsible for that summation (typical pattern:
`A_block = [I_n | I_n]`).

Internal hyperparameters are `θ = [log τ_v, log τ_b]` (two precisions,
no mixing parameter — that is the BYM2 reparameterisation).

With `scale_model = true` (default) the Sørbye-Rue (2014) geometric-
mean scaling is applied to the Besag block, matching R-INLA ≥ 17.06.

The joint precision is block-diagonal:

    Q(τ_v, τ_b) = blockdiag(τ_v · I_n, τ_b · R)

where `R = c · L` is the (optionally scaled) Besag Laplacian.
"""
struct BYM{Pv <: AbstractHyperPrior, Pb <: AbstractHyperPrior,
    G <: GMRFs.AbstractGMRFGraph} <: AbstractLatentComponent
    graph::G
    hyperprior_iid::Pv
    hyperprior_besag::Pb
    scale_model::Bool
end

function BYM(graph::GMRFs.AbstractGMRFGraph;
        hyperprior_iid::AbstractHyperPrior=PCPrecision(),
        hyperprior_besag::AbstractHyperPrior=PCPrecision(),
        scale_model::Bool=true)
    return BYM(graph, hyperprior_iid, hyperprior_besag, scale_model)
end

BYM(W::AbstractMatrix; kwargs...) = BYM(GMRFs.GMRFGraph(W); kwargs...)

# Latent layout: [v; b] of total length 2n.
Base.length(c::BYM) = 2 * GMRFs.num_nodes(c.graph)
nhyperparameters(::BYM) = 2

# Initial internal hyperparameters: log τ_v = 0, log τ_b = 0.
initial_hyperparameters(::BYM) = [0.0, 0.0]

function precision_matrix(c::BYM, θ)
    τ_v = exp(θ[1])
    τ_b = exp(θ[2])
    n = GMRFs.num_nodes(c.graph)
    L = SparseMatrixCSC{Float64, Int}(GMRFs.laplacian_matrix(c.graph))
    if c.scale_model
        c_per_node = _bym_per_node_scale(c.graph)
        s = sqrt.(c_per_node)
        D = Diagonal(s)
        R_scaled = D * L * D
    else
        R_scaled = L
    end

    Is = Int[]
    Js = Int[]
    Vs = Float64[]
    # Q_11 = τ_v · I_n
    for i in 1:n
        push!(Is, i)
        push!(Js, i)
        push!(Vs, τ_v)
    end
    # Q_22 = τ_b · R_scaled
    rows = rowvals(R_scaled)
    vals = nonzeros(R_scaled)
    for col in 1:n
        for k in nzrange(R_scaled, col)
            push!(Is, n + rows[k])
            push!(Js, n + col)
            push!(Vs, τ_b * vals[k])
        end
    end
    return sparse(Is, Js, Vs, 2n, 2n)
end

# Per-node Sørbye-Rue constants on the underlying graph: c_per_node[i]
# is the scaling factor of the connected component containing node `i`
# (Freni-Sterrantino et al. 2018).
function _bym_per_node_scale(g::GMRFs.AbstractGMRFGraph)
    c_k = GMRFs.per_component_scale_factors(g)
    labels = GMRFs.connected_component_labels(g)
    return Float64[c_k[labels[i]] for i in 1:GMRFs.num_nodes(g)]
end

function log_hyperprior(c::BYM, θ)
    return log_prior_density(c.hyperprior_iid, θ[1]) +
           log_prior_density(c.hyperprior_besag, θ[2])
end

# Per-component log NC matching R-INLA's `extra()` for `F_BYM`
# (`inla.c:4868-4870`): the IID block contributes `½ n log τ_v` and
# the intrinsic Besag block contributes `½ (n - K) log τ_b`, with a
# shared Gaussian-NC piece `LOG_NORMC_GAUSSIAN · (n/2 + (n-K)/2)
# = -¼ (2n - K) log(2π)`. The structural `½ log|R̃|_+` term is
# absorbed into the joint `½ log|Q|_+` elsewhere.
function log_normalizing_constant(c::BYM, θ)
    n = GMRFs.num_nodes(c.graph)
    K = GMRFs.nconnected_components(c.graph)
    return -0.25 * (2n - K) * log(2π) + 0.5 * n * θ[1] + 0.5 * (n - K) * θ[2]
end

"""
    GMRFs.constraints(c::BYM) -> LinearConstraint

Sum-to-zero constraint on the Besag block `b` (positions `n+1:2n`),
with one row per connected component of the graph. The `v` block is
left unconstrained.
"""
function GMRFs.constraints(c::BYM)
    n = GMRFs.num_nodes(c.graph)
    base = GMRFs.sum_to_zero_constraints(c.graph)
    A_b = GMRFs.constraint_matrix(base)
    e = GMRFs.constraint_rhs(base)
    r = size(A_b, 1)
    A = zeros(eltype(A_b), r, 2n)
    @views A[:, (n + 1):(2n)] .= A_b
    return GMRFs.LinearConstraint(A, e)
end

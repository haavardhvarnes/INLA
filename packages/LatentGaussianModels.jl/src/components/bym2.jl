"""
    BYM2(graph; hyperprior_prec = PCPrecision(), hyperprior_phi = nothing,
         U = 0.5, α = DEFAULT_BYM2_PHI_ALPHA)

Riebler et al. (2016) BYM2 reparameterisation of the spatial +
unstructured random effect. The latent vector has length `2n` and is
arranged as `x = [b; u]`:

- `b = (1/√τ)(√(1-φ) v + √φ u)` — the combined mixed effect that
  enters the linear predictor. Length `n`.
- `u` — the scaled Besag spatial component with geometric-mean
  marginal variance `1` on its non-null subspace. Length `n`, subject
  to a sum-to-zero constraint per connected component
  (Freni-Sterrantino et al. 2018).

`τ > 0` is the overall precision of `b`; `φ ∈ [0, 1]` is the mixing
parameter (`φ = 0` is pure IID, `φ = 1` is pure spatial). Internal
hyperparameters are `θ = [log(τ), logit(φ)]`.

The joint precision of `x = [b; u]` is

    Q(τ, φ) = [  (τ/(1-φ)) I_n           -(√(τφ)/(1-φ)) I_n        ;
                 -(√(τφ)/(1-φ)) I_n      R_scaled + (φ/(1-φ)) I_n  ]

where `R_scaled = c · L` is the Sørbye-Rue–scaled Besag Laplacian.
Derivation: change of variables from `(v, u)` (with independent
quadratic forms `v'v + u' R_scaled u`) to `(b, u)` via
`v = (√τ b - √φ u)/√(1-φ)`.

If `hyperprior_phi = nothing`, a `PCBYM2Phi(U, α, γ)` is constructed
internally using the eigenvalues `γ` of `R_scaled` on its non-null
subspace. Defaults match Riebler et al. (2016); see
`plans/defaults-parity.md` for the ⚠ unverified-default note on `α`.
"""
struct BYM2{Pτ <: AbstractHyperPrior, Pφ <: AbstractHyperPrior,
            G <: GMRFs.AbstractGMRFGraph} <: AbstractLatentComponent
    graph::G
    c::Float64                   # Sørbye-Rue scaling constant
    γ::Vector{Float64}           # non-null eigenvalues of R_scaled = c·L
    hyperprior_prec::Pτ
    hyperprior_phi::Pφ
end

function BYM2(graph::GMRFs.AbstractGMRFGraph;
              hyperprior_prec::AbstractHyperPrior = PCPrecision(),
              hyperprior_phi::Union{Nothing, AbstractHyperPrior} = nothing,
              U::Real = 0.5, α::Real = DEFAULT_BYM2_PHI_ALPHA)
    # Compute the scaled-structure eigenvalues γ on the non-null subspace.
    c = GMRFs.scale_factor(graph)
    L = GMRFs.laplacian_matrix(graph)
    r = GMRFs.nconnected_components(graph)
    eigs = eigvals(Symmetric(Matrix{Float64}(L)))
    sort!(eigs)
    # Drop the r smallest eigenvalues (numerically zero for the null space).
    nz_eigs = eigs[(r + 1):end]
    γ = c .* nz_eigs
    prior_phi = hyperprior_phi === nothing ? PCBYM2Phi(U, α, γ) : hyperprior_phi
    return BYM2(graph, Float64(c), γ, hyperprior_prec, prior_phi)
end

BYM2(W::AbstractMatrix; kwargs...) = BYM2(GMRFs.GMRFGraph(W); kwargs...)

# The latent field is [b; u] of total length 2n.
Base.length(c::BYM2) = 2 * GMRFs.num_nodes(c.graph)
nhyperparameters(::BYM2) = 2

# Initial internal hyperparameters: log(τ) = 0 (τ = 1), logit(φ) = 0 (φ = 0.5).
initial_hyperparameters(::BYM2) = [0.0, 0.0]

"""
    _bym2_params(θ) -> (τ, φ)

Map internal `θ = [log τ, logit φ]` to user-facing `(τ, φ)`.
"""
function _bym2_params(θ)
    τ = exp(θ[1])
    φ = inv(one(eltype(θ)) + exp(-θ[2]))
    return τ, φ
end

function precision_matrix(c::BYM2, θ)
    τ, φ = _bym2_params(θ)
    n = GMRFs.num_nodes(c.graph)
    L = GMRFs.laplacian_matrix(c.graph)
    R_scaled = c.c .* SparseMatrixCSC{Float64, Int}(L)

    # Block entries:
    a = τ / (1 - φ)                 # Q_11 = a · I
    b = -sqrt(τ * φ) / (1 - φ)      # Q_12 = Q_21 = b · I
    d = φ / (1 - φ)                 # adds to Q_22 diagonal

    # Assemble 2n × 2n sparse block matrix via COO construction.
    Is = Int[]; Js = Int[]; Vs = Float64[]
    # Q_11 = a·I_n  (block rows 1:n, cols 1:n)
    for i in 1:n
        push!(Is, i); push!(Js, i); push!(Vs, a)
    end
    # Q_12 = b·I_n, Q_21 = b·I_n (blocks (1:n, n+1:2n) and (n+1:2n, 1:n))
    for i in 1:n
        push!(Is, i);     push!(Js, n + i); push!(Vs, b)
        push!(Is, n + i); push!(Js, i);     push!(Vs, b)
    end
    # Q_22 = R_scaled + d·I_n (block rows n+1:2n, cols n+1:2n)
    rows = rowvals(R_scaled)
    vals = nonzeros(R_scaled)
    for col in 1:n
        for k in nzrange(R_scaled, col)
            push!(Is, n + rows[k]); push!(Js, n + col); push!(Vs, vals[k])
        end
    end
    for i in 1:n
        push!(Is, n + i); push!(Js, n + i); push!(Vs, d)
    end

    return sparse(Is, Js, Vs, 2n, 2n)
end

function log_hyperprior(c::BYM2, θ)
    lp_τ = log_prior_density(c.hyperprior_prec, θ[1])
    lp_φ = log_prior_density(c.hyperprior_phi, θ[2])
    return lp_τ + lp_φ
end

# R-INLA convention for BYM2: drop the structural log|R̃|_+ from the
# component log NC; the global Marriott-Van Loan correction in
# `laplace_mode` carries the constraint terms. Latent dimension is
# `2n` and the prior has rank `2n - K` where K = nconnected_components.
# After the (v,u) → (b,u) change of variables (Jacobian (τ/(1-φ))^{n/2}),
# the b block contributes the n-dimensional log NC `-½ n log(2π) +
# ½ n log(τ/(1-φ))`; the u block's structural `½ log|R̃|_+` is dropped
# in the R-INLA convention.
function log_normalizing_constant(c::BYM2, θ)
    τ, φ = _bym2_params(θ)
    n = GMRFs.num_nodes(c.graph)
    return -0.5 * n * log(2π) + 0.5 * n * log(τ / (1 - φ))
end

"""
    GMRFs.constraints(c::BYM2) -> LinearConstraint

Sum-to-zero constraint on the spatial block `u` (positions `n+1:2n`),
with one row per connected component of the graph. The `b` block is
left unconstrained.
"""
function GMRFs.constraints(c::BYM2)
    n = GMRFs.num_nodes(c.graph)
    base = GMRFs.sum_to_zero_constraints(c.graph)
    A_u = GMRFs.constraint_matrix(base)
    e = GMRFs.constraint_rhs(base)
    r = size(A_u, 1)
    A = zeros(eltype(A_u), r, 2n)
    @views A[:, (n + 1):(2n)] .= A_u
    return GMRFs.LinearConstraint(A, e)
end

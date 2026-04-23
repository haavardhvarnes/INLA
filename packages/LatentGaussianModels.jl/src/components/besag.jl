"""
    Besag(graph; hyperprior = PCPrecision(), scale_model = true)

Intrinsic CAR (Besag) component on the supplied `graph`
(`AbstractGMRFGraph` or adjacency matrix). One hyperparameter on the
internal scale `θ = log(τ)`.

With `scale_model = true` (default) the Sørbye-Rue (2014) geometric-
mean scaling is applied, matching R-INLA ≥ 17.06. Carries one
sum-to-zero constraint per connected component of `graph`
(Freni-Sterrantino et al. 2018).
"""
struct Besag{P <: AbstractHyperPrior, G <: GMRFs.AbstractGMRFGraph} <: AbstractLatentComponent
    graph::G
    hyperprior::P
    scale_model::Bool
end

function Besag(graph::GMRFs.AbstractGMRFGraph;
               hyperprior::AbstractHyperPrior = PCPrecision(),
               scale_model::Bool = true)
    return Besag(graph, hyperprior, scale_model)
end

Besag(W::AbstractMatrix; kwargs...) = Besag(GMRFs.GMRFGraph(W); kwargs...)

Base.length(c::Besag) = GMRFs.num_nodes(c.graph)
nhyperparameters(::Besag) = 1
initial_hyperparameters(::Besag) = [0.0]

function gmrf(c::Besag, θ)
    return GMRFs.BesagGMRF(c.graph; τ = exp(θ[1]), scale_model = c.scale_model)
end

precision_matrix(c::Besag, θ) = GMRFs.precision_matrix(gmrf(c, θ))
log_hyperprior(c::Besag, θ) = log_prior_density(c.hyperprior, θ[1])
GMRFs.constraints(c::Besag) = GMRFs.sum_to_zero_constraints(c.graph)

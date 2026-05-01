using LatentGaussianModels
using LatentGaussianModels: laplace_mode, LatentGaussianModel,
                            GaussianLikelihood, IID, Intercept, BYM2, PoissonLikelihood,
                            PCPrecision
using LatentGaussianModels: model_constraints, _constrained_marginal_variances
using GMRFs: GMRFs, GMRFGraph, LinearConstraint, NoConstraint,
             constraint_matrix, constraint_rhs
using SparseArrays
using LinearAlgebra
using Random
using Statistics: mean

# Small hand-built component that exposes a user-supplied constraint —
# lets us test the constrained Laplace path independently of BYM2 /
# Besag (which are the downstream consumers).
struct _ConstrainedIID{P} <: LatentGaussianModels.AbstractLatentComponent
    n::Int
    hyperprior::P
    A_constr::Matrix{Float64}
    e_constr::Vector{Float64}
end
Base.length(c::_ConstrainedIID) = c.n
LatentGaussianModels.nhyperparameters(::_ConstrainedIID) = 1
LatentGaussianModels.initial_hyperparameters(::_ConstrainedIID) = [0.0]
function LatentGaussianModels.precision_matrix(c::_ConstrainedIID, θ)
    τ = exp(θ[1])
    return spdiagm(0 => fill(τ, c.n))
end
function LatentGaussianModels.log_hyperprior(c::_ConstrainedIID, θ)
    LatentGaussianModels.log_prior_density(c.hyperprior, θ[1])
end
GMRFs.constraints(c::_ConstrainedIID) = LinearConstraint(c.A_constr, c.e_constr)

@testset "Laplace — sum-to-zero constraint, Gaussian likelihood" begin
    # Gaussian likelihood with identity link + an IID random effect under
    # a hard sum-to-zero constraint. The constrained mode has a closed
    # form: x̂ = (I - 11'/n) ẑ where ẑ is the unconstrained MAP.
    rng = Random.Xoshiro(20260423)
    n = 10
    σ_obs = 0.2
    τ_obs = 1 / σ_obs^2
    τ_u = 4.0
    x_true = randn(rng, n)
    x_true .-= mean(x_true)                            # enforce ground truth
    y = x_true .+ σ_obs .* randn(rng, n)

    # Model: y = u + ε, u ~ N(0, τ_u^{-1} I), 1'u = 0
    A_constr = reshape(ones(n), 1, n)
    c_u = _ConstrainedIID(n, PCPrecision(1.0, 0.01), A_constr, [0.0])
    ℓ = GaussianLikelihood()
    A_proj = sparse(1.0I, n, n)
    model = LatentGaussianModel(ℓ, (c_u,), A_proj)

    # Fit at the true hyperparameters.
    θ = [log(τ_obs), log(τ_u)]
    res = laplace_mode(model, y, θ)

    @test res.converged
    @test res.constraint !== nothing
    @test size(res.constraint.U, 2) == 1

    # C x̂ = 0 to working precision.
    @test abs(sum(res.mode)) < 1.0e-10

    # Analytic constrained MAP: (τ_obs I + τ_u I)x - τ_obs y = λ·1,
    # with 1'x = 0. Scalar system: let α = τ_obs/(τ_obs+τ_u); then
    # x̂ = α (y - ȳ·1).
    α = τ_obs / (τ_obs + τ_u)
    x̂_analytic = α .* (y .- mean(y))
    @test maximum(abs.(res.mode .- x̂_analytic)) < 1.0e-8

    # Constraint-corrected marginal variances: analytic form is
    # (τ_obs+τ_u)^{-1} * (1 - 1/n) = diagonal term minus projection onto
    # the constrained-out direction.
    var_true = (1 - 1 / n) / (τ_obs + τ_u)
    v̂ = _constrained_marginal_variances(res.precision, res.constraint)
    @test maximum(abs.(v̂ .- var_true)) < 1.0e-10
end

@testset "model_constraints assembly" begin
    # Verify that per-component constraints are correctly embedded into
    # the stacked x layout.
    n = 5
    A_u = reshape(ones(n), 1, n)
    c1 = _ConstrainedIID(n, PCPrecision(1.0, 0.01), A_u, [0.0])
    c2 = _ConstrainedIID(n, PCPrecision(1.0, 0.01), A_u, [0.0])
    ℓ = GaussianLikelihood()
    A_proj = sparse([I(n) I(n)])
    model = LatentGaussianModel(ℓ, (c1, c2), A_proj)

    mc = model_constraints(model)
    @test mc isa LinearConstraint
    C = constraint_matrix(mc)
    @test size(C) == (2, 2n)
    @test all(C[1, 1:n] .== 1.0)
    @test all(C[1, (n + 1):(2n)] .== 0.0)
    @test all(C[2, 1:n] .== 0.0)
    @test all(C[2, (n + 1):(2n)] .== 1.0)

    # Pure-proper model: no constraints.
    model2 = LatentGaussianModel(ℓ, (Intercept(),), sparse(ones(n, 1)))
    @test model_constraints(model2) isa NoConstraint
end

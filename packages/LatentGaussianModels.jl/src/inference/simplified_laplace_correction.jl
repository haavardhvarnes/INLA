# Rue-Martino simplified-Laplace mean-shift correction for the latent
# posterior at fixed θ. Used by `fit(::LatentGaussianModel, y, ::INLA)`
# when `latent_strategy = :simplified_laplace`. Independent of the
# Hermite density-shape correction in `_latent_skewness` (marginals.jl):
# this corrects the *summary* `INLAResult.x_mean` / `x_var`; that one
# corrects the per-coordinate density of `p(x_i | y)`.
#
# Reference: Rue, Martino, Chopin (2009) §4.2. The reference repo
# `haavardhvarnes/IntegratedNestedLaplace.jl` (`laplace_eval`) uses the
# same formula; we re-implement here to match our `LaplaceResult` and
# constraint-handling conventions.

"""
    _sla_mean_shift(lp::LaplaceResult, model::LatentGaussianModel, y)
        -> Vector{Float64}

Simplified-Laplace mean-shift correction at the Laplace fit `lp`:

    Δx = ½ H⁻¹ Aᵀ (h³ ⊙ σ²_η)

where:

- `H` is the constraint-regularised Laplace Hessian factored in
  `lp.factor`.
- `h³_i = ∇³_η log p(y_i | η_i, θ_ℓ)` evaluated at `η̂ = A x̂`.
- `σ²_η_i = (A H⁻¹ Aᵀ)_ii` is the conditional variance of the linear
  predictor under the Laplace at `θ`, with the Rue-Held kriging
  correction applied when the model carries hard linear constraints.

Returns a zero vector when `h³ ≡ 0` (Gaussian-likelihood collapse) so
that `latent_strategy = :simplified_laplace` reduces exactly to the
`:gaussian` path on quadratic-in-η likelihoods.

Cost per call: one multi-RHS sparse triangular solve (`H⁻¹ Aᵀ`,
`n_x × n_obs`) plus one vector solve. With `FactorCache` reuse this is
roughly one inner Newton iteration per integration point.
"""
function _sla_mean_shift(lp::LaplaceResult,
                         model::LatentGaussianModel,
                         y)
    A = model.A
    ℓ = model.likelihood
    n_ℓ = nhyperparameters(ℓ)
    θ_ℓ = n_ℓ > 0 ? lp.θ[1:n_ℓ] : Float64[]
    η̂ = A * lp.mode

    h³ = ∇³_η_log_density(ℓ, y, η̂, θ_ℓ)
    all(iszero, h³) && return zeros(Float64, length(lp.mode))

    # Z = H⁻¹ Aᵀ via multi-RHS sparse Cholesky on the cached factor
    # (FactorCache `\` supports AbstractVecOrMat). Materialise Aᵀ as a
    # dense `n_x × n_obs` block — the solver handles dense RHS faster
    # than sparse and the block size is bounded by the latent dimension.
    Z = lp.factor \ Matrix(transpose(A))
    if lp.constraint !== nothing
        # Project each column onto null(C) using the same kriging
        # identity as `_latent_skewness` (marginals.jl). After this,
        # `Z` is the columnwise constrained-inverse `Aᵀ` action and
        # `diag(A · Z)` is the constraint-corrected `σ²_η`.
        U = lp.constraint.U
        W_fact = lp.constraint.W_fact
        C = lp.constraint.C
        Z .-= U * (W_fact \ (C * Z))
    end

    # σ²_η_i = (A H⁻¹ Aᵀ)_ii = Σ_j A_ij Z_ji, computed without forming
    # the full `n_obs × n_obs` matrix `A · Z`. Sparse-times-dense
    # broadcasting respects A's pattern.
    σ²_η = vec(sum(A .* transpose(Z), dims = 2))

    # Δx = ½ H⁻¹ Aᵀ (h³ ⊙ σ²_η). One vector solve, then project onto
    # null(C) so that `C(x̂ + Δx) = C x̂ = e` is preserved exactly.
    rhs = transpose(A) * (h³ .* σ²_η)
    Δx = lp.factor \ Vector(rhs)
    if lp.constraint !== nothing
        U = lp.constraint.U
        W_fact = lp.constraint.W_fact
        C = lp.constraint.C
        Δx .-= U * (W_fact \ (C * Δx))
    end

    return 0.5 .* Δx
end

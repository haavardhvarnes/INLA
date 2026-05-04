# FullLaplace marginal strategy — R-INLA's `strategy = "laplace"`.
#
# Per-`x_i` refitted Laplace via constraint injection: for each output
# grid point `a`, run `laplace_mode` with the augmented constraint
# `[C_model; e_i^T] x = [e_model; a]` and read the constrained Laplace
# log-marginal. The per-θ density `p(x_i = a | θ_k, y)` is then
#
#     p(x_i = a | θ_k, y) ∝ exp(log p̂(y | θ_k, x_i = a) - log p̂(y | θ_k))
#
# with the unconstrained log-marginal `log p̂(y | θ_k)` read off the
# pre-existing `INLAResult.laplaces[k].log_marginal`. The mixture
# `p(x_i | y) = Σ_k w_k p(x_i | θ_k, y)` reuses `INLAResult.θ_weights`.
#
# Each per-θ density is renormalised on the output grid (trapezoid),
# which absorbs the θ-dependent additive constant in the constrained-
# Laplace approximation (Marriott-Van Loan vs. R-INLA's exact
# `½ log|Q_{-i,-i}|` differ by an `a`-independent term per θ_k — see
# the algebra in ADR-026 and Rue-Martino-Chopin (2009) §4.3).
#
# PR-3 ships the core. PR-4 will add: rank-1 `FactorCache` update for
# the per-`x_i` constraint (avoiding a fresh sparse Cholesky per
# evaluation point), an adaptive coarse refit grid + interpolation,
# and `FullLaplace`-aware integration-stage summaries for
# `INLAResult.x_mean` / `x_var`.

"""
    laplace_mode_fixed_xi(model, y, θ, i, a;
                          strategy = Laplace()) -> LaplaceResult

Constrained Laplace fit with the additional hard linear constraint
`x_i = a` stacked onto the model-level constraints declared by
[`model_constraints`](@ref). Forwards to [`laplace_mode`](@ref) with
`extra_constraint = (rows = e_i^T, rhs = [a])`.

Used by [`FullLaplace`](@ref) inside [`posterior_marginal_x`](@ref).
"""
function laplace_mode_fixed_xi(m::LatentGaussianModel,
        y, θ::AbstractVector{<:Real},
        i::Integer, a::Real;
        strategy::Laplace=Laplace())
    1 ≤ i ≤ m.n_x ||
        throw(ArgumentError("laplace_mode_fixed_xi: index $i out of " *
                            "bounds (1:$(m.n_x))"))
    rows = zeros(Float64, 1, m.n_x)
    rows[1, i] = 1.0
    return laplace_mode(m, y, θ;
        strategy=strategy,
        extra_constraint=(rows=rows, rhs=[Float64(a)]))
end

# Per-θ constrained-Laplace density of `x_i` on the output grid `xs`.
# Returns the renormalised per-θ density (integrates to 1 on `xs` via
# trapezoid). The constrained-Laplace log-marginal differs from the
# exact log p_LA(x_i | θ_k, y) by an a-independent constant per θ_k;
# trapezoid renormalisation absorbs that constant exactly.
#
# Failure paths (caught and dropped, mirroring `_inla_integrate`):
#  - `laplace_mode_fixed_xi` throws (typically a `PosDefException` from
#    Cholesky on `H_reg` when `a` lands too far in the tail).
#  - `lp_xi.log_marginal` is non-finite.
#  - All grid points fail → returns a zero vector for this θ_k (the
#    caller falls through to skipping the contribution).
function _full_laplace_per_θ_pdf(model::LatentGaussianModel, y,
        lp_k::LaplaceResult, log_p_y::Real,
        i::Integer, xs::AbstractVector{<:Real},
        laplace::Laplace)
    log_pdf = fill(-Inf, length(xs))
    @inbounds for (j, x_j) in pairs(xs)
        local lp_xi
        try
            lp_xi = laplace_mode_fixed_xi(model, y, lp_k.θ, i, x_j;
                strategy=laplace)
        catch
            continue
        end
        isfinite(lp_xi.log_marginal) || continue
        log_pdf[j] = lp_xi.log_marginal - log_p_y
    end
    max_log = maximum(log_pdf)
    isfinite(max_log) || return zeros(Float64, length(xs))
    pdf = exp.(log_pdf .- max_log)
    Z = _trapz(xs, pdf)
    Z > 0 || return zeros(Float64, length(xs))
    return pdf ./ Z
end

# Mix per-θ FullLaplace densities into the user-facing posterior
# marginal `p(x_i | y) = Σ_k w_k p(x_i | θ_k, y)`. Each per-θ density
# is renormalised independently on the output grid `xs` so that
# additive constants in the constrained-Laplace approximation cancel
# (see ADR-026). The mixture is then renormalised once more for safety
# against grid-truncation error in the per-θ trapezoids.
function _full_laplace_pdf(res::INLAResult, i::Integer,
        model::LatentGaussianModel, y,
        xs::AbstractVector{<:Real};
        laplace::Laplace=Laplace())
    pdf = zeros(Float64, length(xs))
    for k in eachindex(res.laplaces)
        w = res.θ_weights[k]
        w == 0 && continue
        lp_k = res.laplaces[k]
        log_p_y = lp_k.log_marginal
        isfinite(log_p_y) || continue
        pdf_k = _full_laplace_per_θ_pdf(model, y, lp_k, log_p_y, i, xs,
            laplace)
        pdf .+= w .* pdf_k
    end
    Z = length(xs) ≥ 2 ? _trapz(xs, pdf) : zero(eltype(pdf))
    Z > 0 && (pdf ./= Z)
    return pdf
end

# Trapezoid quadrature on a (possibly non-uniform) 1D grid.
function _trapz(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real})
    n = length(xs)
    n == length(ys) ||
        throw(DimensionMismatch("_trapz: xs and ys must have equal length"))
    n < 2 && return zero(eltype(ys))
    s = zero(eltype(ys))
    @inbounds for j in 2:n
        s += 0.5 * (xs[j] - xs[j - 1]) * (ys[j] + ys[j - 1])
    end
    return s
end

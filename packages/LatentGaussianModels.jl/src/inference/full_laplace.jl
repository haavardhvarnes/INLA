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
                          strategy = Laplace(),
                          x0 = nothing) -> LaplaceResult

Constrained Laplace fit with the additional hard linear constraint
`x_i = a` stacked onto the model-level constraints declared by
[`model_constraints`](@ref). Forwards to [`laplace_mode`](@ref) with
`extra_constraint = (rows = e_i^T, rhs = [a])`.

`x0` (optional) seeds the inner Newton iteration. Used by
[`FullLaplace`](@ref) to warm-start adjacent grid-point refits from the
previous fit's mode — for a smooth posterior the conditional mode at
`a + Δa` differs from the mode at `a` by `O(Δa)`, so 1-2 Newton steps
suffice in place of a fresh ~5-10 step descent.

Used by [`FullLaplace`](@ref) inside [`posterior_marginal_x`](@ref).
"""
function laplace_mode_fixed_xi(m::LatentGaussianModel,
        y, θ::AbstractVector{<:Real},
        i::Integer, a::Real;
        strategy::Laplace=Laplace(),
        x0::Union{Nothing, AbstractVector{<:Real}}=nothing)
    1 ≤ i ≤ m.n_x ||
        throw(ArgumentError("laplace_mode_fixed_xi: index $i out of " *
                            "bounds (1:$(m.n_x))"))
    rows = zeros(Float64, 1, m.n_x)
    rows[1, i] = 1.0
    return laplace_mode(m, y, θ;
        strategy=strategy,
        x0=x0,
        extra_constraint=(rows=rows, rhs=[Float64(a)]))
end

# Per-θ constrained-Laplace density of `x_i` on the output grid `xs`.
# Returns the renormalised per-θ density (integrates to 1 on `xs` via
# trapezoid). The constrained-Laplace log-marginal differs from the
# exact log p_LA(x_i | θ_k, y) by an a-independent constant per θ_k;
# trapezoid renormalisation absorbs that constant exactly.
#
# Sweeps grid points outward from the unconstrained-mode coordinate
# `lp_k.mode[i]` in two directions, warm-starting each Newton refit
# from the previous fit's mode. For a smooth posterior the conditional
# mode at `a + Δa` differs from the mode at `a` by `O(Δa)`, so the
# warm-started Newton converges in 1-2 steps in place of the fresh
# ~5-10 step descent. The two-sweep order (out-from-center, not
# left-to-right) keeps the warm-start mode at most one grid step away
# from the next conditional mode in either tail direction.
#
# Adaptive early termination: each sweep stops once the running per-θ
# log-density drops more than `_TRUNCATE_THRESHOLD` (= 25 nats) below
# the running maximum — the density at that point is `< 1.4e-11` of
# the peak, so refits in the deeper tail contribute no measurable mass.
# Truncated grid points retain `log_pdf = -Inf`, so `pdf[j] = 0` after
# the `exp` and trapezoid normalisation.
#
# Failure paths (caught and dropped, mirroring `_inla_integrate`):
#  - `laplace_mode_fixed_xi` throws (typically a `PosDefException` from
#    Cholesky on `H_reg` when `a` lands too far in the tail).
#  - `lp_xi.log_marginal` is non-finite.
#  - All grid points fail → returns a zero vector for this θ_k (the
#    caller falls through to skipping the contribution).
const _TRUNCATE_THRESHOLD = 25.0

function _full_laplace_per_θ_pdf(model::LatentGaussianModel, y,
        lp_k::LaplaceResult, log_p_y::Real,
        i::Integer, xs::AbstractVector{<:Real},
        laplace::Laplace)
    n = length(xs)
    n == 0 && return zeros(Float64, 0)
    log_pdf = fill(-Inf, n)

    # Locate the grid point closest to the unconstrained-mode coordinate
    # — the warm-start anchor for both sweeps.
    center = lp_k.mode[i]
    j_center = argmin(abs.(xs .- center))

    new_mode = _try_fixed_xi_fit!(log_pdf, j_center, model, y, lp_k.θ,
        i, xs[j_center], log_p_y, laplace, lp_k.mode)
    x_right = new_mode === nothing ? Vector{Float64}(lp_k.mode) : copy(new_mode)
    x_left = new_mode === nothing ? Vector{Float64}(lp_k.mode) : copy(new_mode)

    # Sweep right.
    running_max = isfinite(log_pdf[j_center]) ? log_pdf[j_center] : -Inf
    for j in (j_center + 1):n
        new_mode = _try_fixed_xi_fit!(log_pdf, j, model, y, lp_k.θ,
            i, xs[j], log_p_y, laplace, x_right)
        new_mode === nothing || (x_right = new_mode)
        isfinite(log_pdf[j]) || continue
        running_max = max(running_max, log_pdf[j])
        log_pdf[j] - running_max < -_TRUNCATE_THRESHOLD && break
    end

    # Sweep left.
    running_max = isfinite(log_pdf[j_center]) ? log_pdf[j_center] : -Inf
    for j in (j_center - 1):-1:1
        new_mode = _try_fixed_xi_fit!(log_pdf, j, model, y, lp_k.θ,
            i, xs[j], log_p_y, laplace, x_left)
        new_mode === nothing || (x_left = new_mode)
        isfinite(log_pdf[j]) || continue
        running_max = max(running_max, log_pdf[j])
        log_pdf[j] - running_max < -_TRUNCATE_THRESHOLD && break
    end

    max_log = maximum(log_pdf)
    isfinite(max_log) || return zeros(Float64, n)
    pdf = exp.(log_pdf .- max_log)
    Z = _trapz(xs, pdf)
    Z > 0 || return zeros(Float64, n)
    return pdf ./ Z
end

# Single fixed-`x_i` fit + log-pdf write. Returns the converged mode on
# success (for the caller to use as next warm-start), `nothing` on
# failure (caller keeps the previous warm-start).
function _try_fixed_xi_fit!(log_pdf::Vector{Float64}, j::Integer,
        model::LatentGaussianModel, y, θ::AbstractVector{<:Real},
        i::Integer, a::Real, log_p_y::Real,
        laplace::Laplace, x_warm::AbstractVector{<:Real})
    local lp_xi
    try
        lp_xi = laplace_mode_fixed_xi(model, y, θ, i, a;
            strategy=laplace, x0=x_warm)
    catch
        return nothing
    end
    isfinite(lp_xi.log_marginal) || return nothing
    log_pdf[j] = lp_xi.log_marginal - log_p_y
    return lp_xi.mode
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

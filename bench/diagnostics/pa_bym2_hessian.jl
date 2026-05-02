# bench/diagnostics/pa_bym2_hessian.jl
#
# Phase Q diagnostic: Pennsylvania BYM2 ~2× regression at the θ-integration
# layer. Replan-2026-04-28 hypothesis is "wider Hessian at θ̂ → wider grid
# envelope → many design points contribute negligibly". This script collects
# the data needed to confirm or refute that:
#
#   1. Run `inla(model, y; int_strategy = :grid)` to get θ̂, the FiniteDiff
#      production Hessian, and the design-point weights.
#   2. Recompute the Hessian at θ̂ via hand-rolled central differences at
#      multiple step sizes — checks whether FiniteDiff.jl's default step
#      is a load-bearing tuning choice.
#   3. Sample `−log π̂(θ | y)` along each eigenaxis of `Σθ` at `t ∈ {±k σ}`
#      for k = 0.5, 1, 2, 3 — fits a 5-point stencil curvature and compares
#      to `1/σ²` from the production Hessian. If the empirical curvature is
#      significantly larger than the FD value, the FD Hessian is *narrower*
#      than reality (= wider Σ, wider grid envelope, more wasted points).
#   4. Report the design-point effectiveness: ESS, fraction with weight >
#      various thresholds, max weight.
#
# Run from repo root:
#
#   julia --project=bench bench/diagnostics/pa_bym2_hessian.jl
#
# Output is plain-text to stdout. No JSON / fixture writes.

using JLD2: jldopen
using SparseArrays
using LinearAlgebra: I, Symmetric, eigen, Diagonal, logdet, dot
using Printf: @printf, @sprintf
using Statistics: mean

using LatentGaussianModels: PoissonLikelihood, Intercept, FixedEffects, BYM2,
                            LatentGaussianModel, INLA, Laplace, PCPrecision,
                            log_hyperprior, fixed_effects, hyperparameters,
                            log_marginal_likelihood, inla, laplace_mode
using GMRFs: GMRFGraph

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const FIXTURE_PATH = joinpath(REPO_ROOT, "packages", "LatentGaussianModels.jl",
    "test", "oracle", "fixtures", "pennsylvania_bym2.jld2")

println("=" ^ 72)
println("PA BYM2 — Phase Q Hessian diagnostic")
println("=" ^ 72)
println("fixture: ", FIXTURE_PATH)

fx = jldopen(FIXTURE_PATH, "r") do f
    f["fixture"]
end
inp = fx["input"]
y = Int.(inp["cases"])
E = Float64.(inp["expected"])
x_cov = Float64.(inp["x"])
W = inp["W"]
n = length(y)
println("n = $n  (Pennsylvania counties)")

ℓ = PoissonLikelihood(; E = E)
c_int = Intercept()
c_beta = FixedEffects(1)
c_bym2 = BYM2(GMRFGraph(W); hyperprior_prec = PCPrecision(1.0, 0.01))
A = sparse(hcat(
    ones(n),
    reshape(x_cov, n, 1),
    Matrix{Float64}(I, n, n),
    zeros(n, n),
))
model = LatentGaussianModel(ℓ, (c_int, c_beta, c_bym2), A)

# Closure: −log π̂(θ | y). Mirrors `_neg_log_posterior_θ` in inla.jl, with
# the same try/catch on Laplace failure → Inf so it composes cleanly with
# central-difference stencils that may probe extreme θ.
function neg_lp(θ::AbstractVector{<:Real})
    res = try
        laplace_mode(model, y, θ; strategy = Laplace())
    catch
        return Inf
    end
    isfinite(res.log_marginal) || return Inf
    return -(res.log_marginal + log_hyperprior(model, θ))
end

# ---------------------------------------------------------------------
# Stage 1 — production fit, θ̂, design weights, timing
# ---------------------------------------------------------------------

println("\n", "─" ^ 72)
println("Stage 1 — production INLA fit")
println("─" ^ 72)

# Warmup
res_warm = inla(model, y; int_strategy = :grid)
println("warmup mlik = ", round(res_warm.log_marginal, digits = 4))

t_total = @elapsed res = inla(model, y; int_strategy = :grid)
@printf("total fit time           : %7.3f s\n", t_total)
println("θ̂                       : ", round.(res.θ̂, digits = 4))
println("design points            : ", length(res.θ_weights))
println("design-point ESS (1/Σw²) : ",
    round(1.0 / sum(w^2 for w in res.θ_weights), digits = 2))
@printf("max weight               : %.4f\n", maximum(res.θ_weights))
@printf("mean weight              : %.4f\n", mean(res.θ_weights))
println("# weights > 0.10         : ", count(>(0.10), res.θ_weights))
println("# weights > 0.05         : ", count(>(0.05), res.θ_weights))
println("# weights > 0.01         : ", count(>(0.01), res.θ_weights))
println("# weights ≤ 1e-3         : ", count(≤(1.0e-3), res.θ_weights))
println("Σθ (production):")
display(round.(res.Σθ, digits = 4))
F_prod = eigen(Symmetric(res.Σθ))
println("eigenvalues of Σθ        : ", round.(F_prod.values, digits = 5))
println("σ per eigenaxis          : ", round.(sqrt.(F_prod.values), digits = 4))
println("eigenvalues of H = Σ⁻¹   : ", round.(1 ./ F_prod.values, digits = 2))

# ---------------------------------------------------------------------
# Stage 2 — central-difference Hessian sensitivity to step size
# ---------------------------------------------------------------------

# Hand-rolled central-difference Hessian. Standard 5-point on the diagonal,
# 4-point cross-derivative off-diagonal. Step h is per-coordinate; pass an
# absolute h since θ̂ is on log-precision scale where ~1 is natural.
function central_hessian(f, x::AbstractVector{<:Real}, h::Real)
    n = length(x)
    H = Matrix{Float64}(undef, n, n)
    f0 = f(x)
    for i in 1:n
        # ∂²/∂x_i²: (f(x+h e_i) − 2 f(x) + f(x−h e_i)) / h²
        xp = copy(x); xp[i] += h
        xm = copy(x); xm[i] -= h
        H[i, i] = (f(xp) - 2 * f0 + f(xm)) / h^2
    end
    for i in 1:n, j in (i + 1):n
        # ∂²/∂x_i∂x_j: 4-point cross
        xpp = copy(x); xpp[i] += h; xpp[j] += h
        xpm = copy(x); xpm[i] += h; xpm[j] -= h
        xmp = copy(x); xmp[i] -= h; xmp[j] += h
        xmm = copy(x); xmm[i] -= h; xmm[j] -= h
        c = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h^2)
        H[i, j] = c
        H[j, i] = c
    end
    return H
end

println("\n", "─" ^ 72)
println("Stage 2 — Hessian sensitivity to step size (central differences)")
println("─" ^ 72)

θ̂ = res.θ̂
for h in (1.0e-2, 5.0e-3, 1.0e-3, 5.0e-4, 1.0e-4)
    H_h = central_hessian(neg_lp, θ̂, h)
    Σ_h = inv(Symmetric(H_h))
    F_h = eigen(Symmetric(Σ_h))
    @printf("\nh = %.1e\n", h)
    println("  H:  ", round.(H_h, digits = 3))
    println("  Σ:  ", round.(Σ_h, digits = 5))
    println("  eigvals(Σ): ", round.(F_h.values, digits = 5))
    println("  σ axis    : ", round.(sqrt.(max.(F_h.values, 0.0)), digits = 4))
end

# ---------------------------------------------------------------------
# Stage 3 — empirical curvature on principal axes of production Σθ
# ---------------------------------------------------------------------

println("\n", "─" ^ 72)
println("Stage 3 — empirical curvature along eigenaxes of production Σθ")
println("─" ^ 72)
println("Sample −log π̂(θ̂ + t σ_k v_k) for t ∈ {±0.5, ±1, ±2, ±3}.")
println("Fit 5-point stencil at t = ±2σ_k → empirical d²(−log π̂)/dz².")
println("If empirical curvature > 1/σ_k², production Σθ is wider than the")
println("local posterior actually warrants → grid envelope too generous.")

σs = sqrt.(max.(F_prod.values, 0.0))
for k in 1:length(θ̂)
    println("\nAxis $k  (σ = $(round(σs[k], digits = 4)))")
    v = F_prod.vectors[:, k]
    print("   t (in σ) :")
    for t in (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)
        @printf("  %5.1f", t)
    end
    println()
    print("   −log π̂  :")
    nls = Float64[]
    for t in (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)
        nl = neg_lp(θ̂ .+ σs[k] * t .* v)
        push!(nls, nl)
        if isfinite(nl)
            @printf("  %5.2f", nl)
        else
            print("    Inf")
        end
    end
    println()

    # 5-point stencil at z ∈ {-2σ, -σ, 0, σ, 2σ}, scaled by σ_k.
    # f''(0) ≈ (-1/12 f(-2) + 4/3 f(-1) - 5/2 f(0) + 4/3 f(1) - 1/12 f(2)) / h²
    # Here h = σ_k, applied to the function g(z) = neg_lp(θ̂ + z v).
    f_m2 = nls[2]   # t = -2
    f_m1 = nls[3]
    f_0 = nls[5]
    f_p1 = nls[7]
    f_p2 = nls[8]
    if all(isfinite, (f_m2, f_m1, f_0, f_p1, f_p2))
        d2_emp = (-f_m2 / 12 + 4 * f_m1 / 3 - 5 * f_0 / 2 +
                  4 * f_p1 / 3 - f_p2 / 12) / σs[k]^2
        d2_prod = 1.0 / σs[k]^2
        @printf("   empirical d²(−log π̂)/dz² (5-pt @ ±2σ) = %.4f\n", d2_emp)
        @printf("   production 1/σ² (= eigval of H)        = %.4f\n", d2_prod)
        @printf("   ratio empirical / production           = %.3f\n",
            d2_emp / d2_prod)
    else
        println("   skipped (Inf in stencil)")
    end
end

# ---------------------------------------------------------------------
# Stage 4 — number of Laplace fits per integration sweep
# ---------------------------------------------------------------------

println("\n", "─" ^ 72)
println("Stage 4 — per-fit timing")
println("─" ^ 72)
n_pts = length(res.θ_weights)
println("design points: $n_pts")
@printf("avg time per Laplace fit (total/n_pts): %6.3f s\n", t_total / n_pts)
println("(R-INLA's CCD on dim θ = 2 typically uses ≈ 9 points)")
println("If avg-time-per-fit is comparable to R-INLA, the gap is in *count*,")
println("not in per-fit cost — i.e. tune the integration scheme, not the")
println("Hessian.")

# ---------------------------------------------------------------------
# Stage 5 — alternative integration schemes side-by-side
# ---------------------------------------------------------------------

println("\n", "─" ^ 72)
println("Stage 5 — alternative integration schemes")
println("─" ^ 72)
println("Compare wall-clock + integrated posterior summaries across schemes.")
println("R-INLA's effective design at dim θ = 2 is grid + adaptive density")
println("thresholding (`int.dz_threshold ≈ 6`) → typically ~9 active points.")

# Reference (matched to oracle fixture)
mlik_ref = round(res.log_marginal, digits = 4)
hp_grid = hyperparameters(model, res)
prec_ref = round(hp_grid[1].mean, digits = 3)

println("\nGrid (n_per_dim=5, span=3) — production default")
@printf("  time          : %6.3f s\n", t_total)
@printf("  design points : %d\n", length(res.θ_weights))
@printf("  mlik          : %.4f\n", res.log_marginal)
@printf("  E[τ | y]      : %.3f\n", hp_grid[1].mean)

# CCD (1 center + 4 axial + 4 corners = 9 points at dim=2)
res_ccd_warm = inla(model, y; int_strategy = :ccd)
t_ccd = @elapsed res_ccd = inla(model, y; int_strategy = :ccd)
hp_ccd = hyperparameters(model, res_ccd)
println("\nCCD (f0 = √(m+2) ≈ 2)")
@printf("  time          : %6.3f s   (%.2fx vs grid)\n", t_ccd, t_ccd / t_total)
@printf("  design points : %d\n", length(res_ccd.θ_weights))
@printf("  mlik          : %.4f   (Δ = %+.4f vs grid)\n",
    res_ccd.log_marginal, res_ccd.log_marginal - res.log_marginal)
@printf("  E[τ | y]      : %.3f   (Δ = %+.3f vs grid)\n",
    hp_ccd[1].mean, hp_ccd[1].mean - hp_grid[1].mean)

# Grid with n=3 (9 points, same span)
using LatentGaussianModels: Grid as GridScheme
res_g3_warm = inla(model, y; int_strategy = GridScheme(n_per_dim = 3, span = 3.0))
t_g3 = @elapsed res_g3 = inla(model, y; int_strategy = GridScheme(n_per_dim = 3, span = 3.0))
hp_g3 = hyperparameters(model, res_g3)
println("\nGrid (n_per_dim=3, span=3) — 9 points, same envelope")
@printf("  time          : %6.3f s   (%.2fx vs grid-5)\n", t_g3, t_g3 / t_total)
@printf("  design points : %d\n", length(res_g3.θ_weights))
@printf("  mlik          : %.4f   (Δ = %+.4f vs grid-5)\n",
    res_g3.log_marginal, res_g3.log_marginal - res.log_marginal)
@printf("  E[τ | y]      : %.3f   (Δ = %+.3f vs grid-5)\n",
    hp_g3[1].mean, hp_g3[1].mean - hp_grid[1].mean)

# Grid with n=7 — should be tighter on accuracy but expensive
res_g7_warm = inla(model, y; int_strategy = GridScheme(n_per_dim = 7, span = 3.0))
t_g7 = @elapsed res_g7 = inla(model, y; int_strategy = GridScheme(n_per_dim = 7, span = 3.0))
hp_g7 = hyperparameters(model, res_g7)
println("\nGrid (n_per_dim=7, span=3) — 49 points, finer resolution")
@printf("  time          : %6.3f s   (%.2fx vs grid-5)\n", t_g7, t_g7 / t_total)
@printf("  design points : %d\n", length(res_g7.θ_weights))
@printf("  mlik          : %.4f   (Δ = %+.4f vs grid-5)\n",
    res_g7.log_marginal, res_g7.log_marginal - res.log_marginal)
@printf("  E[τ | y]      : %.3f   (Δ = %+.3f vs grid-5)\n",
    hp_g7[1].mean, hp_g7[1].mean - hp_grid[1].mean)

println("\nR-INLA reference for context: 8.66 s, mlik = -231.183, E[τ|y] = 144.50")

# ---------------------------------------------------------------------
# Stage 6 — split timing: mode/Hessian vs integration sweep
# ---------------------------------------------------------------------

println("\n", "─" ^ 72)
println("Stage 6 — split timing: mode/Hessian vs integration sweep")
println("─" ^ 72)
println("If swap-out of n_per_dim doesn't move the needle on total time,")
println("the cost is in the LBFGS mode-finding + FD Hessian, not the sweep.")
println("Time `_θ_mode_and_hessian` separately from the sweep here.")

# Internal access
using LatentGaussianModels: _θ_mode_and_hessian, _safe_inverse_hessian,
                            integration_nodes, _resolve_scheme,
                            initial_hyperparameters, _neg_log_posterior_θ

# Warm up first
let
    _θ_mode_and_hessian(model, y, INLA(int_strategy = :grid))
end

t_mode = @elapsed (θ̂_t, H_t, _) = _θ_mode_and_hessian(model, y,
    INLA(int_strategy = :grid))
@printf("\n_θ_mode_and_hessian time : %6.3f s\n", t_mode)

Σθ_t = _safe_inverse_hessian(H_t)

# Sweep timing for several point counts (Laplace only, no mode-finding)
for n_per_dim in (3, 5, 7)
    scheme = GridScheme(n_per_dim = n_per_dim, span = 3.0)
    pts, _ = integration_nodes(scheme, θ̂_t, Σθ_t)
    # Warmup
    laplace_mode(model, y, pts[1]; strategy = Laplace())
    t_sweep = @elapsed for p in pts
        laplace_mode(model, y, p; strategy = Laplace())
    end
    @printf("Grid n=%d : %d pts, sweep time = %6.3f s, per-fit = %.4f s\n",
        n_per_dim, length(pts), t_sweep, t_sweep / length(pts))
end

# ---------------------------------------------------------------------
# Stage 7 — what does LBFGS spend its time on?
# ---------------------------------------------------------------------

println("\n", "─" ^ 72)
println("Stage 7 — LBFGS optimization profile")
println("─" ^ 72)
println("`_θ_mode_and_hessian` runs LBFGS with `AutoFiniteDiff()` for the")
println("gradient. At dim θ = 2 each FD-gradient call costs 2-4 fits per")
println("evaluation. Count fits and per-fit time during mode-finding.")

# Wrap neg_lp with a counter to see how many calls LBFGS makes
const FIT_COUNT = Ref(0)
const FIT_TIME = Ref(0.0)

function counted_neg_lp(θ::AbstractVector{<:Real})
    t = @elapsed nl = neg_lp(θ)
    FIT_COUNT[] += 1
    FIT_TIME[] += t
    return nl
end

# Time and count for a single θ near θ̂ first (cold start)
FIT_COUNT[] = 0; FIT_TIME[] = 0.0
counted_neg_lp(θ̂)   # warm
FIT_COUNT[] = 0; FIT_TIME[] = 0.0
t_one_at_mode = @elapsed counted_neg_lp(θ̂)
@printf("\nSingle Laplace at θ̂           : %.4f s\n", t_one_at_mode)

# Same but at the LBFGS initial point
θ_init = initial_hyperparameters(model)
@printf("Initial θ                      : %s\n", string(θ_init))
@printf("θ̂                              : %s\n", string(θ̂_t))
@printf("‖θ_init − θ̂‖                   : %.4f\n", sqrt(sum(abs2,
    θ_init .- θ̂_t)))

FIT_COUNT[] = 0; FIT_TIME[] = 0.0
t_one_at_init = @elapsed counted_neg_lp(θ_init)
@printf("Single Laplace at θ_init       : %.4f s\n", t_one_at_init)

# Check Newton iterations at each
res_at_mode = laplace_mode(model, y, θ̂_t; strategy = Laplace())
res_at_init = laplace_mode(model, y, θ_init; strategy = Laplace())
@printf("Newton iters at θ̂              : %d\n", res_at_mode.iterations)
@printf("Newton iters at θ_init         : %d\n", res_at_init.iterations)

# Now profile a fresh LBFGS run with counter
import Optimization
import OptimizationOptimJL

f_count = function (θ, _p)
    return counted_neg_lp(θ)
end

FIT_COUNT[] = 0; FIT_TIME[] = 0.0
optf = Optimization.OptimizationFunction(f_count, Optimization.AutoFiniteDiff())
prob = Optimization.OptimizationProblem(optf, copy(θ_init), nothing)
t_lbfgs = @elapsed opt_res = Optimization.solve(prob, OptimizationOptimJL.LBFGS())
@printf("\nLBFGS (counted)                : %.3f s\n", t_lbfgs)
@printf("# Laplace fits during LBFGS    : %d\n", FIT_COUNT[])
@printf("Σ time inside Laplace fits     : %.3f s (%.0f%% of LBFGS)\n",
    FIT_TIME[], 100 * FIT_TIME[] / t_lbfgs)
@printf("Avg fit time during LBFGS      : %.4f s\n",
    FIT_TIME[] / FIT_COUNT[])
@printf("LBFGS converged θ̂              : %s\n", string(opt_res.u))
@printf("LBFGS iterations               : %s\n", string(opt_res.stats))

# ---------------------------------------------------------------------
# Stage 8 — does loosening g_tol solve it?
# ---------------------------------------------------------------------

println("\n", "─" ^ 72)
println("Stage 8 — LBFGS g_tol scan")
println("─" ^ 72)
println("Hypothesis: default g_tol = 1e-8 is at the FD-gradient noise floor")
println("for AutoFiniteDiff at dim θ = 2; LBFGS hits the iteration limit")
println("before converging the gradient norm. Test progressively looser tols.")

θ̂_ref = θ̂_t
for g_tol in (1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8)
    FIT_COUNT[] = 0; FIT_TIME[] = 0.0
    optf = Optimization.OptimizationFunction(f_count, Optimization.AutoFiniteDiff())
    prob = Optimization.OptimizationProblem(optf, copy(θ_init), nothing)
    t = @elapsed local opt = Optimization.solve(prob,
        OptimizationOptimJL.LBFGS(); g_tol = g_tol)
    Δ = sqrt(sum(abs2, opt.u .- θ̂_ref))
    @printf("g_tol = %.0e : %6.3f s, %5d fits, ‖θ̂ − θ̂_ref‖ = %.2e\n",
        g_tol, t, FIT_COUNT[], Δ)
end

println("\n", "=" ^ 72)
println("Done.")
println("=" ^ 72)

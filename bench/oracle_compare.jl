# bench/oracle_compare.jl
#
# End-to-end Julia-INLA benchmark harness over the R-INLA oracle suite.
#
# For every oracle problem (10 LGM + 1 INLASPDE Meuse) the harness:
#   1. Builds the Julia model (mirroring the corresponding test_*.jl).
#   2. Times `inla(model, y; int_strategy = :grid)` (one warmup, one timed
#      run), and a faster `empirical_bayes(model, y)` Laplace-only run for
#      comparison.
#   3. Pulls the corresponding R-INLA posterior summaries from the JLD2
#      fixture and computes per-quantity relative errors against Julia.
#   4. Prints a markdown table to stdout.
#   5. Writes a JSON file `bench/oracle_compare_julia.json` containing the
#      Julia summaries, R fixture summaries, deltas, and timings.
#
# Missing fixtures are skipped with a warning. A Julia run that errors is
# caught — its error string is recorded in the JSON, the harness keeps
# going.
#
# Invocation (from the repo root):
#
#   julia --project=bench bench/oracle_compare.jl
#
# `bench/Project.toml` `Pkg.develop`s the three ecosystem packages used
# by the harness (`GMRFs`, `LatentGaussianModels`, `INLASPDE`).
# `bench/Manifest.toml` is gitignored. See `bench/README.md` for column
# meanings, expected runtime, and the JSON schema.
#
# No JSON dependency is used; the JSON is hand-written. The harness uses
# only what is already available under the bench project plus stdlib
# (JLD2, SparseArrays, LinearAlgebra, Dates).

using JLD2
using SparseArrays
using LinearAlgebra: I
using Dates: now, UTC

using LatentGaussianModels:
    GaussianLikelihood, PoissonLikelihood, NegativeBinomialLikelihood,
    GammaLikelihood,
    Intercept, FixedEffects, Seasonal, Besag, BYM, BYM2, Leroux,
    Generic0, Generic1,
    PCPrecision, GammaPrecision, LogitBeta,
    LatentGaussianModel, inla, empirical_bayes,
    fixed_effects, hyperparameters, log_marginal_likelihood
using GMRFs: GMRFGraph

using INLASPDE: SPDE2, PCMatern, spde_user_scale

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const LGM_FIXTURE_DIR =
    joinpath(REPO_ROOT, "packages", "LatentGaussianModels.jl",
             "test", "oracle", "fixtures")
const SPDE_FIXTURE_DIR =
    joinpath(REPO_ROOT, "packages", "INLASPDE.jl",
             "test", "oracle", "fixtures")
const OUTPUT_JSON = joinpath(@__DIR__, "oracle_compare_julia.json")
const OUTPUT_MD   = joinpath(@__DIR__, "oracle_compare_julia.md")

# ---------------------------------------------------------------------
# Fixture utilities
# ---------------------------------------------------------------------

function load_fixture(path::AbstractString)
    isfile(path) || return nothing
    return jldopen(path, "r") do f
        return f["fixture"]
    end
end

function _row_value(frame, name::AbstractString, col::AbstractString)
    rn_raw = frame["rownames"]
    rn = rn_raw isa AbstractString ? [String(rn_raw)] : String.(rn_raw)
    idx = findfirst(==(name), rn)
    idx === nothing && return nothing
    col_raw = frame[col]
    return col_raw isa Real ? Float64(col_raw) : Float64(col_raw[idx])
end

# Relative error helpers. Fixed effects scale by max(|r|, 1) — guards
# against blowups when the reference is near zero. Hyperparameters and
# mlik are pure relative.
_rel_fixed(j, r) = abs(j - r) / max(abs(r), 1.0)
_rel_hyper(j, r) = abs(j - r) / abs(r)

# ---------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------

# Each problem returns a NamedTuple:
#   (model, y, n,
#    fixed_pairs   :: Vector{Tuple{Int, String}},        # (julia_fe_idx, R_rowname)
#    hyper_pairs   :: Vector{Tuple{Function, String}})   # (θ̂ -> user-scale, R_rowname)
#
# `fixed_pairs[i]` says: Julia's `fixed_effects(model, res)[idx].mean` should
# be compared against R's `summary_fixed[rowname, "mean"]`.
# `hyper_pairs[i]` says: `transform(res.θ̂)` should be compared against R's
# `summary_hyperpar[rowname, "mean"]`. The transform takes the full θ̂ so
# multi-component user-scale hyperparameters (e.g. SPDE range/sigma) can
# be expressed.

function build_scotland_bym2(inp)
    y = Int.(inp["cases"])
    E = Float64.(inp["expected"])
    x = Float64.(inp["x"])
    W = inp["W"]
    n = length(y)
    ℓ = PoissonLikelihood(; E = E)
    c_int  = Intercept()
    c_beta = FixedEffects(1)
    c_bym2 = BYM2(GMRFGraph(W); hyperprior_prec = PCPrecision(1.0, 0.01))
    A = sparse(hcat(
        ones(n),
        reshape(x, n, 1),
        Matrix{Float64}(I, n, n),
        zeros(n, n),
    ))
    model = LatentGaussianModel(ℓ, (c_int, c_beta, c_bym2), A)
    fixed_pairs = [(1, "(Intercept)"), (2, "x")]
    hyper_pairs = [(θ -> exp(θ[1]), "Precision for region")]
    return (; model, y, n, fixed_pairs, hyper_pairs)
end

function build_scotland_bym(inp)
    y = Int.(inp["cases"])
    E = Float64.(inp["expected"])
    x = Float64.(inp["x"])
    W = inp["W"]
    n = length(y)
    ℓ = PoissonLikelihood(; E = E)
    c_int  = Intercept()
    c_beta = FixedEffects(1)
    c_bym  = BYM(GMRFGraph(W);
                 hyperprior_iid   = PCPrecision(1.0, 0.01),
                 hyperprior_besag = PCPrecision(1.0, 0.01))
    A = sparse(hcat(
        ones(n),
        reshape(x, n, 1),
        Matrix{Float64}(I, n, n),
        Matrix{Float64}(I, n, n),
    ))
    model = LatentGaussianModel(ℓ, (c_int, c_beta, c_bym), A)
    fixed_pairs = [(1, "(Intercept)"), (2, "x")]
    hyper_pairs = [
        (θ -> exp(θ[1]), "Precision for region (iid component)"),
        (θ -> exp(θ[2]), "Precision for region (spatial component)"),
    ]
    return (; model, y, n, fixed_pairs, hyper_pairs)
end

function build_pennsylvania_bym2(inp)
    y = Int.(inp["cases"])
    E = Float64.(inp["expected"])
    x = Float64.(inp["x"])
    W = inp["W"]
    n = length(y)
    ℓ = PoissonLikelihood(; E = E)
    c_int  = Intercept()
    c_beta = FixedEffects(1)
    c_bym2 = BYM2(GMRFGraph(W); hyperprior_prec = PCPrecision(1.0, 0.01))
    A = sparse(hcat(
        ones(n),
        reshape(x, n, 1),
        Matrix{Float64}(I, n, n),
        zeros(n, n),
    ))
    model = LatentGaussianModel(ℓ, (c_int, c_beta, c_bym2), A)
    fixed_pairs = [(1, "(Intercept)"), (2, "x")]
    hyper_pairs = [(θ -> exp(θ[1]), "Precision for region")]
    return (; model, y, n, fixed_pairs, hyper_pairs)
end

function build_synthetic_gamma(inp)
    y = Float64.(inp["y"])
    x = Float64.(inp["x"])
    n = length(y)
    ℓ = GammaLikelihood()
    c_int  = Intercept()
    c_beta = FixedEffects(1)
    A = sparse(hcat(ones(n), reshape(x, n, 1)))
    model = LatentGaussianModel(ℓ, (c_int, c_beta), A)
    fixed_pairs = [(1, "(Intercept)"), (2, "x")]
    hyper_pairs = [
        (θ -> exp(θ[1]), "Precision-parameter for the Gamma observations"),
    ]
    return (; model, y, n, fixed_pairs, hyper_pairs)
end

function build_synthetic_seasonal(inp)
    n      = Int(inp["n"])
    period = Int(inp["period"])
    y      = Float64.(inp["y"])
    ℓ = GaussianLikelihood(hyperprior = GammaPrecision(1.0, 5.0e-5))
    α = Intercept()
    seas = Seasonal(n; period = period,
                    hyperprior = GammaPrecision(1.0, 5.0e-5))
    A_α    = sparse(ones(n, 1))
    A_seas = sparse(I, n, n)
    A      = hcat(A_α, A_seas)
    model = LatentGaussianModel(ℓ, (α, seas), A)
    fixed_pairs = [(1, "(Intercept)")]
    hyper_pairs = [
        (θ -> exp(θ[1]), "Precision for the Gaussian observations"),
        (θ -> exp(θ[2]), "Precision for t"),
    ]
    return (; model, y, n, fixed_pairs, hyper_pairs)
end

function build_synthetic_generic0(inp)
    n_obs = Int(inp["n_obs"])
    n_lat = Int(inp["n_lat"])
    y     = Float64.(inp["y"])
    A_vec = Float64.(inp["A"])
    A_dense = reshape(A_vec, n_lat, n_obs)'
    A = sparse(Matrix{Float64}(A_dense))
    C = SparseMatrixCSC{Float64, Int}(inp["C"])
    ℓ = GaussianLikelihood(hyperprior = GammaPrecision(1.0, 5.0e-5))
    c_g0 = Generic0(C; rankdef = 0,
                    hyperprior = GammaPrecision(1.0, 5.0e-5))
    model = LatentGaussianModel(ℓ, (c_g0,), A)
    fixed_pairs = Tuple{Int, String}[]
    hyper_pairs = [
        (θ -> exp(θ[1]), "Precision for the Gaussian observations"),
        (θ -> exp(θ[2]), "Precision for idx"),
    ]
    return (; model, y, n = n_obs, fixed_pairs, hyper_pairs)
end

function build_synthetic_generic1(inp)
    n_obs = Int(inp["n_obs"])
    n_lat = Int(inp["n_lat"])
    y     = Float64.(inp["y"])
    A_vec = Float64.(inp["A"])
    A_dense = reshape(A_vec, n_lat, n_obs)'
    A = sparse(Matrix{Float64}(A_dense))
    C = SparseMatrixCSC{Float64, Int}(inp["C"])
    ℓ = GaussianLikelihood(hyperprior = GammaPrecision(1.0, 5.0e-5))
    c_g1 = Generic1(C; rankdef = 0,
                    hyperprior = GammaPrecision(1.0, 5.0e-5))
    model = LatentGaussianModel(ℓ, (c_g1,), A)
    fixed_pairs = Tuple{Int, String}[]
    hyper_pairs = [
        (θ -> exp(θ[1]), "Precision for the Gaussian observations"),
        (θ -> exp(θ[2]), "Precision for idx"),
    ]
    return (; model, y, n = n_obs, fixed_pairs, hyper_pairs)
end

function build_synthetic_leroux(inp)
    n      = Int(inp["n"])
    n_obs  = Int(inp["n_obs"])
    y      = Float64.(inp["y"])
    region = Int.(inp["region"])
    W      = SparseMatrixCSC{Float64, Int}(inp["W"])
    ℓ = GaussianLikelihood(hyperprior = GammaPrecision(1.0, 5.0e-5))
    α = Intercept()
    lrx = Leroux(GMRFGraph(W);
                 hyperprior_tau = PCPrecision(1.0, 0.01),
                 hyperprior_rho = LogitBeta(1.0, 1.0))
    rows_α    = collect(1:n_obs)
    cols_α    = ones(Int, n_obs)
    A_α       = sparse(rows_α, cols_α, ones(Float64, n_obs), n_obs, 1)
    rows_lrx  = collect(1:n_obs)
    cols_lrx  = region
    A_lrx     = sparse(rows_lrx, cols_lrx, ones(Float64, n_obs), n_obs, n)
    A         = hcat(A_α, A_lrx)
    model = LatentGaussianModel(ℓ, (α, lrx), A)
    fixed_pairs = [(1, "(Intercept)")]
    hyper_pairs = [
        (θ -> exp(θ[1]), "Precision for the Gaussian observations"),
        (θ -> exp(θ[2]), "Precision for region"),
        (θ -> inv(1 + exp(-θ[3])), "Lambda for region"),
    ]
    return (; model, y, n = n_obs, fixed_pairs, hyper_pairs)
end

function build_synthetic_nbinomial(inp)
    y = Int.(inp["y"])
    x = Float64.(inp["x"])
    n = length(y)
    ℓ = NegativeBinomialLikelihood()
    c_int  = Intercept()
    c_beta = FixedEffects(1)
    A = sparse(hcat(ones(n), reshape(x, n, 1)))
    model = LatentGaussianModel(ℓ, (c_int, c_beta), A)
    fixed_pairs = [(1, "(Intercept)"), (2, "x")]
    hyper_pairs = [
        (θ -> exp(θ[1]),
         "size for the nbinomial observations (1/overdispersion)"),
    ]
    return (; model, y, n, fixed_pairs, hyper_pairs)
end

function build_synthetic_disconnected_besag(inp)
    y = Int.(inp["y"])
    W = inp["W"]
    n = length(y)
    graph = GMRFGraph(W)
    ℓ = PoissonLikelihood()
    c_int = Intercept()
    c_b   = Besag(graph; hyperprior = PCPrecision(1.0, 0.01))
    A = sparse(hcat(ones(n), Matrix{Float64}(I, n, n)))
    model = LatentGaussianModel(ℓ, (c_int, c_b), A)
    fixed_pairs = [(1, "(Intercept)")]
    # No tight hyperparameter target — τ is heavy-tailed at n=12; the
    # R-side row is labelled "Precision for region" because the Besag is
    # built over the regional graph. Reported for diagnostics only.
    hyper_pairs = [(θ -> exp(θ[1]), "Precision for region")]
    return (; model, y, n, fixed_pairs, hyper_pairs)
end

function build_meuse_spde(fxt)
    inp      = fxt["input"]
    y        = Float64.(inp["y"])
    dist_cov = Float64.(inp["dist"])
    points   = fxt["mesh"]["loc"]::Matrix{Float64}
    tv       = fxt["mesh"]["tv"]::Matrix{Int}
    A_field  = SparseMatrixCSC{Float64, Int}(fxt["A_field"])
    n_obs = length(y)
    spde = SPDE2(points, tv; α = 2,
        pc = PCMatern(
            range_U = 0.5, range_α = 0.5,
            sigma_U = 1.0, sigma_α = 0.5,
        ))
    intercept = Intercept(prec = 1.0e-3)
    beta_dist = FixedEffects(1; prec = 1.0e-3)
    A_intercept = ones(n_obs, 1)
    A_dist      = reshape(dist_cov, n_obs, 1)
    A = hcat(A_intercept, A_dist, A_field)
    like  = GaussianLikelihood(hyperprior = PCPrecision(1.0, 0.01))
    model = LatentGaussianModel(like, (intercept, beta_dist, spde), A)
    fixed_pairs = [(1, "intercept"), (2, "dist")]
    # Closure over `spde` for the user-scale conversion.
    range_fn(θ) = spde_user_scale(spde, θ[2:3])[1]
    sigma_fn(θ) = spde_user_scale(spde, θ[2:3])[2]
    hyper_pairs = [
        (θ -> exp(θ[1]), "Precision for the Gaussian observations"),
        (range_fn,       "Range for field"),
        (sigma_fn,       "Stdev for field"),
    ]
    return (; model, y, n = n_obs, fixed_pairs, hyper_pairs)
end

# Registry — order matches the user-facing problem list. Each entry pins
# the fixture path, the builder, and the inla() integration strategy
# (mirrors the matching test). The `int_strategy` is `:grid` for every
# LGM oracle and `:auto` (the default) for Meuse SPDE — match the test.
const PROBLEMS = [
    (name = "scotland_bym2",
     fixture_path = joinpath(LGM_FIXTURE_DIR, "scotland_bym2.jld2"),
     builder = (fx) -> build_scotland_bym2(fx["input"]),
     int_strategy = :grid),
    (name = "scotland_bym",
     fixture_path = joinpath(LGM_FIXTURE_DIR, "scotland_bym.jld2"),
     builder = (fx) -> build_scotland_bym(fx["input"]),
     int_strategy = :grid),
    (name = "pennsylvania_bym2",
     fixture_path = joinpath(LGM_FIXTURE_DIR, "pennsylvania_bym2.jld2"),
     builder = (fx) -> build_pennsylvania_bym2(fx["input"]),
     int_strategy = :grid),
    (name = "synthetic_gamma",
     fixture_path = joinpath(LGM_FIXTURE_DIR, "synthetic_gamma.jld2"),
     builder = (fx) -> build_synthetic_gamma(fx["input"]),
     int_strategy = :grid),
    (name = "synthetic_seasonal",
     fixture_path = joinpath(LGM_FIXTURE_DIR, "synthetic_seasonal.jld2"),
     builder = (fx) -> build_synthetic_seasonal(fx["input"]),
     int_strategy = :grid),
    (name = "synthetic_generic0",
     fixture_path = joinpath(LGM_FIXTURE_DIR, "synthetic_generic0.jld2"),
     builder = (fx) -> build_synthetic_generic0(fx["input"]),
     int_strategy = :grid),
    (name = "synthetic_generic1",
     fixture_path = joinpath(LGM_FIXTURE_DIR, "synthetic_generic1.jld2"),
     builder = (fx) -> build_synthetic_generic1(fx["input"]),
     int_strategy = :grid),
    (name = "synthetic_leroux",
     fixture_path = joinpath(LGM_FIXTURE_DIR, "synthetic_leroux.jld2"),
     builder = (fx) -> build_synthetic_leroux(fx["input"]),
     int_strategy = :grid),
    (name = "synthetic_nbinomial",
     fixture_path = joinpath(LGM_FIXTURE_DIR, "synthetic_nbinomial.jld2"),
     builder = (fx) -> build_synthetic_nbinomial(fx["input"]),
     int_strategy = :grid),
    (name = "synthetic_disconnected_besag",
     fixture_path = joinpath(LGM_FIXTURE_DIR,
                             "synthetic_disconnected_besag.jld2"),
     builder = (fx) -> build_synthetic_disconnected_besag(fx["input"]),
     int_strategy = :grid),
    (name = "meuse_spde",
     fixture_path = joinpath(SPDE_FIXTURE_DIR, "meuse_spde.jld2"),
     builder = (fx) -> build_meuse_spde(fx),
     int_strategy = :auto),
]

# ---------------------------------------------------------------------
# Per-problem run
# ---------------------------------------------------------------------

# Result schema (one entry per problem):
#
#   status      :: "ok" | "skipped" | "error"
#   error       :: String      (only present for "error")
#   reason      :: String      (only present for "skipped")
#   n           :: Int
#   timings     :: NamedTuple{(:inla_warmup, :inla, :empirical_bayes), 3*Float64}
#   julia       :: NamedTuple{(:fixed, :hyperpar, :mlik), ...}
#   r           :: NamedTuple{(:fixed, :hyperpar, :mlik), ...}
#   deltas      :: NamedTuple{(:fixed, :hyperpar, :mlik_abs, :mlik_rel,
#                              :fixed_max_rel, :hyperpar_max_rel), ...}
#
# `fixed` is a Vector of (name, julia, r) NamedTuples; `hyperpar` likewise
# but `r_rowname` instead of `name`. Same shape for the Julia-side.

function run_problem(p::NamedTuple)
    fx = load_fixture(p.fixture_path)
    if fx === nothing
        @warn "fixture missing — skipping $(p.name): $(p.fixture_path)"
        return (
            problem = p.name,
            status  = "skipped",
            reason  = "fixture not found at $(p.fixture_path)",
        )
    end

    local prob
    try
        prob = p.builder(fx)
    catch err
        msg = sprint(showerror, err)
        @warn "model build failed for $(p.name): $msg"
        return (
            problem = p.name,
            status  = "error",
            error   = "build: $msg",
        )
    end

    model, y, n = prob.model, prob.y, prob.n
    fixed_pairs, hyper_pairs = prob.fixed_pairs, prob.hyper_pairs
    int_strategy = p.int_strategy

    # Run twice: warmup, then timed. Both inside try/catch so a numerical
    # blow-up on one problem does not abort the whole harness.
    # Note: `@elapsed expr` opens a `let` scope, so any assignments need
    # to flow back through the macro return. We use `(t = @elapsed (...);
    # value)` form by calling the work in a helper that returns the
    # result alongside the elapsed time.
    t_warmup = NaN
    t_inla   = NaN
    t_eb     = NaN
    res      = nothing
    try
        # Warmup discarded.
        t_warmup = @elapsed inla(model, y; int_strategy = int_strategy)
        # Timed run — capture into outer-scope `res` via Ref.
        ref = Ref{Any}(nothing)
        t_inla = @elapsed begin
            ref[] = inla(model, y; int_strategy = int_strategy)
        end
        res = ref[]
    catch err
        msg = sprint(showerror, err)
        @warn "inla failed for $(p.name): $msg"
        return (
            problem = p.name,
            status  = "error",
            error   = "inla: $msg",
            n       = n,
            timings = (inla_warmup = t_warmup, inla = t_inla, empirical_bayes = t_eb),
        )
    end

    # Empirical-Bayes (Laplace at θ̂) — the cheaper comparison point.
    # Wrap defensively: failure here does not invalidate the INLA run.
    eb_log_marginal = NaN
    try
        ref_eb = Ref{Any}(nothing)
        t_eb = @elapsed begin
            ref_eb[] = empirical_bayes(model, y)
        end
        eb_res = ref_eb[]
        eb_log_marginal = eb_res === nothing ? NaN : eb_res.log_marginal
    catch err
        msg = sprint(showerror, err)
        @warn "empirical_bayes failed for $(p.name): $msg"
        eb_log_marginal = NaN
    end

    # ---- Julia summaries -----------------------------------------------
    fe = fixed_effects(model, res)
    julia_fixed = NamedTuple{(:name, :mean, :sd), Tuple{String, Float64, Float64}}[]
    for (idx, _) in fixed_pairs
        if idx <= length(fe)
            push!(julia_fixed,
                  (name = fe[idx].name, mean = fe[idx].mean, sd = fe[idx].sd))
        end
    end
    julia_hyperpar = NamedTuple{(:rowname, :mean), Tuple{String, Float64}}[]
    for (transform, rowname) in hyper_pairs
        push!(julia_hyperpar,
              (rowname = rowname, mean = Float64(transform(res.θ̂))))
    end
    julia_mlik = log_marginal_likelihood(res)

    # ---- R fixture summaries -------------------------------------------
    sf = fx["summary_fixed"]
    sh = fx["summary_hyperpar"]
    r_fixed = NamedTuple{(:rowname, :mean, :sd),
                         Tuple{String, Float64, Float64}}[]
    for (_, rowname) in fixed_pairs
        m = _row_value(sf, rowname, "mean")
        s = _row_value(sf, rowname, "sd")
        push!(r_fixed, (rowname = rowname,
                        mean = m === nothing ? NaN : m,
                        sd   = s === nothing ? NaN : s))
    end
    r_hyperpar = NamedTuple{(:rowname, :mean), Tuple{String, Float64}}[]
    for (_, rowname) in hyper_pairs
        m = _row_value(sh, rowname, "mean")
        push!(r_hyperpar,
              (rowname = rowname, mean = m === nothing ? NaN : m))
    end
    r_mlik = haskey(fx, "mlik") ? Float64(fx["mlik"][1]) : NaN

    # R-INLA elapsed wall-time. `cpu.used` is c(user, system, child-user,
    # child-system) — the 4-th element is total elapsed (`elapsed`).
    r_inla_elapsed = NaN
    if haskey(fx, "cpu_used")
        cu = fx["cpu_used"]
        if cu isa AbstractVector && length(cu) >= 4
            r_inla_elapsed = Float64(cu[4])
        end
    end

    # ---- Deltas --------------------------------------------------------
    fe_rels = Float64[]
    fixed_deltas = NamedTuple{(:name, :rel),
                              Tuple{String, Float64}}[]
    for i in eachindex(julia_fixed)
        rel = _rel_fixed(julia_fixed[i].mean, r_fixed[i].mean)
        push!(fe_rels, rel)
        push!(fixed_deltas, (name = julia_fixed[i].name, rel = rel))
    end
    hp_rels = Float64[]
    hyper_deltas = NamedTuple{(:rowname, :rel),
                              Tuple{String, Float64}}[]
    for i in eachindex(julia_hyperpar)
        rel = _rel_hyper(julia_hyperpar[i].mean, r_hyperpar[i].mean)
        push!(hp_rels, rel)
        push!(hyper_deltas,
              (rowname = julia_hyperpar[i].rowname, rel = rel))
    end
    mlik_abs = abs(julia_mlik - r_mlik)
    mlik_rel = isfinite(r_mlik) && r_mlik != 0.0 ?
               mlik_abs / abs(r_mlik) : NaN
    fixed_max_rel = isempty(fe_rels) ? NaN : maximum(fe_rels)
    hyper_max_rel = isempty(hp_rels) ? NaN : maximum(hp_rels)

    return (
        problem = p.name,
        status  = "ok",
        n       = n,
        timings = (inla_warmup = t_warmup,
                   inla        = t_inla,
                   empirical_bayes = t_eb,
                   r_inla       = r_inla_elapsed,
                   speedup_vs_r = isfinite(r_inla_elapsed) && t_inla > 0 ?
                                  r_inla_elapsed / t_inla : NaN),
        julia = (fixed = julia_fixed,
                 hyperpar = julia_hyperpar,
                 mlik = julia_mlik,
                 empirical_bayes_log_marginal = eb_log_marginal),
        r = (fixed = r_fixed,
             hyperpar = r_hyperpar,
             mlik = r_mlik),
        deltas = (fixed = fixed_deltas,
                  hyperpar = hyper_deltas,
                  mlik_abs = mlik_abs,
                  mlik_rel = mlik_rel,
                  fixed_max_rel = fixed_max_rel,
                  hyperpar_max_rel = hyper_max_rel),
    )
end

# ---------------------------------------------------------------------
# Markdown table output
# ---------------------------------------------------------------------

# Format a maybe-NaN float for the markdown summary table. The fallback
# string keeps the column widths even and makes "no data" obvious.
_fmt(x; sigdigits = 4) = (x === nothing || !isfinite(x)) ?
                         "—" : string(round(x; sigdigits = sigdigits))

function print_markdown_table(io::IO, results)
    println(io, "## Quality (relative error vs R-INLA fixture)")
    println(io)
    println(io, "| problem | n | fixed_max_rel | hyperpar_max_rel | mlik_rel | mlik_abs |")
    println(io, "|---|---:|---:|---:|---:|---:|")
    for r in results
        if r.status == "skipped"
            println(io, "| $(r.problem) | — | — | — | — | _skipped_ |")
            continue
        elseif r.status == "error"
            err = haskey(r, :error) ? r.error : ""
            println(io, "| $(r.problem) | $(get(r, :n, "—")) | — | — | — | _error: $(first(err, 40))_ |")
            continue
        end
        println(io, "| $(r.problem) | $(r.n) | $(_fmt(r.deltas.fixed_max_rel)) | $(_fmt(r.deltas.hyperpar_max_rel)) | $(_fmt(r.deltas.mlik_rel)) | $(_fmt(r.deltas.mlik_abs)) |")
    end

    println(io)
    println(io, "## Performance (wall-clock seconds, single thread)")
    println(io)
    println(io, "| problem | n | julia_inla_s | julia_eb_s | r_inla_s | speedup×_vs_r |")
    println(io, "|---|---:|---:|---:|---:|---:|")
    for r in results
        if r.status != "ok"
            println(io, "| $(r.problem) | — | — | — | — | — |")
            continue
        end
        println(io, "| $(r.problem) | $(r.n) | $(_fmt(r.timings.inla)) | $(_fmt(r.timings.empirical_bayes)) | $(_fmt(r.timings.r_inla)) | $(_fmt(r.timings.speedup_vs_r; sigdigits=3)) |")
    end
end

# ---------------------------------------------------------------------
# Hand-written JSON encoder
# ---------------------------------------------------------------------
#
# We avoid pulling in JSON3 / JSON. The encoder handles only the types
# we emit: Nothing, Bool, Int, Float64, String, Vector, NamedTuple, and
# Dict{String, Any}. Float NaN and ±Inf are emitted as `null` so the
# output is strict JSON.

_json_escape(s::AbstractString) =
    replace(String(s),
            '\\' => "\\\\",
            '"' => "\\\"",
            '\n' => "\\n",
            '\r' => "\\r",
            '\t' => "\\t")

function _json_value(io::IO, x)
    if x === nothing
        print(io, "null")
    elseif x isa Bool
        print(io, x ? "true" : "false")
    elseif x isa Integer
        print(io, x)
    elseif x isa AbstractFloat
        if isnan(x) || isinf(x)
            print(io, "null")
        else
            print(io, x)
        end
    elseif x isa Symbol
        print(io, "\"", _json_escape(String(x)), "\"")
    elseif x isa AbstractString
        print(io, "\"", _json_escape(x), "\"")
    elseif x isa AbstractVector
        _json_array(io, x)
    elseif x isa Tuple
        _json_array(io, collect(x))
    elseif x isa NamedTuple
        _json_object(io, pairs(x))
    elseif x isa AbstractDict
        _json_object(io, x)
    else
        # Fallback: stringify anything we don't recognise.
        print(io, "\"", _json_escape(string(x)), "\"")
    end
    return nothing
end

function _json_array(io::IO, xs)
    print(io, "[")
    first = true
    for x in xs
        first || print(io, ",")
        first = false
        _json_value(io, x)
    end
    print(io, "]")
    return nothing
end

function _json_object(io::IO, pairs_iter)
    print(io, "{")
    first = true
    for (k, v) in pairs_iter
        first || print(io, ",")
        first = false
        print(io, "\"", _json_escape(string(k)), "\":")
        _json_value(io, v)
    end
    print(io, "}")
    return nothing
end

write_json(path::AbstractString, x) = open(path, "w") do io
    _json_value(io, x)
    print(io, "\n")
end

# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------

function main()
    println(stderr, "[oracle_compare] running ", length(PROBLEMS),
            " problems …")
    results = Any[]
    for p in PROBLEMS
        println(stderr, "[oracle_compare] -> ", p.name)
        r = run_problem(p)
        push!(results, r)
    end

    println()
    println("# Oracle compare — Julia INLA vs R-INLA fixtures")
    println()
    println("Generated by `bench/oracle_compare.jl`. Tolerances are documented")
    println("per-problem in the matching `test/oracle/test_*.jl` file.")
    println()
    print_markdown_table(stdout, results)

    write_json(OUTPUT_JSON,
               (generated_at = string(now_utc()),
                problems     = results))
    open(OUTPUT_MD, "w") do io
        println(io, "*Auto-generated by `bench/oracle_compare.jl`. ",
                "Last refreshed: ", string(now_utc()), " UTC.*")
        println(io)
        print_markdown_table(io, results)
    end

    println()
    println("# JSON written to: ", OUTPUT_JSON)
    println("# Markdown fragment written to: ", OUTPUT_MD)
    return results
end

# UTC timestamp for the JSON header — `Dates` is in the stdlib.
now_utc() = string(now(UTC))

main()

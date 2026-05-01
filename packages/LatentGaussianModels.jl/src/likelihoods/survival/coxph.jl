"""
    CoxphAugmented(y, E, subject, interval, breakpoints, n_subjects, n_intervals)

Result of [`inla_coxph`](@ref) data augmentation. Contains the per-row
augmented Poisson response, exposure offsets, subject and baseline-interval
indices, the breakpoints used, and the original subject / interval counts.

Pass `(y, E)` to a [`PoissonLikelihood`](@ref) (`E` as the offset), and use
[`coxph_design`](@ref) to build the joint design matrix that pairs the
piecewise-constant baseline-hazard component with the original covariates.
"""
struct CoxphAugmented
    y::Vector{Int}
    E::Vector{Float64}
    subject::Vector{Int}
    interval::Vector{Int}
    breakpoints::Vector{Float64}
    n_subjects::Int
    n_intervals::Int
end

function Base.show(io::IO, aug::CoxphAugmented)
    print(io, "CoxphAugmented(n_subjects=", aug.n_subjects,
        ", n_intervals=", aug.n_intervals,
        ", n_rows=", length(aug.y), ")")
end

"""
    inla_coxph(time, event;
               breakpoints = nothing, nbreakpoints = 15) -> CoxphAugmented

Cox proportional-hazards data augmentation, mirroring R-INLA's
`inla.coxph`. Each subject is expanded into one row per baseline-hazard
interval `[τ_k, τ_{k+1})` they survive into; the augmented response
becomes a binary indicator and the exposure offset becomes the time
spent in that interval. The augmented data are then suitable for a
Poisson regression with linear predictor

    η_ik = γ_k + x_iᵀ β

where `γ_k` is the piecewise-constant baseline log-hazard for interval
`k` (typically given an [`RW1`](@ref) prior). The piecewise-exponential
likelihood for time-to-event under right-censoring is recovered exactly
by this Poisson augmentation (Holford 1980; Laird & Olivier 1981;
Clayton 1991).

# Arguments

- `time::AbstractVector{<:Real}` — observation times (`>0`).
- `event::AbstractVector{<:Integer}` — event indicators
  (`1` = uncensored event, `0` = right-censored).

# Keyword arguments

- `breakpoints::AbstractVector{<:Real}` — explicit interval boundaries.
  Must be sorted, start at `0`, and end at or above `maximum(time)`.
  When `nothing` (default), quantile-based breakpoints over the
  observed event times are constructed (see `nbreakpoints`).
- `nbreakpoints::Integer = 15` — number of interior quantile-based
  breakpoints when `breakpoints` is not given.

# Example

```julia
aug = inla_coxph(time, event)

ℓ           = PoissonLikelihood(E = aug.E)
c_baseline  = RW1(aug.n_intervals; hyperprior = PCPrecision(1.0, 0.01))
c_beta      = FixedEffects(size(X, 2))
A           = coxph_design(aug, X)

model = LatentGaussianModel(ℓ, (c_baseline, c_beta), A)
res   = inla(model, aug.y)
```
"""
function inla_coxph(time::AbstractVector{<:Real},
        event::AbstractVector{<:Integer};
        breakpoints::Union{Nothing, AbstractVector{<:Real}}=nothing,
        nbreakpoints::Integer=15)
    n = length(time)
    n > 0 || throw(ArgumentError("time must be non-empty"))
    length(event) == n ||
        throw(DimensionMismatch("time and event must have the same length"))
    all(>(0), time) ||
        throw(ArgumentError("time must be strictly positive"))
    all(in((0, 1)), event) ||
        throw(ArgumentError("event must be 0 (censored) or 1 (event)"))
    nbreakpoints ≥ 2 ||
        throw(ArgumentError("nbreakpoints must be ≥ 2"))

    bp::Vector{Float64} = breakpoints === nothing ?
                          _default_coxph_breakpoints(time, event, Int(nbreakpoints)) :
                          collect(Float64, breakpoints)

    length(bp) ≥ 2 ||
        throw(ArgumentError("breakpoints must have length ≥ 2"))
    issorted(bp) ||
        throw(ArgumentError("breakpoints must be sorted"))
    bp[1] == 0 ||
        throw(ArgumentError("breakpoints[1] must equal 0; got $(bp[1])"))
    bp[end] ≥ maximum(time) ||
        throw(ArgumentError(
            "breakpoints[end]=$(bp[end]) must be ≥ maximum(time)=$(maximum(time))"))

    K = length(bp) - 1

    # Pre-size with an upper bound on rows: every subject contributes at
    # most K rows, but typically far fewer. Using `sizehint!` keeps the
    # push! loop allocation-light without overshooting too much.
    y_aug = Int[]
    sizehint!(y_aug, n * 2)
    E_aug = Float64[]
    sizehint!(E_aug, n * 2)
    subj_aug = Int[]
    sizehint!(subj_aug, n * 2)
    intv_aug = Int[]
    sizehint!(intv_aug, n * 2)

    for i in 1:n
        t_i = float(time[i])
        δ_i = Int(event[i])
        # k_last = unique k ∈ 1:K with bp[k] < t_i ≤ bp[k+1].
        k_last = clamp(searchsortedfirst(bp, t_i) - 1, 1, K)
        for k in 1:k_last
            E_ik = min(t_i, bp[k + 1]) - bp[k]
            E_ik > 0 || continue
            y_ik = (k == k_last && δ_i == 1) ? 1 : 0
            push!(y_aug, y_ik)
            push!(E_aug, E_ik)
            push!(subj_aug, i)
            push!(intv_aug, k)
        end
    end

    return CoxphAugmented(y_aug, E_aug, subj_aug, intv_aug, bp, n, K)
end

# Default breakpoints: quantiles over event times. Mirrors R-INLA's
# behaviour of placing knots where most of the information about the
# baseline hazard sits, padded with 0 and a sentinel slightly above
# max(time) so that the longest-living subject has a non-zero exposure
# in the final interval.
function _default_coxph_breakpoints(time, event, nbreakpoints)
    event_times = [float(time[i]) for i in eachindex(time) if event[i] == 1]
    isempty(event_times) && (event_times = float.(time))
    sort!(event_times)
    qs = [(k - 0.5) / nbreakpoints for k in 1:nbreakpoints]
    inner = Statistics.quantile(event_times, qs)
    tmax = float(maximum(time))
    bp_raw = [0.0; collect(Float64, inner); tmax * 1.001 + 1e-9]
    return collect(Float64, unique(bp_raw))
end

"""
    coxph_design(aug::CoxphAugmented, X::AbstractMatrix) -> SparseMatrixCSC

Assemble the joint design matrix for the augmented Cox-PH-as-Poisson
fit: a sparse `[B | Xrep]` block where `B` is the one-hot
baseline-interval indicator and `Xrep` repeats each subject's
covariate row across that subject's augmented rows.

`X` must have `aug.n_subjects` rows.
"""
function coxph_design(aug::CoxphAugmented, X::AbstractMatrix)
    size(X, 1) == aug.n_subjects ||
        throw(DimensionMismatch(
            "X has $(size(X, 1)) rows; aug.n_subjects = $(aug.n_subjects)"))

    nrows = length(aug.y)
    p = size(X, 2)

    B = sparse(1:nrows, aug.interval, ones(nrows), nrows, aug.n_intervals)

    if p == 0
        return B
    end

    Xrep = sparse(X[aug.subject, :])
    return hcat(B, Xrep)
end

"""
    coxph_design(aug::CoxphAugmented) -> SparseMatrixCSC

Baseline-only design matrix (no covariates).
"""
coxph_design(aug::CoxphAugmented) = sparse(
    1:length(aug.y), aug.interval, ones(length(aug.y)),
    length(aug.y), aug.n_intervals)

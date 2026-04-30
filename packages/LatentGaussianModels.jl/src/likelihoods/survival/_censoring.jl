"""
    Censoring

Per-observation censoring mode for survival likelihoods. One of `NONE`,
`RIGHT`, `LEFT`, `INTERVAL`.

| Mode       | log p(y_i \\| η_i, θ)                          |
|-----------|------------------------------------------------|
| `NONE`    | `log f(t_i)` — uncensored event time           |
| `RIGHT`   | `log S(t_i)` — event time ≥ y[i]               |
| `LEFT`    | `log F(t_i)` — event time ≤ y[i]               |
| `INTERVAL`| `log[S(y[i]) - S(time_hi[i])]`                 |

Constructable from `Symbol` for keyword-call ergonomics:
`Censoring(:none) == NONE`. Storage is the enum (one byte per row), not
the `Symbol`, so the inner Newton loop's per-row branch is a cheap
integer compare.

See ADR-018 in `plans/decisions.md` for the full design rationale.
"""
@enum Censoring::UInt8 NONE RIGHT LEFT INTERVAL

function Censoring(s::Symbol)
    s === :none && return NONE
    s === :right && return RIGHT
    s === :left && return LEFT
    s === :interval && return INTERVAL
    throw(ArgumentError(
        "Unknown censoring mode :$s; expected one of " *
        ":none, :right, :left, :interval"))
end

"""
    logsubexp(a, b)

`log(exp(a) - exp(b))` for `a > b`, computed via `log1p` for digit
retention near `a ≈ b`.
"""
@inline function logsubexp(a, b)
    a > b || throw(ArgumentError("logsubexp requires a > b; got a=$a, b=$b"))
    return a + log1p(-exp(b - a))
end

"""
    validate_censoring(censoring, time_hi, y)

Public-boundary validation for survival-likelihood inputs. Asserts that
`censoring` and `time_hi` (when non-`nothing`) match `y` in length, and
that every `INTERVAL` row has `time_hi[i] > y[i]`.

Called once per `fit`; assertions are not in the inner Newton loop.
"""
validate_censoring(::Nothing, _, _) = nothing

function validate_censoring(censoring::AbstractVector{Censoring},
        time_hi::Union{Nothing, AbstractVector},
        y)
    length(censoring) == length(y) || throw(DimensionMismatch(
        "censoring vector length $(length(censoring)) does not match " *
        "y length $(length(y))"))
    has_interval = any(==(INTERVAL), censoring)
    if has_interval
        time_hi !== nothing || throw(ArgumentError(
            "INTERVAL censoring rows present but `time_hi` is `nothing`; " *
            "supply `time_hi::AbstractVector` with upper bounds"))
        length(time_hi) == length(y) || throw(DimensionMismatch(
            "time_hi length $(length(time_hi)) does not match " *
            "y length $(length(y))"))
        @inbounds for i in eachindex(censoring)
            if censoring[i] === INTERVAL && !(time_hi[i] > y[i])
                throw(ArgumentError(
                    "INTERVAL row $i: time_hi[$i] = $(time_hi[i]) must " *
                    "be > y[$i] = $(y[i])"))
            end
        end
    end
    return nothing
end

"""
    _coerce_censoring(c)

Internal helper: accepts `nothing`, `AbstractVector{Censoring}`, or
`AbstractVector{Symbol}` (e.g. `[:none, :right, :none]`) and returns
the canonical `Vector{Censoring}` storage form. Anything else throws.
"""
_coerce_censoring(::Nothing) = nothing
_coerce_censoring(c::AbstractVector{Censoring}) = c
_coerce_censoring(c::AbstractVector{Symbol}) = Censoring.(c)
function _coerce_censoring(c)
    throw(ArgumentError(
        "censoring must be `nothing`, an `AbstractVector{Censoring}`, " *
        "or an `AbstractVector{Symbol}`; got $(typeof(c))"))
end

# Censoring-aware accessors that broadcast correctly regardless of
# whether `censoring` is `nothing` (all-`NONE`) or a vector. Hot path —
# kept `@inline` so the typed `Nothing` branch constant-folds away.
@inline _censoring_at(::Nothing, _) = NONE
@inline _censoring_at(c::AbstractVector{Censoring}, i) = @inbounds c[i]

@inline _time_hi_at(::Nothing, _) = 0.0  # unread for non-`INTERVAL`
@inline _time_hi_at(t::AbstractVector, i) = @inbounds t[i]

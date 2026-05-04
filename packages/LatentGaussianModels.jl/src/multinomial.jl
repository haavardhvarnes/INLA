"""
    multinomial_to_poisson(Y; class_names = 1:K)

Reshape a multinomial-counts matrix `Y::AbstractMatrix{<:Integer}`
of shape `(n_rows, K)` into the long-format independent-Poisson
layout per ADR-024. Returns a `NamedTuple`:

- `y::Vector{Int}` — length `n_rows * K` count vector with row-major
  layout `y[(i-1)*K + k] = Y[i, k]`.
- `row_id::Vector{Int}` — same length; `row_id[(i-1)*K + k] = i`.
- `class_id::Vector{Int}` — same length; `class_id[(i-1)*K + k] = k`.
- `n_rows::Int`, `K::Int`, `n_long::Int = n_rows * K`.
- `class_names::Vector` — pass-through of the supplied class names.

The reformulation is the Multinomial-Poisson trick (Baker 1994; Chen
1985): the multinomial likelihood `Y_i ~ Multinomial(N_i, π_i)` is
equivalent to `Y_ik ~ Poisson(λ_ik)` with
`λ_ik = exp(α_i + x_i^⊤ β_k)`, where `α_i` is a per-row nuisance
intercept attached as `IID(n_rows; τ_init = -10.0, fix_τ = true)`
(matching R-INLA's `prec = list(initial = -10, fixed = TRUE)`).

Use [`multinomial_design_matrix`](@ref) to build the class-specific
covariate block; combine with the IID `α` block via
`hcat`/`StackedMapping` to get the full LGM design matrix.
"""
function multinomial_to_poisson(Y::AbstractMatrix{<:Integer}; class_names=nothing)
    n_rows, K = size(Y)
    n_rows >= 1 ||
        throw(ArgumentError("multinomial_to_poisson: Y must have at least one row"))
    K >= 2 ||
        throw(ArgumentError("multinomial_to_poisson: Y must have at least 2 classes"))
    n_long = n_rows * K
    y = Vector{Int}(undef, n_long)
    row_id = Vector{Int}(undef, n_long)
    class_id = Vector{Int}(undef, n_long)
    for i in 1:n_rows, k in 1:K
        idx = (i - 1) * K + k
        y[idx] = Int(Y[i, k])
        row_id[idx] = i
        class_id[idx] = k
    end
    cn = class_names === nothing ? collect(1:K) : collect(class_names)
    length(cn) == K || throw(DimensionMismatch(
        "multinomial_to_poisson: class_names has length $(length(cn)); " *
        "must equal K = $K"))
    return (; y=y, row_id=row_id, class_id=class_id,
        n_rows=n_rows, K=K, n_long=n_long, class_names=cn)
end

"""
    multinomial_design_matrix(helper, X; reference_class = helper.K)

Build the class-specific covariate block of the long-format design
matrix from an `(n_rows, p)` covariate matrix `X` and the layout
helper returned by [`multinomial_to_poisson`](@ref).

Returns a sparse `(n_long, (K - 1) * p)` matrix `A_β`. The rows
match the order of `helper.row_id`, `helper.class_id`; the reference
class's block of `p` coefficients is dropped to identify the model
(corner-point parameterisation, Agresti 2010 §8.5).

For long-format index `idx` with `i = helper.row_id[idx]`,
`k = helper.class_id[idx]`:

- if `k != reference_class`: columns `((k′ - 1) * p + 1) : (k′ * p)`
  carry `x_i^⊤`, where `k′ ∈ 1:(K-1)` is the index of `k` after
  dropping the reference.
- if `k == reference_class`: the row is all zeros (the linear
  predictor for the reference class is just the per-row α_i).

The per-row nuisance intercept `α_i` is *not* included here; attach
it as a separate `IID` block — see the `multinomial_to_poisson`
docstring for the recipe.
"""
function multinomial_design_matrix(helper::NamedTuple, X::AbstractMatrix;
        reference_class::Integer=helper.K)
    n_rows, p = size(X)
    n_rows == helper.n_rows || throw(DimensionMismatch(
        "multinomial_design_matrix: X has $(n_rows) rows; " *
        "helper expects $(helper.n_rows)"))
    K = helper.K
    1 <= reference_class <= K || throw(ArgumentError(
        "multinomial_design_matrix: reference_class = $reference_class " *
        "out of [1, $K]"))
    block_index = Vector{Int}(undef, K)
    kp = 0
    for k in 1:K
        if k == reference_class
            block_index[k] = 0
        else
            kp += 1
            block_index[k] = kp
        end
    end
    n_long = helper.n_long
    n_cols = (K - 1) * p
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    n_nonref = n_rows * (K - 1) * p
    sizehint!(rows, n_nonref)
    sizehint!(cols, n_nonref)
    sizehint!(vals, n_nonref)
    for idx in 1:n_long
        i = helper.row_id[idx]
        k = helper.class_id[idx]
        kk = block_index[k]
        kk == 0 && continue
        col_offset = (kk - 1) * p
        for j in 1:p
            v = X[i, j]
            if v != 0
                push!(rows, idx)
                push!(cols, col_offset + j)
                push!(vals, Float64(v))
            end
        end
    end
    return sparse(rows, cols, vals, n_long, n_cols)
end

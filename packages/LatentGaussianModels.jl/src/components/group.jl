"""
    Group(components::AbstractVector{C}) where {C <: AbstractLatentComponent}
    Group(factory, group_id::AbstractVector{<:Integer}; kwargs...)

Stack a list of inner components, all sharing a single hyperparameter
block, with potentially different sizes per group. Generalises
[`Replicate`](@ref) to the non-uniform case — `Group([AR1(5), AR1(5),
AR1(5)])` and `Replicate(AR1(5), 3)` produce equivalent priors, but
`Group([AR1(10), AR1(7), AR1(12)])` cannot be expressed as a Replicate.

# Hyperparameter sharing

Like Replicate, the wrapper exposes the *same* `nhyperparameters`,
`initial_hyperparameters`, and `log_hyperprior` as the first inner
component, and the same θ slice flows verbatim into each group's
`precision_matrix(component, θ)` call. All inner components must have
the same number of hyperparameters; the constructor validates this.
Beyond hyperparameter count, ensuring that the inner components share
prior structure (i.e. they were all built with the same prior types
and parameters) is the user's responsibility.

# Public API

- `Group(components)`: explicit per-group component vector. The Vector's
  element type is locked at construction.
- `Group(factory, group_id; kwargs...)`: convenience form. `group_id`
  is an integer vector with labels `1:G` (each present at least once);
  the constructor counts members per group and calls `factory(s_g;
  kwargs...)` for each group size `s_g`. Works for components whose
  primary constructor takes the size as a positional argument and
  prior options as keyword arguments — `IID`, `AR1`, `RW1`, `RW2`,
  `Seasonal`. Components needing more than a size (e.g. `Besag`,
  `Generic0`) must use the Vector form.

# Mechanics

- `length(g) = sum(length, components)`.
- `precision_matrix(g, θ) = blockdiag(Q₁, Q₂, …, Q_G)` where
  `Q_g = precision_matrix(components[g], θ)`.
- `prior_mean(g, θ)` is the concatenation of per-group prior means.
- `GMRFs.constraints(g)` block-stacks each group's constraint at its
  column offset within the stacked latent; groups with `NoConstraint`
  contribute no rows.
- `log_normalizing_constant(g, θ) = Σ_g log_normalizing_constant(
  components[g], θ)`.

# Example

```julia
# Replicated AR1 with unequal panel lengths (subjects with different
# numbers of visits), all sharing (τ, ρ).
panels = [AR1(10), AR1(7), AR1(12)]
g      = Group(panels)
m      = LatentGaussianModel(GaussianLikelihood(), (g,), A)
```
"""
struct Group{C <: AbstractLatentComponent} <: AbstractLatentComponent
    components::Vector{C}

    function Group{C}(components::AbstractVector{C}) where {C <: AbstractLatentComponent}
        n = length(components)
        n ≥ 1 ||
            throw(ArgumentError("Group: components must be non-empty"))
        nh = nhyperparameters(components[1])
        for i in 2:n
            nhyperparameters(components[i]) == nh || throw(ArgumentError(
                "Group: components[$i] has $(nhyperparameters(components[i])) " *
                "hyperparameters; components[1] has $nh — all inner components " *
                "must agree under the shared-θ contract"))
        end
        return new{C}(collect(components))
    end
end

function Group(components::AbstractVector{C}) where {C <: AbstractLatentComponent}
    return Group{C}(components)
end

function Group(factory, group_id::AbstractVector{<:Integer}; kwargs...)
    isempty(group_id) &&
        throw(ArgumentError("Group: group_id must be non-empty"))
    G = maximum(group_id)
    G ≥ 1 ||
        throw(ArgumentError("Group: group_id labels must be ≥ 1"))
    sizes = [count(==(g), group_id) for g in 1:G]
    all(>(0), sizes) || throw(ArgumentError(
        "Group: group_id labels must be 1:$G with each present at least once " *
        "(missing groups: $(findall(==(0), sizes)))"))
    components = [factory(s; kwargs...) for s in sizes]
    return Group(components)
end

Base.length(g::Group) = sum(length, g.components)

nhyperparameters(g::Group) = nhyperparameters(g.components[1])
initial_hyperparameters(g::Group) = initial_hyperparameters(g.components[1])
log_hyperprior(g::Group, θ) = log_hyperprior(g.components[1], θ)

function precision_matrix(g::Group, θ)
    blocks = [SparseMatrixCSC{Float64, Int}(precision_matrix(c, θ)) for c in g.components]
    return blockdiag(blocks...)
end

function prior_mean(g::Group, θ)
    return reduce(vcat, [prior_mean(c, θ) for c in g.components])
end

function log_normalizing_constant(g::Group, θ)
    s = zero(eltype(θ))
    for c in g.components
        s += log_normalizing_constant(c, θ)
    end
    return s
end

# Aggregate per-group constraints at the right column offsets in the
# stacked latent. Groups with `NoConstraint` contribute no rows; mixed
# constrained / unconstrained groups are supported.
function GMRFs.constraints(g::Group)
    A_blocks = Matrix{Float64}[]
    e_blocks = Vector{Float64}[]
    col_offset = 0
    for c in g.components
        n_c = length(c)
        kc = GMRFs.constraints(c)
        if !(kc isa GMRFs.NoConstraint)
            A_c = GMRFs.constraint_matrix(kc)
            e_c = GMRFs.constraint_rhs(kc)
            k_c = size(A_c, 1)
            A_full = zeros(Float64, k_c, length(g))
            @views A_full[:, (col_offset + 1):(col_offset + n_c)] .= A_c
            push!(A_blocks, A_full)
            push!(e_blocks, Vector{Float64}(e_c))
        end
        col_offset += n_c
    end
    isempty(A_blocks) && return GMRFs.NoConstraint()
    A = reduce(vcat, A_blocks)
    e = reduce(vcat, e_blocks)
    return GMRFs.LinearConstraint(A, e)
end

function gmrf(g::Group, θ)
    Q = precision_matrix(g, θ)
    rd = 0
    for c in g.components
        rd += GMRFs.rankdef(gmrf(c, θ))
    end
    return GMRFs.Generic0GMRF(Q; τ=1.0, rankdef=rd)
end

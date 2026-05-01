"""
    AbstractGMRFGraph

The topology on which a GMRF lives. A thin wrapper over a
`Graphs.AbstractGraph` that also caches the sparsity pattern of the
precision matrix (lower-triangular positions).

Concrete subtypes must implement:

- `graph(g) -> AbstractGraph`
- `num_nodes(g) -> Int`
- `adjacency_matrix(g) -> SparseMatrixCSC{Bool,Int}` *(symmetric, zero
  diagonal, Bool-valued)*
"""
abstract type AbstractGMRFGraph end

"""
    graph(g::AbstractGMRFGraph)

Return the wrapped `Graphs.AbstractGraph`.
"""
function graph end

"""
    num_nodes(g::AbstractGMRFGraph)

Number of latent nodes in the graph (`length` of any vector sampled
from a GMRF on `g`).
"""
function num_nodes end

"""
    adjacency_matrix(g::AbstractGMRFGraph)

Symmetric `Bool` adjacency matrix of the graph. The `(i,j)` entry is
`true` iff `i` and `j` are distinct and connected by an edge. Diagonal
is `false`. This is the precision-matrix sparsity pattern off-diagonal;
the diagonal is always present regardless.
"""
function adjacency_matrix end

"""
    laplacian_matrix(g::AbstractGMRFGraph)

The combinatorial graph Laplacian `L = D - A` as a
`SparseMatrixCSC{Int,Int}`, where `D` is the degree diagonal and `A`
is the adjacency matrix. This is the precision structure matrix for
intrinsic Besag / ICAR models (before `τ` scaling).
"""
function laplacian_matrix end

"""
    GMRFGraph{G<:AbstractGraph} <: AbstractGMRFGraph

Default concrete graph: wraps a `Graphs.SimpleGraph` plus the adjacency
matrix so we do not rebuild it every time we need the sparsity pattern.
"""
struct GMRFGraph{G <: AbstractGraph} <: AbstractGMRFGraph
    g::G
    A::SparseMatrixCSC{Bool, Int}   # adjacency
    L::SparseMatrixCSC{Int, Int}    # Laplacian (D - A)

    function GMRFGraph(g::G) where {G <: AbstractGraph}
        A = _adjacency(g)
        L = _laplacian_from_adjacency(A)
        new{G}(g, A, L)
    end
end

"""
    GMRFGraph(W::AbstractMatrix)

Build from an explicit `n × n` adjacency matrix `W`. Interprets any
nonzero entry as an edge; symmetry is required (checked) and the
diagonal is ignored. Loops are not currently supported — we error on a
nonzero diagonal to avoid silently downgrading a self-loop to nothing,
since gmrflib users sometimes pass a diagonal-loaded matrix by mistake.
"""
function GMRFGraph(W::AbstractMatrix)
    n, m = size(W)
    n == m || throw(DimensionMismatch("adjacency matrix must be square, got $n×$m"))
    if !issymmetric(W)
        throw(ArgumentError("adjacency matrix must be symmetric"))
    end
    any(!iszero, diag(W)) &&
        throw(ArgumentError("adjacency matrix must have zero diagonal; got nonzero diagonal entries"))

    g = SimpleGraph(n)
    # iterate over strictly upper triangle of nonzeros
    for j in 1:n, i in 1:(j - 1)
        if !iszero(W[i, j])
            add_edge!(g, i, j)
        end
    end
    return GMRFGraph(g)
end

graph(g::GMRFGraph) = g.g
num_nodes(g::GMRFGraph) = nv(g.g)
adjacency_matrix(g::GMRFGraph) = g.A
laplacian_matrix(g::GMRFGraph) = g.L

"""
    nconnected_components(g::AbstractGMRFGraph)

Number of connected components of the underlying graph. Relevant for
constraint generation on disconnected intrinsic models (one
sum-to-zero constraint per component, Freni-Sterrantino et al. 2018).
"""
nconnected_components(g::AbstractGMRFGraph) = length(connected_components(graph(g)))

"""
    connected_component_labels(g::AbstractGMRFGraph) -> Vector{Int}

Assign each node an integer component label in `1:nconnected_components(g)`.
"""
function connected_component_labels(g::AbstractGMRFGraph)
    comps = connected_components(graph(g))
    n = num_nodes(g)
    labels = zeros(Int, n)
    for (k, comp) in enumerate(comps)
        for i in comp
            labels[i] = k
        end
    end
    return labels
end

# ---------- helpers (not exported) ----------

function _adjacency(g::AbstractGraph)
    n = nv(g)
    I = Int[]
    J = Int[]
    V = Bool[]
    sizehint!(I, 2 * ne(g))
    sizehint!(J, 2 * ne(g))
    sizehint!(V, 2 * ne(g))
    for e in Graphs.edges(g)
        u, v = Graphs.src(e), Graphs.dst(e)
        push!(I, u)
        push!(J, v)
        push!(V, true)
        push!(I, v)
        push!(J, u)
        push!(V, true)
    end
    return sparse(I, J, V, n, n)
end

function _laplacian_from_adjacency(A::SparseMatrixCSC{Bool, Int})
    n = size(A, 1)
    # degrees = sum of Bool adjacency rows
    deg = Int.(sum(A; dims=2)[:, 1])
    # L = D - A, entries are integers (degree on diagonal, -1 on edges)
    return spdiagm(0 => deg) - convert(SparseMatrixCSC{Int, Int}, A)
end

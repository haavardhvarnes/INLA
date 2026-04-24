"""
    GMRFs

Sparse Gaussian Markov Random Fields for Julia. Standalone-useful and
the numerical core of the Julia INLA ecosystem.

See the package README and [`plans/plan.md`](../plans/plan.md) for the
design document.
"""
module GMRFs

using LinearAlgebra
using Random
using SparseArrays
using Statistics

using Graphs: Graphs, AbstractGraph, SimpleGraph, add_edge!, nv, ne,
              connected_components
import Graphs: adjacency_matrix, laplacian_matrix
using Distributions: Distributions
import Distributions: logpdf
using LinearSolve: LinearSolve
using SelectedInversion: selinv, selinv_diag

# Public abstract types — load first so concrete types can subtype them.
include("graph.jl")
include("precision.jl")
include("gmrf.jl")
include("nullspace.jl")
include("sampling.jl")
include("logdensity.jl")
include("constraints.jl")
include("marginals.jl")
include("factorization.jl")

# Graph + precision
export AbstractGMRFGraph, GMRFGraph
export graph, num_nodes,
       nconnected_components, connected_component_labels
# Re-export the Graphs.jl names we extend, so `using GMRFs` exposes
# them as method-rich functions without a Graphs.jl import on the
# caller side.
export adjacency_matrix, laplacian_matrix
export SymmetricQ, tabulated_precision

# GMRFs
export AbstractGMRF
export IIDGMRF, RW1GMRF, RW2GMRF, AR1GMRF, SeasonalGMRF, BesagGMRF, Generic0GMRF
export precision_matrix, prior_mean, rankdef, null_space_basis
export scale_model, scale_factor, is_scaled

# We extend `Base.rand`, `Random.rand!`, and `Distributions.logpdf`
# with methods on `AbstractGMRF`. `rand` / `rand!` come from Base/Random
# and need no re-export; `logpdf` is re-exported so `using GMRFs` is
# enough to call `logpdf(gmrf, x)`.
export logpdf

# Constraints
export AbstractConstraint, NoConstraint, LinearConstraint
export constraints, constraint_matrix, constraint_rhs, nconstraints
export sum_to_zero_constraints

# Marginals + factorisation
export marginal_variances
export FactorCache, update!, factor

end # module

"""
    lumped_mass(C) -> C_lumped

Diagonal mass lumping: replace the full mass matrix `C` with the diagonal
matrix whose `i`-th entry is the sum of row `i` of `C`. This preserves
the total mass (`sum(C_lumped) == sum(C)`, which equals the domain area
for P1 FEM) while making `C_lumped` trivially invertible.

Used in SPDE-Matérn precision assembly for α = 2 to keep `G₂ = G₁ C̃⁻¹ G₁`
sparse, following Lindgren, Rue and Lindström (2011, Appendix C).

# Arguments
- `C::AbstractSparseMatrix` — the full mass matrix from `assemble_fem_matrices`.

# Returns
- `C_lumped::SparseMatrixCSC` — diagonal sparse matrix with the same
  size and element type as `C`.
"""
function lumped_mass(C::AbstractSparseMatrix)
    n = size(C, 1)
    n == size(C, 2) ||
        throw(ArgumentError("C must be square; got size $(size(C))"))
    d = vec(sum(C; dims=2))
    return sparse(1:n, 1:n, d, n, n)
end

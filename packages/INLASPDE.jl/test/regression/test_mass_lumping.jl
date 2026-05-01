# Mass lumping: `C̃[i,i] = Σ_j C[i,j]` and zero off-diagonal. Two
# invariants must hold:
#   (1) diagonality — `C̃` has only diagonal entries;
#   (2) conservation — `sum(C̃) == sum(C)` (total area is preserved).

using LinearAlgebra
using SparseArrays

@testset "lumped_mass — 2-triangle unit square (hand-computed)" begin
    C, _ = assemble_fem_matrices(POINTS_SQ, TRIS_SQ)
    Cl = lumped_mass(C)

    @test size(Cl) == size(C)
    @test Matrix(Cl) ≈ Diagonal(diag(Cl))         # diagonality
    @test sum(Cl)≈sum(C) rtol=1.0e-12          # conservation

    # Hand-computed lumped diagonal:
    #   row sums of the 4×4 mass matrix above.
    # Vertex 1 (in both triangles): 1/6 + 1/24 + 1/12 + 1/24 = 1/3
    # Vertex 2 (T₁ only):           1/24 + 1/12 + 1/24       = 1/6
    # Vertex 3 (in both triangles): 1/12 + 1/24 + 1/6 + 1/24 = 1/3
    # Vertex 4 (T₂ only):           1/24 + 1/24 + 1/12       = 1/6
    @test diag(Cl)≈[1 / 3, 1 / 6, 1 / 3, 1 / 6] rtol=1.0e-12
end

@testset "lumped_mass — conservation on a random triangulation" begin
    # A 3×3 regular grid split into right triangles — 18 triangles, area 1.
    nx, ny = 3, 3
    xs = range(0.0, 1.0; length=nx + 1)
    ys = range(0.0, 1.0; length=ny + 1)
    npts = (nx + 1) * (ny + 1)
    points = Matrix{Float64}(undef, npts, 2)
    idx(i, j) = (j - 1) * (nx + 1) + i
    for j in 1:(ny + 1), i in 1:(nx + 1)
        points[idx(i, j), 1] = xs[i]
        points[idx(i, j), 2] = ys[j]
    end
    tris = Int[]
    for j in 1:ny, i in 1:nx
        push!(tris, idx(i, j), idx(i + 1, j), idx(i + 1, j + 1))
        push!(tris, idx(i, j), idx(i + 1, j + 1), idx(i, j + 1))
    end
    triangles = permutedims(reshape(tris, 3, :))

    C, _ = assemble_fem_matrices(points, triangles)
    Cl = lumped_mass(C)

    @test sum(Cl)≈1.0 rtol=1.0e-12
    @test all(>(0), diag(Cl))   # every vertex has positive mass
end

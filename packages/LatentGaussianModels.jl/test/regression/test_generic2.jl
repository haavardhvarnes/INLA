using LatentGaussianModels: Generic2, PCPrecision, GammaPrecision,
                            precision_matrix, log_hyperprior,
                            log_prior_density, nhyperparameters,
                            initial_hyperparameters,
                            log_normalizing_constant
using GMRFs: NoConstraint, LinearConstraint, constraints, Generic0GMRF
import GMRFs

@testset "Generic2 — block precision structure" begin
    # Plain SPD C; the block precision should match the R-INLA eq. 1
    # form `[τu I, -τu I; -τu I, τu I + τv C]` exactly.
    C = sparse([2.0 -0.5 0.0;
                -0.5 2.0 -0.5;
                0.0 -0.5 2.0])
    n = size(C, 1)

    c = Generic2(C; rankdef=0)
    @test length(c) == 2n
    @test nhyperparameters(c) == 2
    @test initial_hyperparameters(c) == [0.0, 0.0]

    θ = [log(2.0), log(3.0)]   # τv = 2, τu = 3
    τv, τu = 2.0, 3.0
    Q = precision_matrix(c, θ)
    @test size(Q) == (2n, 2n)

    Qd = Matrix(Q)
    I_n = Matrix(I, n, n)
    @test Qd[1:n, 1:n] ≈ τu .* I_n
    @test Qd[1:n, (n + 1):2n] ≈ -τu .* I_n
    @test Qd[(n + 1):2n, 1:n] ≈ -τu .* I_n
    @test Qd[(n + 1):2n, (n + 1):2n] ≈ τu .* I_n .+ τv .* Matrix(C)
    @test issymmetric(Qd)

    @test constraints(c) isa NoConstraint
end

@testset "Generic2 — log NC matches Schur-complement formula" begin
    # |Q|_+ = τ_u^n · τ_v^(n-rd) · |C|_+ ⇒
    #   ½ log|Q|_+ = ½ n log τ_u + ½ (n-rd) log τ_v + ½ log|C|_+.
    # The user-independent ½ log|C|_+ is dropped (F_GENERIC0 convention),
    # leaving `-½ (2n-rd) log(2π) + ½ n θ₂ + ½ (n-rd) θ₁`.
    C = sparse([3.0 -1.0 0.0 0.0;
                -1.0 3.0 -1.0 0.0;
                0.0 -1.0 3.0 -1.0;
                0.0 0.0 -1.0 3.0])
    n = size(C, 1)
    c = Generic2(C; rankdef=0)
    θ = [0.7, 1.3]
    expected = -0.5 * (2n) * log(2π) + 0.5 * n * θ[2] + 0.5 * n * θ[1]
    @test log_normalizing_constant(c, θ) ≈ expected
end

@testset "Generic2 — rank-deficient C with constraint" begin
    # ICAR-Laplacian C on a 5-node path graph (rankdef = 1, null = 1).
    # The joint Q has the same rankdef (Schur complement preserves rank
    # since the (1,1) block is full rank), so a single sum-to-zero on
    # the v-block resolves the ambiguity.
    n = 5
    C = sparse([1.0 -1.0 0.0 0.0 0.0;
                -1.0 2.0 -1.0 0.0 0.0;
                0.0 -1.0 2.0 -1.0 0.0;
                0.0 0.0 -1.0 2.0 -1.0;
                0.0 0.0 0.0 -1.0 1.0])
    Aeq = zeros(1, 2n)
    Aeq[1, (n + 1):2n] .= 1.0  # sum-to-zero on the v block
    e = zeros(1)
    c = Generic2(C; rankdef=1, constraint=LinearConstraint(Aeq, e))

    @test length(c) == 2n
    @test constraints(c) isa LinearConstraint

    θ = [0.4, -0.2]
    expected = -0.5 * (2n - 1) * log(2π) +
               0.5 * n * θ[2] +
               0.5 * (n - 1) * θ[1]
    @test log_normalizing_constant(c, θ) ≈ expected
end

@testset "Generic2 — scale_model applies Sørbye-Rue" begin
    # Compare scaled vs unscaled C: with scale_model = true the stored
    # C should equal `c · C_orig` for some scalar c > 0 (the geometric-
    # mean scale factor), so the precision_matrix at τv = 1 reflects the
    # rescaling.
    n = 4
    Cmat = sparse([2.0 -1.0 0.0 0.0;
                   -1.0 2.0 -1.0 0.0;
                   0.0 -1.0 2.0 -1.0;
                   0.0 0.0 -1.0 2.0])
    c_unscaled = Generic2(Cmat; rankdef=0, scale_model=false)
    c_scaled = Generic2(Cmat; rankdef=0, scale_model=true)
    @test c_unscaled.C ≈ Cmat                # unscaled passthrough
    @test !(c_scaled.C ≈ Cmat)               # scaling should change C
    @test issymmetric(c_scaled.C)

    # The scaling factor the package computed should also surface in the
    # block (2,2) entry of Q at τ_v = 1, τ_u = 0 (would be exactly τ_v C̃).
    θ = [0.0, -50.0]   # τv = 1, τu ≈ 0
    Q = Matrix(precision_matrix(c_scaled, θ))
    τu = exp(θ[2])
    D_block = Q[(n + 1):2n, (n + 1):2n]
    @test D_block ≈ τu .* Matrix(I, n, n) .+ Matrix(c_scaled.C)
end

@testset "Generic2 — log_hyperprior sums two priors" begin
    C = sparse([2.0 -0.5; -0.5 2.0])
    p1 = GammaPrecision(1.0, 5.0e-5)
    p2 = GammaPrecision(1.0, 1.0e-3)
    c = Generic2(C; hyperprior_τv=p1, hyperprior_τu=p2)

    θ = [0.3, -0.4]
    expected = log_prior_density(p1, θ[1]) + log_prior_density(p2, θ[2])
    @test log_hyperprior(c, θ) ≈ expected
end

@testset "Generic2 — gmrf wrapper precision matches" begin
    C = sparse([2.0 -0.5 0.0;
                -0.5 2.0 -0.5;
                0.0 -0.5 2.0])
    c = Generic2(C; rankdef=0)
    θ = [log(1.7), log(2.4)]
    g = LatentGaussianModels.gmrf(c, θ)
    @test g isa Generic0GMRF
    @test GMRFs.precision_matrix(g) ≈ precision_matrix(c, θ)
end

@testset "Generic2 — invalid input rejection" begin
    Crec = sparse([1.0 0.0 0.0; 0.0 1.0 0.0])
    @test_throws DimensionMismatch Generic2(Crec)

    Csq_nsym = sparse([1.0 0.0; 1.0 1.0])
    @test_throws ArgumentError Generic2(Csq_nsym)

    C_ok = sparse([1.0 0.0; 0.0 1.0])
    @test_throws ArgumentError Generic2(C_ok; rankdef=-1)
    @test_throws ArgumentError Generic2(C_ok; rankdef=3)
end

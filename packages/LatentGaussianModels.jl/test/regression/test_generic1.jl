using LatentGaussianModels: Generic0, Generic1, PCPrecision, GammaPrecision,
                            precision_matrix, log_hyperprior, log_prior_density,
                            nhyperparameters,
                            initial_hyperparameters, log_normalizing_constant
using GMRFs: NoConstraint, LinearConstraint, constraints

@testset "Generic1 — eigenvalue rescaling" begin
    R = sparse([2.0 -0.5 0.0;
                -0.5 2.0 -0.5;
                0.0 -0.5 2.0])
    λ_max_R = maximum(eigvals(Symmetric(Matrix(R))))

    c = Generic1(R; rankdef=0)
    @test length(c) == 3
    @test nhyperparameters(c) == 1
    @test initial_hyperparameters(c) == [0.0]
    @test c.λ_max_original ≈ λ_max_R

    # `c.R` should equal `R / λ_max(R)`, hence λ_max(c.R) ≈ 1.
    @test maximum(eigvals(Symmetric(Matrix(c.R))))≈1.0 atol=1.0e-12
    @test Matrix(c.R) ≈ Matrix(R) ./ λ_max_R

    # precision_matrix(c, [θ]) = exp(θ) · R̃
    Q = precision_matrix(c, [log(3.0)])
    @test Matrix(Q) ≈ 3.0 .* (Matrix(R) ./ λ_max_R)

    @test constraints(c) isa NoConstraint
end

@testset "Generic1 — agrees with Generic0(R/λ_max)" begin
    R = sparse([3.0 -1.0 0.0 0.0;
                -1.0 3.0 -1.0 0.0;
                0.0 -1.0 3.0 -1.0;
                0.0 0.0 -1.0 3.0])
    λ_max_R = maximum(eigvals(Symmetric(Matrix(R))))
    R_scaled = SparseMatrixCSC{Float64, Int}(R) ./ λ_max_R

    c1 = Generic1(R)
    c0 = Generic0(R_scaled)

    θ = [0.4]
    @test Matrix(precision_matrix(c1, θ)) ≈ Matrix(precision_matrix(c0, θ))
    @test log_normalizing_constant(c1, θ) ≈ log_normalizing_constant(c0, θ)
    @test log_hyperprior(c1, θ) ≈ log_hyperprior(c0, θ)
end

@testset "Generic1 — rank-deficient with constraint" begin
    n = 5
    R = sparse([1.0 -1.0 0.0 0.0 0.0;
                -1.0 2.0 -1.0 0.0 0.0;
                0.0 -1.0 2.0 -1.0 0.0;
                0.0 0.0 -1.0 2.0 -1.0;
                0.0 0.0 0.0 -1.0 1.0])
    Aeq = ones(1, n)
    e = zeros(1)
    c = Generic1(R; rankdef=1, constraint=LinearConstraint(Aeq, e))

    @test length(c) == n
    @test maximum(eigvals(Symmetric(Matrix(c.R))))≈1.0 atol=1.0e-12
    @test constraints(c) isa LinearConstraint

    # log NC: -½ (n - rd) log(2π) + ½ (n - rd) log τ (R-INLA F_GENERIC1
    # without β-mixing reduces to F_GENERIC0).
    @test log_normalizing_constant(c, [0.7]) ≈
          -0.5 * (n - 1) * log(2π) + 0.5 * (n - 1) * 0.7
end

@testset "Generic1 — invalid input rejection" begin
    Rrec = sparse([1.0 0.0 0.0; 0.0 1.0 0.0])
    @test_throws DimensionMismatch Generic1(Rrec)

    Rsq_nsym = sparse([1.0 0.0; 1.0 1.0])
    @test_throws ArgumentError Generic1(Rsq_nsym)

    R_ok = sparse([1.0 0.0; 0.0 1.0])
    @test_throws ArgumentError Generic1(R_ok; rankdef=-1)

    # All-zero R has λ_max = 0 → rejected.
    Rzero = spzeros(3, 3)
    @test_throws ArgumentError Generic1(Rzero)
end

@testset "Generic1 — custom hyperprior" begin
    R = sparse([2.0 -0.5; -0.5 2.0])
    c = Generic1(R; hyperprior=GammaPrecision(1.0, 5.0e-5))
    θ = [0.3]
    expected = log_prior_density(GammaPrecision(1.0, 5.0e-5), θ[1])
    @test log_hyperprior(c, θ) ≈ expected
end

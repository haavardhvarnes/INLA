using LatentGaussianModels: Seasonal, PCPrecision, GammaPrecision,
    precision_matrix, log_hyperprior, log_prior_density, nhyperparameters,
    initial_hyperparameters, log_normalizing_constant
using GMRFs: SeasonalGMRF, LinearConstraint, constraints, constraint_matrix,
    constraint_rhs

@testset "Seasonal — delegates to SeasonalGMRF" begin
    n = 12
    s = 3
    c = Seasonal(n; period = s)

    @test length(c) == n
    @test c.period == s
    @test nhyperparameters(c) == 1
    @test initial_hyperparameters(c) == [0.0]

    # Precision: c uses τ = exp(θ[1]); compare against direct GMRF call.
    θ = [log(2.5)]
    Q_lgm = precision_matrix(c, θ)
    Q_ref = GMRFs.precision_matrix(SeasonalGMRF(n; period = s, τ = 2.5))
    @test Matrix(Q_lgm) ≈ Matrix(Q_ref)
end

@testset "Seasonal — constraint matches GMRF default (single sum-to-zero)" begin
    n = 10
    s = 4
    c = Seasonal(n; period = s)

    kc = constraints(c)
    @test kc isa LinearConstraint
    @test size(constraint_matrix(kc)) == (1, n)
    @test constraint_rhs(kc) == zeros(1)

    # Same constraint as the bare GMRF.
    kc_ref = constraints(SeasonalGMRF(n; period = s))
    @test constraint_matrix(kc) == constraint_matrix(kc_ref)
end

@testset "Seasonal — log NC matches R-INLA F_SEASONAL convention" begin
    # `-½(n - rd_eff) log(2π) + ½(n - rd_eff) log τ` with rd_eff = s.
    # The +1 over the "raw" rd = s-1 accounts for the sum-to-zero
    # constraint hitting `range(Q)` (the all-ones vector lies in the
    # column space of the seasonal structure matrix), so one PD
    # direction is consumed by the constraint and the τ-scaled prior
    # dimension on the constraint surface is `n - s`, not `n - (s-1)`.
    n = 9
    s = 3
    c = Seasonal(n; period = s)
    θ = [0.7]
    rd_eff = s
    @test log_normalizing_constant(c, θ) ≈
        -0.5 * (n - rd_eff) * log(2π) + 0.5 * (n - rd_eff) * θ[1]
end

@testset "Seasonal — invalid input rejection" begin
    @test_throws ArgumentError Seasonal(4; period = 4)
    @test_throws ArgumentError Seasonal(10; period = 1)
end

@testset "Seasonal — custom hyperprior" begin
    c = Seasonal(8; period = 2, hyperprior = GammaPrecision(1.0, 5.0e-5))
    θ = [0.3]
    expected = log_prior_density(GammaPrecision(1.0, 5.0e-5), θ[1])
    @test log_hyperprior(c, θ) ≈ expected
end

using LatentGaussianModels: Intercept, FixedEffects, IID, RW1, RW2, AR1, Besag,
                            PCPrecision, precision_matrix, log_hyperprior, nhyperparameters,
                            initial_hyperparameters
using GMRFs: GMRFGraph, RW1GMRF, RW2GMRF, IIDGMRF, AR1GMRF, BesagGMRF,
             num_nodes

@testset "Intercept" begin
    c = Intercept()
    @test length(c) == 1
    @test nhyperparameters(c) == 0
    Q = precision_matrix(c, Float64[])
    @test size(Q) == (1, 1)
    @test Q[1, 1] ≈ 1.0e-3
    @test log_hyperprior(c, Float64[]) == 0.0
end

@testset "FixedEffects" begin
    c = FixedEffects(3)
    @test length(c) == 3
    @test nhyperparameters(c) == 0
    Q = precision_matrix(c, Float64[])
    @test size(Q) == (3, 3)
    @test Matrix(Q) ≈ 1.0e-3 * I(3)
end

@testset "IID component" begin
    c = IID(5)
    @test length(c) == 5
    @test nhyperparameters(c) == 1
    @test initial_hyperparameters(c) == [0.0]
    Q = precision_matrix(c, [log(4.0)])
    @test Matrix(Q) ≈ 4.0 * I(5)
    @test isfinite(log_hyperprior(c, [0.0]))
end

@testset "RW1 component" begin
    c = RW1(6)
    @test length(c) == 6
    @test nhyperparameters(c) == 1
    Q = precision_matrix(c, [log(2.0)])
    # Compare against direct RW1GMRF
    Qref = GMRFs.precision_matrix(RW1GMRF(6; τ=2.0))
    @test Matrix(Q) ≈ Matrix(Qref)
end

@testset "RW2 component" begin
    c = RW2(7)
    Q = precision_matrix(c, [0.0])
    Qref = GMRFs.precision_matrix(RW2GMRF(7; τ=1.0))
    @test Matrix(Q) ≈ Matrix(Qref)
end

@testset "AR1 component" begin
    c = AR1(4)
    @test nhyperparameters(c) == 2
    @test initial_hyperparameters(c) == [0.0, 0.0]
    θ = [log(2.0), atanh(0.3)]
    Q = precision_matrix(c, θ)
    Qref = GMRFs.precision_matrix(AR1GMRF(4; ρ=0.3, τ=2.0))
    @test Matrix(Q) ≈ Matrix(Qref)
end

@testset "Besag component" begin
    W = [0 1 0 0;
         1 0 1 0;
         0 1 0 1;
         0 0 1 0]
    g = GMRFGraph(W)
    c = Besag(g; scale_model=false)
    @test length(c) == 4
    Q = precision_matrix(c, [log(3.0)])
    Qref = GMRFs.precision_matrix(BesagGMRF(g; τ=3.0, scale_model=false))
    @test Matrix(Q) ≈ Matrix(Qref)
end

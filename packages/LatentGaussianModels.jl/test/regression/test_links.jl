using LatentGaussianModels: inverse_link, ∂inverse_link, ∂²inverse_link,
                            IdentityLink, LogLink, LogitLink, ProbitLink, CloglogLink

# Finite-difference helper
function fd(f, x, h=1.0e-6)
    return (f(x + h) - f(x - h)) / (2h)
end

@testset "IdentityLink" begin
    @test inverse_link(IdentityLink(), 0.7) == 0.7
    @test ∂inverse_link(IdentityLink(), 0.7) == 1
    @test ∂²inverse_link(IdentityLink(), 0.7) == 0
end

@testset "LogLink" begin
    for η in (-1.5, 0.0, 2.3)
        @test inverse_link(LogLink(), η) ≈ exp(η)
        @test ∂inverse_link(LogLink(), η) ≈ exp(η)
        @test ∂²inverse_link(LogLink(), η) ≈ exp(η)
        @test ∂inverse_link(LogLink(), η)≈fd(x -> inverse_link(LogLink(), x), η) atol=1.0e-5
    end
end

@testset "LogitLink" begin
    for η in (-5.0, -0.5, 0.0, 2.0)
        p = inverse_link(LogitLink(), η)
        @test 0 < p < 1
        @test ∂inverse_link(LogitLink(), η)≈fd(x -> inverse_link(LogitLink(), x), η) atol=1.0e-5
        @test ∂²inverse_link(LogitLink(), η)≈fd(x -> ∂inverse_link(LogitLink(), x), η) atol=1.0e-5
    end
end

@testset "ProbitLink" begin
    for η in (-2.0, 0.0, 1.5)
        @test 0 < inverse_link(ProbitLink(), η) < 1
        @test ∂inverse_link(ProbitLink(), η)≈fd(x -> inverse_link(ProbitLink(), x), η) atol=1.0e-5
        @test ∂²inverse_link(ProbitLink(), η)≈fd(x -> ∂inverse_link(ProbitLink(), x), η) atol=1.0e-5
    end
end

@testset "CloglogLink" begin
    for η in (-1.0, 0.0, 1.0)
        @test 0 < inverse_link(CloglogLink(), η) < 1
        @test ∂inverse_link(CloglogLink(), η)≈fd(x -> inverse_link(CloglogLink(), x), η) atol=1.0e-5
        @test ∂²inverse_link(CloglogLink(), η)≈fd(x -> ∂inverse_link(CloglogLink(), x), η) atol=1.0e-5
    end
end

# M5 core — end-to-end synthetic Matérn-recovery integration test.
#
# This is the in-package stand-in for the Meuse vignette. We build a
# mesh, draw a Matérn field from the SPDE prior at a known `(ρ, σ)`,
# generate Gaussian observations at random interior locations via the
# `MeshProjector`, fit the full `LatentGaussianModel` with Empirical
# Bayes, and verify that the posterior summaries recover the data-
# generating hyperparameters within a loose tolerance and that the
# latent posterior mean correlates with the true field.
#
# The R-INLA oracle fixture and the vendored Meuse zinc dataset are
# deferred to M6 (see plans/plan.md) — those need a pinned R/R-INLA
# environment and a data-generation script, which lives outside the
# scope of v0.1.
#
# Reproducibility: all randomness goes through `MersenneTwister(seed)`.

using Random: MersenneTwister, randn, rand
using Statistics: cor
using GMRFs: Generic0GMRF
using LatentGaussianModels: LatentGaussianModel, GaussianLikelihood,
    empirical_bayes

@testset "M5 — synthetic Matérn recovery via Empirical Bayes" begin
    rng = MersenneTwister(42)

    # Mesh on [0, 1]² with a small outer buffer to soften boundary
    # effects on the SPDE solution.
    sq = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    mesh = inla_mesh_2d(;
        boundary = sq, max_edge = 0.1, offset = 0.2, min_angle = 25.0,
    )
    @test num_vertices(mesh) > 100

    # PC-Matérn prior kept wide so the data, not the prior, drives the
    # posterior mode on (ρ, σ).
    pc = PCMatern(
        range_U = 0.3, range_α = 0.5,
        sigma_U = 1.0, sigma_α = 0.5,
    )
    spde = SPDE2(mesh; pc = pc)

    # Truth: pick (ρ, σ) comfortably inside the domain diameter.
    ρ_true, σ_true = 0.3, 1.0
    θ_true = collect(spde_internal_scale(spde, ρ_true, σ_true))

    # Draw a true field from the SPDE prior. Wrap Q in a Generic0GMRF;
    # the sparse FEM product picks up floating-point asymmetry, so
    # symmetrize before handing off to the GMRF constructor.
    Q_true = LatentGaussianModels.precision_matrix(spde, θ_true)
    Q_sym = (Q_true + Q_true') / 2
    x_true = rand(rng, Generic0GMRF(Q_sym; τ = 1.0))

    # Gaussian observations at random interior locations.
    n_obs = 150
    locs = 0.05 .+ 0.9 .* rand(rng, n_obs, 2)
    P = MeshProjector(mesh, locs)
    σ_noise = 0.2
    y = (P * x_true) .+ σ_noise .* randn(rng, n_obs)

    # End-to-end fit: GaussianLikelihood with identity link + SPDE2.
    like = GaussianLikelihood()
    model = LatentGaussianModel(like, spde, P.A)
    res = empirical_bayes(model, y)

    # θ layout is [log τ_lik, log τ_spde, log κ_spde].
    @test length(res.θ̂) == 3
    @test res.laplace.converged

    # Recovery of the SPDE hyperparameters on the user scale. A factor-
    # of-two band is generous but appropriate for n = 150 and EB: we are
    # checking that the inference pipeline runs end-to-end and lands on
    # the correct order of magnitude, not hitting a tight R-INLA-like
    # oracle tolerance (that test lives in M6).
    ρ̂, σ̂ = spde_user_scale(spde, res.θ̂[2:3])
    @test 0.5 * ρ_true < ρ̂ < 2.0 * ρ_true
    @test 0.5 * σ_true < σ̂ < 2.0 * σ_true

    # Recovery of the Gaussian noise scale via τ_lik = σ_noise^{-2}.
    σ̂_noise = 1 / sqrt(exp(res.θ̂[1]))
    @test 0.5 * σ_noise < σ̂_noise < 2.0 * σ_noise

    # Posterior latent mean should correlate with the truth.
    @test cor(res.laplace.mode, x_true) > 0.6
end

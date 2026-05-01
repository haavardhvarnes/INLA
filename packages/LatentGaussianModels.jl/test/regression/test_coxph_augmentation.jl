# Regression tests for the Cox-PH-to-Poisson data augmentation. Verifies
# the structural invariants that make the augmented Poisson fit equivalent
# to the original piecewise-exponential survival model:
#
#   1. Each subject contributes a row exactly for every interval they
#      enter; subjects with `t_i ≤ bp[k]` contribute nothing to interval k.
#   2. Total exposure per subject ≡ t_i (the augmentation is a partition).
#   3. Total event count per subject equals the original event indicator.
#   4. Augmentation + subject restriction reproduces the original record.
#   5. The expected augmented Poisson log-density matches the
#      piecewise-exponential log-density evaluated at the true linear
#      predictor — this is the algebraic justification for the augmentation.
#
# The 5th check is the key correctness gate: if the per-row Poisson
# contributions sum to the exact piecewise-exponential log-density (up
# to terms independent of η), then any inference engine that fits the
# Poisson model is also a valid fit of the original Cox PH. We pin
# this for several random configurations.

using Test
using Random
using SparseArrays
using Statistics
using LatentGaussianModels: inla_coxph, coxph_design, CoxphAugmented,
                            PoissonLikelihood, log_density

@testset "inla_coxph: basic augmentation invariants" begin
    Random.seed!(20260430)
    n = 50
    bp = [0.0, 1.0, 2.0, 3.0, 4.0]
    K = length(bp) - 1
    time = clamp.(rand(n) .* 3.5 .+ 0.1, 0.1, 3.95)
    event = rand(0:1, n)

    aug = inla_coxph(time, event; breakpoints=bp)

    # Type / size invariants ---------------------------------------------
    @test aug isa CoxphAugmented
    @test aug.n_subjects == n
    @test aug.n_intervals == K
    @test aug.breakpoints == bp
    @test length(aug.y) == length(aug.E) ==
          length(aug.subject) == length(aug.interval)
    @test all(>=(0), aug.E)
    @test all(in(0:1), aug.y)
    @test all(in(1:n), aug.subject)
    @test all(in(1:K), aug.interval)

    # Invariant 2: total exposure per subject ≡ time -----------------------
    for i in 1:n
        rows = findall(==(i), aug.subject)
        @test sum(aug.E[rows])≈time[i] atol=1e-12
    end

    # Invariant 3: event count per subject ≡ event[i] ----------------------
    for i in 1:n
        rows = findall(==(i), aug.subject)
        @test sum(aug.y[rows]) == event[i]
    end

    # Invariant 1: subjects contribute up to k_last ------------------------
    for i in 1:n
        rows = findall(==(i), aug.subject)
        ks = aug.interval[rows]
        @test ks == sort(ks) == collect(1:length(ks))
        # The last interval visited is the one containing t_i:
        k_last = ks[end]
        @test bp[k_last] < time[i] ≤ bp[k_last + 1]
        # And the event flag sits exactly on the last row, iff event=1
        @test aug.y[rows[end]] == event[i]
        @test all(==(0), aug.y[rows[1:(end - 1)]])
    end
end

@testset "inla_coxph: explicit small example" begin
    # 3 subjects, 4 intervals; check every row by hand.
    time = [0.5, 2.5, 3.8]
    event = [1, 0, 1]
    bp = [0.0, 1.0, 2.0, 3.0, 4.0]

    aug = inla_coxph(time, event; breakpoints=bp)

    # Subject 1: dies at t=0.5, only crosses interval 1.
    # Row: (subj=1, k=1, E=0.5, y=1)
    # Subject 2: censored at 2.5, crosses intervals 1,2,3.
    # Rows: (2,1,1.0,0), (2,2,1.0,0), (2,3,0.5,0)
    # Subject 3: dies at 3.8, crosses intervals 1,2,3,4.
    # Rows: (3,1,1.0,0), (3,2,1.0,0), (3,3,1.0,0), (3,4,0.8,1)
    expected_subject = [1, 2, 2, 2, 3, 3, 3, 3]
    expected_interval = [1, 1, 2, 3, 1, 2, 3, 4]
    expected_E = [0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.8]
    expected_y = [1, 0, 0, 0, 0, 0, 0, 1]

    @test aug.subject == expected_subject
    @test aug.interval == expected_interval
    @test aug.E ≈ expected_E
    @test aug.y == expected_y
end

@testset "inla_coxph: defaults / breakpoint construction" begin
    Random.seed!(11)
    n = 100
    time = rand(n) .* 5 .+ 0.01
    event = rand(0:1, n)

    aug = inla_coxph(time, event)
    @test aug.breakpoints[1] == 0.0
    @test aug.breakpoints[end] ≥ maximum(time)
    @test issorted(aug.breakpoints)
    @test aug.n_intervals ≥ 2
    # Total exposure = sum of times.
    @test sum(aug.E)≈sum(time) atol=1e-10
end

@testset "inla_coxph: input validation" begin
    @test_throws ArgumentError inla_coxph(Float64[], Int[])
    @test_throws DimensionMismatch inla_coxph([1.0, 2.0], [1])
    @test_throws ArgumentError inla_coxph([0.0, 1.0], [1, 1])     # time > 0
    @test_throws ArgumentError inla_coxph([1.0, 2.0], [2, 1])     # bad event
    @test_throws ArgumentError inla_coxph([1.0], [1]; nbreakpoints=1)
    @test_throws ArgumentError inla_coxph([1.0], [1];
        breakpoints=[0.0, 0.5])  # bp[end] < max(time)
    @test_throws ArgumentError inla_coxph([1.0], [1];
        breakpoints=[0.5, 1.0])  # bp[1] != 0
    @test_throws ArgumentError inla_coxph([1.0], [1];
        breakpoints=[1.0, 0.0])  # not sorted
end

@testset "coxph_design: structure" begin
    time = [0.5, 2.5, 3.8]
    event = [1, 0, 1]
    bp = [0.0, 1.0, 2.0, 3.0, 4.0]
    aug = inla_coxph(time, event; breakpoints=bp)

    # Baseline-only design.
    B = coxph_design(aug)
    @test size(B) == (length(aug.y), aug.n_intervals)
    @test sum(B; dims=2) == ones(length(aug.y), 1)  # row sums = 1
    @test B isa SparseMatrixCSC
    # Each row has a 1 at column = interval index.
    for r in 1:length(aug.y)
        @test B[r, aug.interval[r]] == 1
    end

    # With covariates.
    X = [10.0 100.0; 20.0 200.0; 30.0 300.0]
    A = coxph_design(aug, X)
    @test size(A) == (length(aug.y), aug.n_intervals + size(X, 2))
    # Baseline block is identical.
    @test A[:, 1:(aug.n_intervals)] == B
    # Covariate block: row r has X[subject[r], :].
    Xrep = A[:, (aug.n_intervals + 1):end]
    for r in 1:length(aug.y)
        @test Xrep[r, :] == X[aug.subject[r], :]
    end

    # Wrong row count rejected.
    @test_throws DimensionMismatch coxph_design(aug, ones(2, 2))
end

# -----------------------------------------------------------------------
# Algebraic equivalence: augmented Poisson log-density matches the
# piecewise-exponential survival log-density (up to η-independent terms).
#
# For one subject with event at time t in interval k_last:
#   log L_PWexp = -∑_{k<k_last} λ_k Δ_k - λ_{k_last}(t - τ_{k_last-1})
#                 + δ · log λ_{k_last}
# Augmented Poisson with E_ik = exposure, y_ik = (k=k_last)·δ, λ = E_ik exp(η_k):
#   log L_Pois  = ∑_k [y_ik (log E_ik + η_k) - E_ik exp(η_k) - log y_ik!]
# Matching: -E_ik exp(η_k) sums to -∑ λ_k Δ_k; y_ik (log E_ik + η_k)
# delivers δ (log E_{k_last,i} + η_{k_last}). The residual term
# (δ log E_{k_last,i}) is η-independent and so doesn't affect the mode
# of x | y. We verify by direct comparison.
# -----------------------------------------------------------------------

@testset "Algebraic equivalence: augmented Poisson ↔ piecewise-exp" begin
    Random.seed!(2026)
    n = 30
    bp = [0.0, 0.7, 1.4, 2.1, 2.8, 3.5]
    K = length(bp) - 1
    time = clamp.(rand(n) .* 3.4 .+ 0.05, 0.05, 3.45)
    event = rand(0:1, n)

    aug = inla_coxph(time, event; breakpoints=bp)
    γ = randn(K)        # baseline log-hazards
    η = γ[aug.interval] # piecewise-constant linear predictor (no covariates)

    ℓ = PoissonLikelihood(E=aug.E)
    logL_pois = log_density(ℓ, aug.y, η, Float64[])

    # Independent piecewise-exponential log-density:
    logL_pwe = 0.0
    for i in 1:n
        t_i = time[i]
        δ_i = event[i]
        k_last = clamp(searchsortedfirst(bp, t_i) - 1, 1, K)
        for k in 1:k_last
            Δ_ik = min(t_i, bp[k + 1]) - bp[k]
            logL_pwe -= exp(γ[k]) * Δ_ik
        end
        if δ_i == 1
            logL_pwe += γ[k_last]
        end
    end

    # Difference is the η-independent ∑ y_ik log E_ik - log y_ik! term.
    # Subject i contributes log E_{k_last,i} when δ_i = 1, else 0.
    # Plus -log y_ik! = 0 since y_ik ∈ {0,1}.
    aux = 0.0
    for i in 1:n
        if event[i] == 1
            t_i = time[i]
            k_last = clamp(searchsortedfirst(bp, t_i) - 1, 1, K)
            E_last_i = min(t_i, bp[k_last + 1]) - bp[k_last]
            aux += log(E_last_i)
        end
    end

    @test logL_pois≈logL_pwe + aux atol=1e-9
end

# Verify the equivalence holds when we add covariate effects.
@testset "Equivalence with covariates" begin
    Random.seed!(7)
    n = 25
    bp = [0.0, 1.0, 2.0, 3.0]
    K = length(bp) - 1
    time = clamp.(rand(n) .* 2.9 .+ 0.05, 0.05, 2.95)
    event = rand(0:1, n)
    X = randn(n, 2)
    β = [0.4, -0.2]
    γ = randn(K)

    aug = inla_coxph(time, event; breakpoints=bp)
    A = coxph_design(aug, X)
    η = A * vcat(γ, β)

    ℓ = PoissonLikelihood(E=aug.E)
    logL_pois = log_density(ℓ, aug.y, η, Float64[])

    # Direct piecewise-exponential.
    logL_pwe = 0.0
    for i in 1:n
        t_i = time[i]
        δ_i = event[i]
        η_i_cov = X[i, :] ⋅ β   # subject-specific shift
        k_last = clamp(searchsortedfirst(bp, t_i) - 1, 1, K)
        for k in 1:k_last
            Δ_ik = min(t_i, bp[k + 1]) - bp[k]
            logL_pwe -= exp(γ[k] + η_i_cov) * Δ_ik
        end
        if δ_i == 1
            logL_pwe += γ[k_last] + η_i_cov
        end
    end

    aux = 0.0
    for i in 1:n
        if event[i] == 1
            t_i = time[i]
            k_last = clamp(searchsortedfirst(bp, t_i) - 1, 1, K)
            aux += log(min(t_i, bp[k_last + 1]) - bp[k_last])
        end
    end
    @test logL_pois≈logL_pwe + aux atol=1e-9
end

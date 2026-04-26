# Temporal — Tokyo rainfall (cyclic RW2 + Bernoulli)

A canonical R-INLA temporal example: for each calendar day `i = 1, …,
366`, model the probability that rainfall exceeded a threshold in
Tokyo, with a *cyclic* random-walk smoother across the day-of-year. The
Tokyo dataset itself ships with R-INLA; here we use a synthetic
day-of-year series with the same shape so the vignette is
self-contained. Swapping in the real fixture is mechanical: replace `y`
and `n_trials` with the loaded fixture vectors.

The model is

```math
\begin{aligned}
y_i &\sim \mathrm{Binomial}(N_i, p_i), \\
\mathrm{logit}(p_i) &= \eta_i = b_i, \\
b &\sim \text{cyclic RW2}(\tau),
\end{aligned}
```

with a PC prior on `τ`. Cyclic RW2 (`cyclic = true`) penalises second
differences across the wrap-around `b_366 → b_1 → b_2`, so the smoother
treats day 366 and day 1 as adjacent.

## Synthetic day-of-year series

```@example tokyo
using Random, SparseArrays, LinearAlgebra
using LatentGaussianModels

rng = MersenneTwister(20260426)
n   = 366

# Smooth seasonal logit-probability with a winter-dry, summer-wet shape.
days  = 1:n
phase = 2π .* (days .- 1) ./ n
true_eta = -1.5 .+ 0.9 .* sin.(phase .- π/2) .+ 0.2 .* cos.(2 .* phase)
p_true   = @. 1 / (1 + exp(-true_eta))

# Two replicate years per day → N_i = 2.
n_trials = fill(2, n)
y = [rand(rng) < p_true[i] ? rand(rng) < p_true[i] ? 2 : 1 : rand(rng) < p_true[i] ? 1 : 0 for i in 1:n]
nothing # hide
```

## Building the model

The latent vector is just the cyclic RW2 effect: `x = b ∈ ℝ^n`. The
projector is the identity since `η_i = b_i`.

```@example tokyo
b_cyc = RW2(n; cyclic = true, hyperprior = PCPrecision(1.0, 0.01))
ℓ     = BinomialLikelihood(n_trials)

A = sparse(Matrix{Float64}(I, n, n))

model = LatentGaussianModel(ℓ, (b_cyc,), A)
nothing # hide
```

## Fitting

```@example tokyo
res = inla(model, y; int_strategy = :grid)
nothing # hide
```

## Posterior summaries

```@example tokyo
hyperparameters(model, res)
```

```@example tokyo
log_marginal_likelihood(res)
```

The fitted seasonal probabilities are the inverse-logit of the latent
mode. The mean and 95% credible interval per day live inside `res`:

```@example tokyo
b̂   = res.x_mean
p̂   = @. 1 / (1 + exp(-b̂))
extrema(p̂)
```

## Why the cyclic flag matters

Without `cyclic = true`, the RW2 penalty would not connect day 366 to
day 1, and the fitted smoother would be free to leave a discontinuity at
the year boundary. The PC-prior strength is identical; only the
neighbour structure changes. R-INLA's `model = "rw2", cyclic = TRUE`
maps to the same precision matrix up to the constraint
parameterisation.

For the canonical R-INLA Tokyo dataset, the corresponding oracle
fixture is on the v0.2 roadmap; this vignette provides the structural
template.

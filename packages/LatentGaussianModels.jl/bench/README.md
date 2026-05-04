# LatentGaussianModels.jl — package-level benchmarks

In-package benchmarks tracking the wall-clock cost of feature paths
that are sensitive to per-coordinate Newton refits or
factorisation-heavy inner loops. Distinct from the top-level
[`bench/`](../../../bench/) harness, which is the **end-to-end** Julia
vs R-INLA parity benchmark.

## Scripts

| Script               | Tracks                                                                                                        | Sanity bound                                 |
|---                   |---                                                                                                            |---                                           |
| `full_laplace.jl`    | `posterior_marginal_x` per-coordinate time: `FullLaplace` vs `SimplifiedLaplace` on a Poisson + Generic0 fit. | Wall-clock < 5 s (guards against blow-up).   |

### Why the FullLaplace / SimplifiedLaplace ratio is large

Phase L PR-4 ships warm-start Newton + adaptive truncation; the
measured ratio at `n = 40, grid = 51, n_θ = 5` is ~40-50×. The
Phase L plan's aspirational ≤ 5× target turned out to be structurally
infeasible without a major architectural change:

- `SimplifiedLaplace` is *closed-form per grid point* — an Edgeworth
  third-cumulant correction to the per-θ Gaussian. Per-evaluation
  cost is ~µs (a few multiplications + an `exp`).
- `FullLaplace` refits a *constrained Laplace* per grid point. The
  irreducible per-fit cost — even with rank-1 CHOLMOD updates and
  per-`(θ, i)` caching — is one numeric Cholesky update + 2-3 sparse
  triangular solves for the kriging projection and the
  Marriott-Van Loan log-determinant: ~30-100 µs at `n = 40`.

So the ratio is fundamentally `O(t_Cholesky / t_arithmetic)`, around
30-100× depending on `n` and grid size. Future perf work
(rank-1 CHOLMOD update, per-`(θ, i)` caching) can compress the
constant by ~2-3× but cannot close it to single digits. The acceptance
gate for the FullLaplace strategy is correctness against the Brunei
oracle (PR-5), not the per-coordinate timing ratio.

## Running

From the package root:

```bash
julia --project=. bench/full_laplace.jl
```

The first run pays the JIT cost; subsequent runs in the same Julia
session show a clean ratio. The corresponding regression test
(`test/regression/test_full_laplace_perf.jl`) runs a smaller version
of the same problem and asserts the ratio bound in CI.

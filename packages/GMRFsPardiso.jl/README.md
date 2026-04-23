# GMRFsPardiso.jl

MKL Pardiso / Panua Pardiso factorization backend for
[`GMRFs.jl`](../GMRFs.jl/).

Pardiso is typically faster than CHOLMOD for symmetric-indefinite and
very large sparse systems. R-INLA uses it as an optional backend and on
Linux-MKL systems it outperforms CHOLMOD on the hot inner-Newton loop
by a factor of 1.5–3×.

## Status

Planning. Activates once `GMRFs.jl` M5 (factorization backends) lands.

## Quick example

```julia
using GMRFs, GMRFsPardiso
using LinearSolve: MKLPardisoFactorize

gmrf = BesagGMRF(W; τ = 1.0)
cache = FactorCache(gmrf; factorization = MKLPardisoFactorize())
# ... use cache as normal
```

## Why a separate package, not a weakdep of GMRFs

- **License-gated.** Pardiso has two flavors — MKL Pardiso (comes with
  MKL) and Panua Pardiso (paid license). A weakdep that silently fails
  to load because of a missing license file is a bad user experience.
- **Heavy binary artifact dependencies.** `MKL_jll` is hundreds of MB.
  Users who don't want Pardiso should not be asked by Pkg whether to
  install it.
- **Explicit opt-in.** Running `using GMRFsPardiso` is a clearer signal
  of intent than "I happen to have Pardiso in my project."

## See also

- [`GMRFs.jl`](../GMRFs.jl/) — the host.
- [`LinearSolve.jl`](https://github.com/SciML/LinearSolve.jl)
  `MKLPardisoFactorize` / `PanuaPardisoFactorize` — the actual
  dispatch this package enables for GMRFs.

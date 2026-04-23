# GMRFsPardiso.jl — package plan

## Goal

Pardiso factorization backend for `GMRFs.jl`. Not a standalone package;
only useful when GMRFs.jl is in the environment.

## Module layout

```
src/
├── GMRFsPardiso.jl        # main module
├── mkl_factor.jl          # MKLPardisoFactorize specialization
├── panua_factor.jl        # PanuaPardisoFactorize specialization
└── iparm.jl               # Pardiso iparm tuning for GMRF-shaped systems

test/
├── runtests.jl
├── regression/
│   ├── test_factorize_solve.jl         # agree with CHOLMOD
│   └── test_symbolic_reuse.jl          # across multiple θ updates
└── benchmarks/
    └── bench_vs_cholmod.jl
```

## Milestones

### M1 — MKL Pardiso path (2 weeks, after GMRFs.jl M5 lands)

- [ ] `FactorCache` specialization on `MKLPardisoFactorize`.
- [ ] Symbolic factorization reuse across θ updates.
- [ ] Tuned iparm for symmetric positive-definite GMRF Q.
- [ ] Test: solutions agree with CHOLMOD path to tight tolerance.

### M2 — Panua Pardiso path (1 week)

- [ ] Same as M1 for `PanuaPardisoFactorize`.
- [ ] License-detection error path with a clear message.

### M3 — Benchmark comparison (ongoing)

- [ ] Comparative timing vs CHOLMOD on 10³, 10⁴, 10⁵ nodes.
- [ ] Record numbers per tagged release.

## Risk items

- **Pardiso.jl upstream instability.** The Julia Pardiso wrapper has
  had periods of disrepair. If it becomes unmaintained, defer this
  sub-package and document CHOLMOD as the only backend.
- **License detection.** The failure mode for a missing Panua license
  is easy to misdiagnose. Tests must cover the "no license" path and
  assert a clear error message.

## Out of scope

- Iterative solvers. GMRF Q is sparse symmetric positive (semi)-definite;
  direct factorization is the right default. Iterative solvers are a
  separate question for 10⁶+ SPDE meshes — not this package.

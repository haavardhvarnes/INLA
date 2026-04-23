# Guidance for Claude Code in GMRFsPardiso.jl

Extends [`/CLAUDE.md`](../../CLAUDE.md). Scoped to the Pardiso backend.

## Scope

This package owns:
- Specializations of `GMRFs.FactorCache` on `Pardiso.PardisoSolver` /
  the `LinearSolve.MKLPardisoFactorize()` and `PanuaPardisoFactorize()`
  algorithm types.
- Any numerical-analysis tweaks (iparm settings, symmetric-indefinite
  vs positive-definite flags) that make Pardiso faster on GMRF-shaped
  systems.

Out of scope:
- Generic sparse-linear-algebra abstractions. LinearSolve.jl owns those.
- License handling. That's Pardiso.jl's problem.

## Dependencies allowed

Core:
- `GMRFs` — the host whose types we specialize.
- `LinearSolve` — algorithm type dispatch.
- `Pardiso` — the actual Pardiso wrapper.

Nothing else without an ADR.

## Testing

- `test/regression/` — factorize + solve a known system with Pardiso,
  compare result against the CHOLMOD path from GMRFs.jl to tight tol.
- `test/benchmarks/` — comparative timing vs CHOLMOD on representative
  GMRF sizes (10³, 10⁴, 10⁵ nodes). Informational only.

## Style

- Pardiso iparm tuning is arcane. Every non-default iparm setting must
  have a comment citing the Pardiso user manual section that justifies it.
- No silent fallback to CHOLMOD. If Pardiso fails to initialize, raise
  a clear error that names the license / environment issue.

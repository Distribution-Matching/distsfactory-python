# Changelog

## 0.1.0 — 2026-05-21

First public release. Python port of [DistributionsFactories.jl](https://github.com/Distribution-Matching/DistributionsFactories.jl) (Julia is the parameterization master).

### Coverage

- **27 distribution families** across real, positive, unit, and integer supports.
- **Specification styles**: moments (`mean`/`var`/`std`/`cv`/`scv`/`second_moment`), quantiles (`median`/`q1`/`q3`/`iqr`/`quantiles`), mode (combined with moments or quantiles).
- **`support=`** kwarg places a distribution on an arbitrary interval via affine transform or truncation. Returns native scipy `truncnorm` / `truncexpon` / `truncpareto` / `truncweibull_min` when available, otherwise a thin wrapper.
- **`partial_dist`** pins some scipy parameters and solves the rest from moment constraints (1D brentq / 2D damped Newton).
- **Feasibility predicates**: `dist_exists`, `available_distributions`. Includes the Langevin envelope check for truncated location-scale families (Normal, Laplace, Logistic).

### Testing

- 574 self-contained tests.
- Cross-package oracle in `tests/data/cross_oracle.json` (regenerable from the Julia repo) provides an extra numerical-parity layer.

### Known residuals

See [README "Known residuals"](README.md#known-residuals) — five bugs inherited from the Julia master, each tracked under the upstream issue.

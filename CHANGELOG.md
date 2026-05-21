# Changelog

## 0.2.0 — 2026-05-21

### Public `.parent` accessor on the truncation / flip wrappers

The fallback wrapper classes `_TruncatedDist` and `_FlippedDist` (used when scipy
doesn't ship a native truncated form, e.g. truncated Laplace) now expose the
underlying distribution as `.parent`. The older `._inner` attribute is preserved
as a property alias so any existing code keeps working.

```python
d = make_dist("laplace", mean=2.0, var=4.0, support=(0.0, math.inf))
d.mean()        # 2.0 — the truncated dist's actual mean
d.parent        # the un-truncated Laplace (frozen scipy distribution)
d.parent.kwds   # {'loc': ..., 'scale': ...}
d._inner        # same as d.parent — kept for back-compat
```

`_TruncatedDist` also gains `.support_interval`, returning `(lo, hi)` as a tuple
(mirrors the data the existing `.support()` method already returned).

This matches the R sibling package's 0.2.0 `$parent` / `$support` accessor surface.
Scipy-native truncated distributions (e.g. truncnorm, truncexpon) were already
clean — they use `(a, b, loc, scale)` kwarg names that don't collide with
`mean`/`var`/`std`/`median` method names, so this change only affects the
fallback wrappers.

### Other

- 8 new regression tests in `tests/test_wrapper_accessors.py`. 582 tests pass
  (was 574).

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

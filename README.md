# distsfactory

A Python package for constructing probability distributions from partial specifications — moments, quantiles, mode, and support.

Part of the DistributionsFactories family (alongside [DistributionsFactories.jl](https://github.com/Distribution-Matching/DistributionsFactories.jl) for Julia and [distsfactory-r](https://github.com/yoninazarathy/distsfactory-r)). The Julia package is the parameterization master; this Python port mirrors its behaviour and is cross-validated against it.

## Design

- **Built on `scipy.stats`** — returns frozen scipy distribution objects (or scipy-compatible wrappers) that work with the rest of the scientific Python ecosystem.
- Specify what you know (mean, variance, quantiles, mode, support) and get back a ready-to-use distribution.
- Accepts distribution names as strings or `scipy.stats` distribution objects directly.

## Quick start

```python
from distsfactory import make_dist

# Construct from moments — returns a frozen scipy.stats distribution
d = make_dist("gamma", mean=5, var=3)
type(d)            # scipy.stats._distn_infrastructure.rv_continuous_frozen

d.pdf(2)           # density
d.cdf(0.95)        # CDF
d.ppf(0.5)         # quantile (percent point function)
d.rvs(100)         # random samples
d.mean()           # 5.0
d.var()            # 3.0

# scipy.stats objects work too
import scipy.stats as st
d = make_dist(st.gamma, mean=5, var=3)

# Truncated Normal on [-1, 4] with mean 1 and standard deviation 0.8
d = make_dist("normal", mean=1.0, std=0.8, support=(-1, 4))
type(d)                    # scipy.stats._distn_infrastructure.rv_continuous_frozen
                           # (a scipy.stats.truncnorm)
d.mean(), d.std()          # (1.0, 0.8)  — moments after truncation
d.kwds                     # {'a': -2.408, 'b': 3.666,
                           #  'loc': 0.9822, 'scale': 0.8232}
                           # — parent Normal(μ, σ) solved so the truncated
                           #   distribution hits the requested (mean, std)
```

Whenever scipy provides a native truncated form (`truncnorm`, `truncexpon`, …) we return that. For families where scipy doesn't (`Truncated{Laplace}`, `Truncated{Gamma}`, …) the return is a `distsfactory._support._TruncatedDist` wrapper with the same `pdf`/`cdf`/`ppf`/`mean`/`var`/`rvs` surface.

## Supported distributions

27 families across continuous (real, positive, unit) and discrete supports. Each family supports a subset of specification types — `mean+var` is universal; quantile- and mode-based forms are implemented where they exist in the Julia package.

### Continuous on `(-∞, ∞)`

| Distribution | Support | Free params | Methods |
|---|---|---|---|
| Normal | `(-∞, ∞)` | 2 | mean+var, q1+q3, mode+var, mode+iqr |
| Student's T | `(-∞, ∞)` | 1 (+2 via `partial_dist`) | mean+var (μ=0); arbitrary via `partial_dist("tdist", df=ν)` |
| Cauchy | `(-∞, ∞)` | 2 | two quantiles (moments undefined) |
| Laplace | `(-∞, ∞)` | 2 | mean+var, two quantiles, mode+iqr |
| Logistic | `(-∞, ∞)` | 2 | mean+var, two quantiles, mode+iqr, mean+quantile |
| Gumbel | `(-∞, ∞)` | 2 | mean+var, two quantiles |
| Uniform | `(-∞, ∞)` | 2 | mean+var |
| Symmetric Triangular | `(-∞, ∞)` | 2 | mean+var |
| Triangular (asymmetric) | `(-∞, ∞)` | 3 | mean+var+mode |

### Continuous on `[0, ∞)`

| Distribution | Free params | Methods |
|---|---|---|
| Gamma | 2 | mean+var, mean+mode, mode+var, mode+iqr, mode+quantile, two quantiles, mean+quantile |
| Erlang | 2 | mean+var (k rounded — see [#4](https://github.com/Distribution-Matching/distsfactory-python/issues/4)) |
| Exponential | 1 | mean, var, single quantile, mean+var |
| Chi-squared | 1 | mean+var, mean alone, var alone |
| Chi | 2 | mean+var |
| Rayleigh | 1 | mean, var, mode, single quantile, mean+var |
| Log-normal | 2 | mean+var, two quantiles, mean+quantile |
| Weibull | 2 | mean+var, two quantiles |
| Frechet | 2 | mean+var |
| F | 2 | mean+var |
| Inverse Gamma | 2 | mean+var |
| Pareto | 2 | mean+var, two quantiles |
| Folded Normal | 2 | mean+var (2D Newton) |

### Continuous on `[0, 1]`

| Distribution | Free params | Methods |
|---|---|---|
| Beta | 2 | mean+var, mean+mode, two quantiles, mean+quantile |

### Discrete

| Distribution | Support | Methods |
|---|---|---|
| Binomial | `{0, …, n}` | mean+var |
| Discrete Uniform | `{a, …, b}` | mean+var |
| Discrete Symmetric Triangular | `{μ-n, …, μ+n}` | mean+var |
| Discrete Triangular | `{a, …, b}` (mode at c) | mean+var+mode (approximate) |
| Poisson | `{0, 1, 2, …}` | mean, var, mean+var |
| Negative Binomial | `{0, 1, 2, …}` | mean+var |
| Geometric | `{0, 1, 2, …}` | mean, var, single quantile, mean+var |

## Specification styles

```python
# Moment-based
make_dist("gamma", mean=5, var=3)
make_dist("gamma", mean=5, std=2)          # std -> var
make_dist("gamma", mean=5, cv=0.5)         # coefficient of variation
make_dist("gamma", mean=4, scv=0.5)        # squared CV
make_dist("gamma", mean=5, second_moment=28)  # E[X²] -> var
make_dist("exponential", mean=3)           # 1-parameter family

# Quantile-based
make_dist("exponential", median=2.0)
make_dist("logistic", q1=2, q3=8)
make_dist("normal", q1=-1, q3=1)
make_dist("gamma", quantiles=[(0.1, 1.0), (0.9, 10.0)])
make_dist("beta", mean=0.4, median=0.38)
make_dist("normal", median=5, iqr=2)

# Mode-based
make_dist("rayleigh", mode=2)
make_dist("gamma", mean=5, mode=3)
make_dist("beta", mean=0.4, mode=0.35)
make_dist("gamma", mode=3, iqr=4)
make_dist("normal", mode=3, var=4)
make_dist("logistic", mode=5, iqr=4)

# 3-parameter triangular (mean + var + mode)
make_dist("triangular", mean=5, var=2, mode=4)
make_dist("discrete_triangular", mean=5, var=2, mode=5)
```

## Support — affine transforms and truncation

The `support=` keyword places a distribution on an arbitrary support. The package chooses between an affine transform (when the requested support has the same shape as the natural one) and truncation (when it's strictly contained).

```python
import math

# Affine shift — Gamma on [3, ∞)
make_dist("gamma", mean=8, var=3, support=(3, math.inf))

# Affine flip — Gamma on (-∞, 10]
make_dist("gamma", mean=5, var=3, support=(-math.inf, 10))

# Affine scale — Beta on [2, 7]
make_dist("beta", mean=3.5, var=0.5, support=(2, 7))

# Truncation — Normal on [-0.5, 0.5] with moment matching (2D Newton)
make_dist("normal", mean=0.1, var=0.05, support=(-0.5, 0.5))

# Truncation — Gamma/Beta with actual moment matching (generic 2D Newton)
make_dist("gamma", mean=3, var=1, support=(0, 10))
make_dist("beta", mean=0.5, var=0.02, support=(0.2, 0.8))

# Discrete shift — Binomial on {10, …, 15}
make_dist("binomial", mean=12, var=1.2, support=range(10, 16))

# Truncated Poisson on {2, …, 10} (var is determined by mean)
make_dist("poisson", mean=2.5, var=0.59, support=range(2, 11))
```

For location-scale Student-t, use `partial_dist` together with `support=`:

```python
from distsfactory import partial_dist, make_dist

spec = partial_dist("tdist", df=5)
# Half-truncated location-scale Student-t
d = make_dist(spec, mean=2.0, var=1.0, support=(0.0, math.inf))
```

(Two-sided `Truncated{TDist}` is not implemented yet — same gap as in the Julia package; see [#2](https://github.com/Distribution-Matching/distsfactory-python/issues/2).)

## Partial specifications — `partial_dist`

The Python analog of Julia's `@dist` macro. Pin some scipy parameters and leave the rest to be solved from moment constraints.

```python
from distsfactory import partial_dist, make_dist

# Pin Gamma's shape, solve scale from mean
spec = partial_dist("gamma", a=3.0)
d = make_dist(spec, mean=5.0)
d.kwds                # {'a': 3.0, 'scale': 1.6666…}

# Pin Gamma's shape, solve scale from variance
d = make_dist(spec, var=3.0)

# Pin Logistic's location, solve scale from variance
spec = partial_dist("logistic", loc=2.0)
d = make_dist(spec, var=22.3)

# Pin Beta's α, solve β from mean
spec = partial_dist("beta", a=2.0)
d = make_dist(spec, mean=0.4)
d.kwds                # {'a': 2.0, 'b': 3.0}

# Full instance — Student-t with df=7, solve loc/scale from (mean, var)
spec = partial_dist("tdist", df=7)
d = make_dist(spec, mean=5.0, var=2.0)
```

`partial_dist` uses the canonical (Julia-compatible) parameter set for each family. For example, `gamma` exposes `(a, scale)` — not `(a, loc, scale)` — because Julia's Gamma has only `(α, θ)`. To shift the support, use `support=`.

## Feasibility checks

```python
from distsfactory import dist_exists

dist_exists("beta", mean=0.5, var=0.1)         # True
dist_exists("beta", mean=0.5, var=0.3)         # False (var too large)
dist_exists("exponential", mean=2.5, var=6.25) # True (var == mean²)
dist_exists("exponential", mean=2.5, var=1.5)  # False
dist_exists("tdist", mean=1, var=2)            # False (TDist requires mean=0)
```

When `make_dist` fails for the same input, it raises `ValueError` carrying the same reason string.

## Discovery

```python
from distsfactory import available_distributions
import math

# All distributions feasible for these moments
available_distributions(mean=5, var=3)
# ['normal', 'laplace', 'logistic', 'gumbel', 'uniform', 'sym_triangular',
#  'gamma', 'erlang', 'lognormal', 'weibull', 'frechet', 'inverse_gamma',
#  'pareto', 'folded_normal']

# Filter by natural support: tuple, range, or category string
available_distributions(support="positive")
available_distributions(support=(0, math.inf), mean=5, var=3)
available_distributions(support=(0, 1), mean=0.5, var=0.05)  # -> ['beta']
available_distributions(support="integer_nonneg")
```

## Testing

The test suite is self-contained — `pytest` covers the package end-to-end with no Julia install required.

One additional file, `tests/test_cross_julia.py`, reads a checked-in JSON oracle (`tests/data/cross_oracle.json`) of reference values generated by the Julia package. It catches any cross-language numerical drift. Regenerate the oracle after material changes with:

```
julia --project=../DistributionsFactories.jl scripts/build_cross_oracle.jl
```

## Installation

Not yet published. Development install:

```
pip install -e ".[dev]"
```

## Known residuals (parity with Julia)

The Julia package is the parameterization master. Inherited issues are tracked locally with a link to the upstream Julia issue:

- [#1](https://github.com/Distribution-Matching/distsfactory-python/issues/1) — Gamma `from_mean_mode` accepts negative mode (bug; mirrors [Julia #7](https://github.com/Distribution-Matching/DistributionsFactories.jl/issues/7))
- [#2](https://github.com/Distribution-Matching/distsfactory-python/issues/2) — two-sided `Truncated{TDist}` not implemented (mirrors [Julia #1](https://github.com/Distribution-Matching/DistributionsFactories.jl/issues/1))
- [#3](https://github.com/Distribution-Matching/distsfactory-python/issues/3) — `parse_spec` silently drops `std` when both `var` and `std` passed (mirrors [Julia #10](https://github.com/Distribution-Matching/DistributionsFactories.jl/issues/10))
- [#4](https://github.com/Distribution-Matching/distsfactory-python/issues/4) — Erlang is approximate but undocumented (mirrors [Julia #11](https://github.com/Distribution-Matching/DistributionsFactories.jl/issues/11))
- [#5](https://github.com/Distribution-Matching/distsfactory-python/issues/5) — Frechet single-point start at `CV²==1` (mirrors [Julia #12](https://github.com/Distribution-Matching/DistributionsFactories.jl/issues/12))

## Authors

Ron Ashri, Sarat Moka, Yoni Nazarathy

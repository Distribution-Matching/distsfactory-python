# distsfactory

A Python package for constructing probability distributions from partial specifications — moments, quantiles, mode, and more.

Part of the DistributionsFactories family (alongside [DistributionsFactories.jl](https://github.com/Ron-Ash/DistributionsFactories.jl) for Julia and [distsfactory-r](https://github.com/yoninazarathy/distsfactory-r)).

## Design

- **Built on `scipy.stats`** — returns frozen scipy distribution objects that work with the rest of the scientific Python ecosystem
- Specify what you know (mean, variance, quantiles, mode) and get back a ready-to-use distribution
- Accepts distribution names as strings or `scipy.stats` distribution objects directly

## Quick start

```python
from distsfactory import make_dist

# Construct from moments — returns a frozen scipy.stats distribution
d = make_dist("gamma", mean=5, var=3)

d.pdf(2)          # density
d.cdf(0.95)       # CDF
d.ppf(0.5)        # quantile (percent point function)
d.rvs(100)        # random samples
d.mean()          # 5.0
d.var()            # 3.0

# Also accepts scipy.stats objects directly
import scipy.stats as st
d = make_dist(st.gamma, mean=5, var=3)
```

## Supported distributions

| Distribution | Supported specifications |
|---|---|
| **Gamma** | mean+var, mean+mode, mode+var, mode+quantile, mode+iqr, two quantiles, mean+quantile |
| **Exponential** | mean, mean+var, var, single quantile |
| **Logistic** | mean+var, two quantiles, mode+iqr, mean+quantile |
| **Beta** | mean+var, mean+mode, two quantiles, mean+quantile |

## Specification styles

```python
# Moment-based
make_dist("gamma", mean=5, var=3)
make_dist("gamma", mean=5, std=2)          # std -> var
make_dist("gamma", mean=5, cv=0.5)         # coefficient of variation
make_dist("gamma", mean=4, scv=0.5)        # squared CV
make_dist("exponential", mean=3)           # 1-parameter family

# Quantile-based
make_dist("exponential", median=2.0)
make_dist("logistic", q1=2, q3=8)
make_dist("gamma", quantiles=[(0.1, 1.0), (0.9, 10.0)])
make_dist("beta", mean=0.4, median=0.38)

# Mode-based
make_dist("gamma", mean=5, mode=3)
make_dist("beta", mean=0.4, mode=0.35)
make_dist("gamma", mode=3, iqr=4)
make_dist("logistic", mode=5, iqr=4)
```

## Feasibility checks

```python
from distsfactory import dist_exists

dist_exists("beta", mean=0.5, var=0.1)        # True
dist_exists("beta", mean=0.5, var=0.3)        # False (var too large)
dist_exists("exponential", mean=2.5, var=6.25) # True (var == mean^2)
dist_exists("exponential", mean=2.5, var=1.5)  # False
```

## Discovery

```python
from distsfactory import available_distributions

available_distributions(mean=5, var=3)
# ['gamma', 'logistic']

available_distributions(mean=5, var=25)
# ['gamma', 'exponential', 'logistic']

available_distributions(mean=0.5, var=0.05)
# ['gamma', 'logistic', 'beta']
```

## Installation

Not yet published. Development in progress.

```
pip install -e ".[dev]"    # for development
```

## Authors

Ron Ashri, Sarat Moka, Yoni Nazarathy

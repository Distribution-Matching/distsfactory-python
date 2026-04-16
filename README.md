# distsfactory

A Python package for constructing probability distributions from partial specifications — moments, quantiles, mode, and more.

Part of the DistributionsFactories family (alongside [DistributionsFactories.jl](https://github.com/Ron-Ash/DistributionsFactories.jl) for Julia and [distsfactory-r](https://github.com/yoninazarathy/distsfactory-r)).

## Design

- **Built on `scipy.stats`** — returns frozen scipy distribution objects that work with the rest of the scientific Python ecosystem
- Specify what you know (mean, variance, quantiles, mode) and get back a ready-to-use distribution
- Accepts distribution names as strings or `scipy.stats` distribution objects directly

## Planned interface

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

# Various specification styles
make_dist("normal", mean=10, std=2)
make_dist("normal", q1=10, q3=30)
make_dist("beta", mean=0.4, median=0.35)
make_dist("exponential", mean=3)
make_dist("gamma", mean=5, cv=0.5)

# Discovery: which distributions fit these constraints?
from distsfactory import available_distributions
available_distributions(mean=5, var=3)
```

## Installation

Not yet published. Development in progress.

```
pip install distsfactory
```

## Authors

Ron Ashri, Sarat Moka, Yoni Nazarathy

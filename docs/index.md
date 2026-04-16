# distsfactory

Construct probability distributions from partial specifications — moments, quantiles, mode, and more.

## Overview

`distsfactory` lets you create `scipy.stats` distributions by specifying what you *know* about the distribution (its mean, variance, quantiles, mode, etc.) rather than its raw parameters. The package solves for the parameters that match your constraints.

```python
from distsfactory import make_dist

# "I need a Gamma with mean 5 and variance 3"
d = make_dist("gamma", mean=5, var=3)
d.mean()   # 5.0
d.var()    # 3.0
d.rvs(10)  # 10 random samples
```

## Core functions

### `make_dist(dist, **kwargs)`

The main entry point. Takes a distribution name (or `scipy.stats` object) and keyword constraints, returns a frozen `scipy.stats` distribution.

::: distsfactory._api.make_dist

### `dist_exists(dist, **kwargs)`

Check whether a distribution can be constructed with the given constraints.

::: distsfactory._api.dist_exists

### `available_distributions(**kwargs)`

List all distributions feasible for the given constraints.

::: distsfactory._api.available_distributions

## Supported distributions

| Distribution | Specs |
|---|---|
| Gamma | mean+var, mean+mode, mode+var, mode+quantile, mode+iqr, two quantiles, mean+quantile |
| Exponential | mean, mean+var, var, single quantile |
| Logistic | mean+var, two quantiles, mode+iqr, mean+quantile |
| Beta | mean+var, mean+mode, two quantiles, mean+quantile |

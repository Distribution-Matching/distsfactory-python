"""Round-trip tests: instantiate a distribution, extract (mean, var), refit, compare.

Mirrors the per-distribution round-trip tests in DistributionsFactories.jl/test/
(e.g. `test_mean_var_Beta`). For each known-good parameter grid, build the
scipy frozen dist directly, then use `make_dist` with the resulting moments
and check the reconstructed dist's params match the original.

Self-contained — no Julia dependency.
"""

import math
import numpy as np
import pytest
from scipy import stats
from distsfactory import make_dist


def _close(a, b, rel=1e-6, abs_tol=1e-9):
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0, 3.5, 7.0])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0, 3.5, 7.0])
def test_beta_roundtrip(alpha, beta):
    true = stats.beta(a=alpha, b=beta)
    d = make_dist("beta", mean=true.mean(), var=true.var())
    assert _close(d.kwds["a"], alpha)
    assert _close(d.kwds["b"], beta)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.5, 5.0, 10.0])
@pytest.mark.parametrize("theta", [0.5, 1.0, 2.0, 5.0])
def test_gamma_roundtrip(alpha, theta):
    true = stats.gamma(a=alpha, scale=theta)
    d = make_dist("gamma", mean=true.mean(), var=true.var())
    assert _close(d.kwds["a"], alpha)
    assert _close(d.kwds["scale"], theta)


@pytest.mark.parametrize("k", [1, 2, 3, 5, 10, 30])
def test_chisq_roundtrip(k):
    true = stats.chi2(df=k)
    d = make_dist("chisq", mean=true.mean(), var=true.var())
    assert _close(d.kwds["df"], k)


@pytest.mark.parametrize("theta", [0.1, 0.5, 1.0, 2.5, 10.0])
def test_exponential_roundtrip(theta):
    true = stats.expon(scale=theta)
    d = make_dist("exponential", mean=true.mean(), var=true.var())
    assert _close(d.kwds["scale"], theta)


@pytest.mark.parametrize("mu", [-3.0, 0.0, 2.5])
@pytest.mark.parametrize("sigma", [0.5, 1.0, 3.0])
def test_normal_roundtrip(mu, sigma):
    true = stats.norm(loc=mu, scale=sigma)
    d = make_dist("normal", mean=true.mean(), var=true.var())
    assert _close(d.kwds["loc"], mu)
    assert _close(d.kwds["scale"], sigma)


@pytest.mark.parametrize("mu", [-2.0, 0.0, 5.0])
@pytest.mark.parametrize("s", [0.5, 1.0, 3.0])
def test_logistic_roundtrip(mu, s):
    true = stats.logistic(loc=mu, scale=s)
    d = make_dist("logistic", mean=true.mean(), var=true.var())
    assert _close(d.kwds["loc"], mu)
    assert _close(d.kwds["scale"], s)


@pytest.mark.parametrize("mu", [-2.0, 0.0, 5.0])
@pytest.mark.parametrize("b", [0.5, 1.0, 3.0])
def test_laplace_roundtrip(mu, b):
    true = stats.laplace(loc=mu, scale=b)
    d = make_dist("laplace", mean=true.mean(), var=true.var())
    assert _close(d.kwds["loc"], mu)
    assert _close(d.kwds["scale"], b)


@pytest.mark.parametrize("mu_log", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("sigma_log", [0.1, 0.5, 1.0])
def test_lognormal_roundtrip(mu_log, sigma_log):
    true = stats.lognorm(s=sigma_log, scale=math.exp(mu_log))
    d = make_dist("lognormal", mean=true.mean(), var=true.var())
    assert _close(d.kwds["s"], sigma_log)
    assert _close(d.kwds["scale"], math.exp(mu_log))


@pytest.mark.parametrize("k", [0.7, 1.0, 1.5, 3.0, 5.0])
@pytest.mark.parametrize("lam", [0.5, 1.0, 3.0])
def test_weibull_roundtrip(k, lam):
    true = stats.weibull_min(c=k, scale=lam)
    m, v = true.mean(), true.var()
    if not (math.isfinite(m) and math.isfinite(v) and v > 0):
        pytest.skip("non-finite moments")
    d = make_dist("weibull", mean=m, var=v)
    assert _close(d.kwds["c"], k, rel=1e-5)
    assert _close(d.kwds["scale"], lam, rel=1e-5)


@pytest.mark.parametrize("alpha", [3.0, 4.5, 7.0])
@pytest.mark.parametrize("s", [0.5, 1.0, 3.0])
def test_frechet_roundtrip(alpha, s):
    true = stats.invweibull(c=alpha, scale=s)
    m, v = true.mean(), true.var()
    d = make_dist("frechet", mean=m, var=v)
    assert _close(d.kwds["c"], alpha, rel=1e-4)
    assert _close(d.kwds["scale"], s, rel=1e-4)


@pytest.mark.parametrize("sigma", [0.5, 1.0, 3.0, 7.0])
def test_rayleigh_roundtrip(sigma):
    true = stats.rayleigh(scale=sigma)
    d = make_dist("rayleigh", mean=true.mean(), var=true.var())
    assert _close(d.kwds["scale"], sigma)


@pytest.mark.parametrize("alpha", [3.0, 5.0, 10.0])
@pytest.mark.parametrize("beta", [0.5, 1.0, 5.0])
def test_inverse_gamma_roundtrip(alpha, beta):
    true = stats.invgamma(a=alpha, scale=beta)
    d = make_dist("inverse_gamma", mean=true.mean(), var=true.var())
    assert _close(d.kwds["a"], alpha)
    assert _close(d.kwds["scale"], beta)


@pytest.mark.parametrize("n", [1, 5, 20, 50])
@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_binomial_roundtrip(n, p):
    true = stats.binom(n=n, p=p)
    d = make_dist("binomial", mean=true.mean(), var=true.var())
    assert d.kwds["n"] == n
    assert _close(d.kwds["p"], p)


@pytest.mark.parametrize("mu", [0.1, 0.5, 1.0, 5.0, 20.0])
def test_poisson_roundtrip(mu):
    true = stats.poisson(mu=mu)
    d = make_dist("poisson", mean=true.mean(), var=true.var())
    assert _close(d.kwds["mu"], mu)


@pytest.mark.parametrize("r", [1, 2, 5, 10])
@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7])
def test_negative_binomial_roundtrip(r, p):
    true = stats.nbinom(n=r, p=p)
    d = make_dist("negative_binomial", mean=true.mean(), var=true.var())
    # NegativeBinomial has 2 free params; the formula recovers exactly.
    assert _close(d.kwds["n"], r)
    assert _close(d.kwds["p"], p)


@pytest.mark.parametrize("loc", [-3, 0, 5])
@pytest.mark.parametrize("hi_minus_lo", [3, 5, 10])
def test_discrete_uniform_roundtrip(loc, hi_minus_lo):
    high = loc + hi_minus_lo + 1  # scipy semantics: support {loc, ..., high-1}
    true = stats.randint(low=loc, high=high)
    d = make_dist("discrete_uniform", mean=true.mean(), var=true.var())
    assert _close(d.mean(), true.mean())
    assert _close(d.var(), true.var())

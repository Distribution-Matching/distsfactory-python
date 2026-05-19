"""Tests for positive-support continuous distributions."""

import math
import pytest
from distsfactory import make_dist, dist_exists


def _close(a, b, rel=1e-6):
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-12)


class TestLogNormal:
    def test_mean_var(self):
        d = make_dist("lognormal", mean=5, var=3)
        assert _close(d.mean(), 5.0)
        assert _close(d.var(), 3.0)

    def test_two_quantiles(self):
        d = make_dist("lognormal", quantiles=[(0.1, 1.0), (0.9, 10.0)])
        assert _close(d.ppf(0.1), 1.0)
        assert _close(d.ppf(0.9), 10.0)

    def test_mean_quantile(self):
        d = make_dist("lognormal", mean=3.0, median=2.5)
        assert _close(d.mean(), 3.0)
        assert _close(d.ppf(0.5), 2.5, rel=1e-5)


class TestWeibull:
    def test_mean_var(self):
        d = make_dist("weibull", mean=5, var=3)
        assert _close(d.mean(), 5.0)
        assert _close(d.var(), 3.0)

    def test_high_cv(self):
        d = make_dist("weibull", mean=2, var=10)
        assert _close(d.mean(), 2.0)
        assert _close(d.var(), 10.0)

    def test_two_quantiles(self):
        d = make_dist("weibull", quantiles=[(0.1, 1.0), (0.9, 5.0)])
        assert _close(d.ppf(0.1), 1.0)
        assert _close(d.ppf(0.9), 5.0)


class TestFrechet:
    def test_mean_var(self):
        # Frechet has heavy tail, so var/mu^2 must be small enough.
        d = make_dist("frechet", mean=5, var=3)
        assert _close(d.mean(), 5.0)
        assert _close(d.var(), 3.0, rel=1e-5)


class TestChisq:
    def test_mean_var(self):
        d = make_dist("chisq", mean=4, var=8)
        assert _close(d.mean(), 4.0)
        assert _close(d.var(), 8.0)

    def test_inconsistent_var_rejected(self):
        assert not dist_exists("chisq", mean=4, var=7)

    def test_from_mean_only(self):
        d = make_dist("chisq", mean=5)
        assert _close(d.mean(), 5.0)


class TestChi:
    def test_mean_var(self):
        # Pick a known feasible point: nu=4 has mean=2*Gamma(2.5)/Gamma(2)
        # Actually easier: pick nu, compute mean/var, then ask for them back.
        from scipy import stats
        nu = 4
        c = stats.chi(df=nu)
        mu, var = c.mean(), c.var()
        d = make_dist("chi", mean=mu, var=var)
        assert _close(d.mean(), mu, rel=1e-6)
        assert _close(d.var(), var, rel=1e-6)


class TestRayleigh:
    def test_mean_var(self):
        cv2 = (4 - math.pi) / math.pi
        d = make_dist("rayleigh", mean=5, var=5 ** 2 * cv2)
        assert _close(d.mean(), 5.0)

    def test_from_mean_only(self):
        d = make_dist("rayleigh", mean=5)
        assert _close(d.mean(), 5.0)

    def test_from_var(self):
        cv2 = (4 - math.pi) / math.pi
        d = make_dist("rayleigh", var=5 ** 2 * cv2)
        assert _close(d.mean(), 5.0, rel=1e-6)

    def test_from_mode(self):
        d = make_dist("rayleigh", mode=2.0)
        # mode = sigma, so mean = sigma * sqrt(pi/2) = 2*sqrt(pi/2)
        assert _close(d.mean(), 2.0 * math.sqrt(math.pi / 2))

    def test_from_quantile(self):
        d = make_dist("rayleigh", median=2.0)
        assert _close(d.ppf(0.5), 2.0)


class TestFDist:
    def test_mean_var(self):
        # FDist needs 1 < mean < 2. Pick mean=1.5, then var > 1.5^2*0.5/0.5 = 2.25
        d = make_dist("fdist", mean=1.5, var=10)
        assert _close(d.mean(), 1.5, rel=1e-6)
        assert _close(d.var(), 10.0, rel=1e-5)

    def test_mean_le_one_rejected(self):
        assert not dist_exists("fdist", mean=0.9, var=5)


class TestInverseGamma:
    def test_mean_var(self):
        d = make_dist("inverse_gamma", mean=5, var=3)
        assert _close(d.mean(), 5.0)
        assert _close(d.var(), 3.0)


class TestPareto:
    def test_mean_var(self):
        d = make_dist("pareto", mean=5, var=3)
        assert _close(d.mean(), 5.0)
        assert _close(d.var(), 3.0)

    def test_two_quantiles(self):
        d = make_dist("pareto", quantiles=[(0.1, 1.0), (0.9, 10.0)])
        assert _close(d.ppf(0.1), 1.0)
        assert _close(d.ppf(0.9), 10.0)


class TestFoldedNormal:
    def test_mean_var(self):
        d = make_dist("folded_normal", mean=2.5, var=1.2)
        assert _close(d.mean(), 2.5, rel=1e-6)
        assert _close(d.var(), 1.2, rel=1e-6)


class TestErlang:
    def test_mean_var(self):
        # Erlang with k=4, theta=1: mean=4, var=4
        d = make_dist("erlang", mean=4, var=4)
        assert _close(d.mean(), 4.0)
        assert _close(d.var(), 4.0)

    def test_rounds_to_integer_k(self):
        # mean^2/var ~ 3.5 -> k=4 (rounded). After that, theta = var/mean = 1.4,
        # so the resulting Gamma has mean = 4*1.4 = 5.6 (not exactly the request).
        d = make_dist("erlang", mean=5, var=2.0)
        a = d.kwds["a"]
        assert float(a).is_integer()

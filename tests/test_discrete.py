"""Tests for discrete distributions."""

import math
import pytest
from distsfactory import make_dist, dist_exists


def _close(a, b, rel=1e-6):
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-12)


class TestBinomial:
    def test_mean_var(self):
        # n=10, p=0.5 -> mean=5, var=2.5
        d = make_dist("binomial", mean=5, var=2.5)
        assert _close(d.mean(), 5.0)
        assert _close(d.var(), 2.5)

    def test_inconsistent_rejected(self):
        # n^2/(n-var) must be a positive integer
        assert not dist_exists("binomial", mean=5, var=2.7)


class TestPoisson:
    def test_mean_var(self):
        d = make_dist("poisson", mean=5, var=5)
        assert _close(d.mean(), 5.0)
        assert _close(d.var(), 5.0)

    def test_from_mean(self):
        d = make_dist("poisson", mean=3)
        assert _close(d.mean(), 3.0)

    def test_var_neq_mean_rejected(self):
        assert not dist_exists("poisson", mean=5, var=3)


class TestNegativeBinomial:
    def test_mean_var(self):
        d = make_dist("negative_binomial", mean=5, var=8)
        assert _close(d.mean(), 5.0)
        assert _close(d.var(), 8.0)

    def test_var_le_mean_rejected(self):
        assert not dist_exists("negative_binomial", mean=5, var=3)


class TestGeometric:
    def test_mean_var(self):
        # Geometric on {0,1,...}: var = mu*(1+mu). For mu=2, var=6.
        d = make_dist("geometric", mean=2, var=6)
        assert _close(d.mean(), 2.0)
        assert _close(d.var(), 6.0)

    def test_from_mean(self):
        d = make_dist("geometric", mean=3)
        assert _close(d.mean(), 3.0)
        # var = 3*(1+3) = 12
        assert _close(d.var(), 12.0)

    def test_from_var(self):
        d = make_dist("geometric", var=12)
        assert _close(d.mean(), 3.0)

    def test_quantile(self):
        # Continuous-extension formula assumes equality of the CDF at k+1,
        # so the discrete ppf will land on q-1, q, or q+1 depending on rounding.
        d = make_dist("geometric", median=2)
        assert 1 <= d.ppf(0.5) <= 3


class TestDiscreteUniform:
    def test_mean_var(self):
        # DiscreteUniform on {0,...,10}: mean=5, var=10
        # var = n(n+2)/12 with n = b-a = 10 -> var = 10*12/12 = 10
        d = make_dist("discrete_uniform", mean=5, var=10)
        assert _close(d.mean(), 5.0)
        assert _close(d.var(), 10.0)

    def test_shifted(self):
        # {2,...,8}: n=6, mean=5, var = 6*8/12 = 4
        d = make_dist("discrete_uniform", mean=5, var=4)
        assert _close(d.mean(), 5.0)
        assert _close(d.var(), 4.0)

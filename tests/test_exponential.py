"""Tests for Exponential distribution construction."""

import math
import pytest
from distsfactory import make_dist, dist_exists


class TestExponentialMean:
    def test_basic(self):
        d = make_dist("exponential", mean=3)
        assert math.isclose(d.mean(), 3.0, rel_tol=1e-6)
        assert math.isclose(d.var(), 9.0, rel_tol=1e-6)

    def test_small_mean(self):
        d = make_dist("exponential", mean=0.1)
        assert math.isclose(d.mean(), 0.1, rel_tol=1e-6)

    def test_large_mean(self):
        d = make_dist("exponential", mean=1000)
        assert math.isclose(d.mean(), 1000.0, rel_tol=1e-6)


class TestExponentialMeanVar:
    def test_consistent(self):
        d = make_dist("exponential", mean=2.5, var=6.25)
        assert math.isclose(d.mean(), 2.5, rel_tol=1e-6)

    def test_inconsistent_raises(self):
        with pytest.raises(ValueError, match=r"var = mu\^2"):
            make_dist("exponential", mean=2.5, var=1.5)


class TestExponentialVar:
    def test_from_var(self):
        d = make_dist("exponential", var=9.0)
        assert math.isclose(d.mean(), 3.0, rel_tol=1e-6)
        assert math.isclose(d.var(), 9.0, rel_tol=1e-6)


class TestExponentialQuantile:
    def test_median(self):
        d = make_dist("exponential", median=2.0)
        assert math.isclose(d.ppf(0.5), 2.0, rel_tol=1e-6)

    def test_q1(self):
        d = make_dist("exponential", q1=1.0)
        assert math.isclose(d.ppf(0.25), 1.0, rel_tol=1e-6)

    def test_q3(self):
        d = make_dist("exponential", q3=5.0)
        assert math.isclose(d.ppf(0.75), 5.0, rel_tol=1e-6)


class TestExponentialExists:
    def test_feasible(self):
        assert dist_exists("exponential", mean=2.5, var=6.25) is True

    def test_infeasible_var(self):
        assert dist_exists("exponential", mean=2.5, var=1.5) is False

    def test_negative_mean(self):
        assert dist_exists("exponential", mean=-1, var=1) is False


class TestExponentialAliases:
    def test_exp_alias(self):
        d = make_dist("exp", mean=3)
        assert math.isclose(d.mean(), 3.0, rel_tol=1e-6)

    def test_expon_alias(self):
        d = make_dist("expon", mean=3)
        assert math.isclose(d.mean(), 3.0, rel_tol=1e-6)

"""Tests for Beta distribution construction."""

import math
import pytest
from distsfactory import make_dist, dist_exists


class TestBetaMeanVar:
    def test_basic(self):
        d = make_dist("beta", mean=0.4, var=0.02)
        assert math.isclose(d.mean(), 0.4, rel_tol=1e-6)
        assert math.isclose(d.var(), 0.02, rel_tol=1e-6)

    def test_symmetric(self):
        d = make_dist("beta", mean=0.5, var=0.05)
        assert math.isclose(d.mean(), 0.5, rel_tol=1e-6)
        assert math.isclose(d.var(), 0.05, rel_tol=1e-6)

    def test_skewed_right(self):
        d = make_dist("beta", mean=0.8, var=0.01)
        assert math.isclose(d.mean(), 0.8, rel_tol=1e-6)
        assert math.isclose(d.var(), 0.01, rel_tol=1e-6)

    def test_from_std(self):
        d = make_dist("beta", mean=0.5, std=0.1)
        assert math.isclose(d.mean(), 0.5, rel_tol=1e-6)
        assert math.isclose(d.var(), 0.01, rel_tol=1e-6)

    def test_var_too_large_raises(self):
        with pytest.raises(ValueError, match="variance too large"):
            make_dist("beta", mean=0.5, var=0.3)


class TestBetaMeanMode:
    def test_basic(self):
        d = make_dist("beta", mean=0.4, mode=0.35)
        assert math.isclose(d.mean(), 0.4, rel_tol=1e-4)

    def test_symmetric(self):
        # mean=0.5, mode=0.5 is underdetermined (any Beta(a,a) with a>1)
        with pytest.raises(ValueError):
            make_dist("beta", mean=0.5, mode=0.5)


class TestBetaQuantiles:
    def test_two_quantiles(self):
        d = make_dist("beta", q1=0.2, q3=0.6)
        assert math.isclose(d.ppf(0.25), 0.2, rel_tol=1e-3)
        assert math.isclose(d.ppf(0.75), 0.6, rel_tol=1e-3)

    def test_arbitrary_quantiles(self):
        d = make_dist("beta", quantiles=[(0.1, 0.15), (0.9, 0.85)])
        assert math.isclose(d.ppf(0.1), 0.15, rel_tol=1e-3)
        assert math.isclose(d.ppf(0.9), 0.85, rel_tol=1e-3)

    def test_mean_and_median(self):
        d = make_dist("beta", mean=0.4, median=0.38)
        assert math.isclose(d.mean(), 0.4, rel_tol=1e-3)
        assert math.isclose(d.ppf(0.5), 0.38, rel_tol=1e-3)


class TestBetaExists:
    def test_feasible(self):
        assert dist_exists("beta", mean=0.5, var=0.1) is True

    def test_var_too_large(self):
        assert dist_exists("beta", mean=0.5, var=0.3) is False

    def test_mean_out_of_range(self):
        assert dist_exists("beta", mean=1.5, var=0.1) is False
        assert dist_exists("beta", mean=-0.1, var=0.1) is False

    def test_boundary_var(self):
        # var exactly at the boundary mu*(1-mu) = 0.25
        assert dist_exists("beta", mean=0.5, var=0.25) is False
        # just under
        assert dist_exists("beta", mean=0.5, var=0.24) is True

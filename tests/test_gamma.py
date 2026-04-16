"""Tests for Gamma distribution construction."""

import math
import pytest
from distsfactory import make_dist, dist_exists


class TestGammaMeanVar:
    def test_basic(self):
        d = make_dist("gamma", mean=5, var=3)
        assert math.isclose(d.mean(), 5.0, rel_tol=1e-6)
        assert math.isclose(d.var(), 3.0, rel_tol=1e-6)

    def test_high_variance(self):
        d = make_dist("gamma", mean=10, var=50)
        assert math.isclose(d.mean(), 10.0, rel_tol=1e-6)
        assert math.isclose(d.var(), 50.0, rel_tol=1e-6)

    def test_low_variance(self):
        d = make_dist("gamma", mean=100, var=1)
        assert math.isclose(d.mean(), 100.0, rel_tol=1e-6)
        assert math.isclose(d.var(), 1.0, rel_tol=1e-6)

    def test_from_std(self):
        d = make_dist("gamma", mean=5, std=2)
        assert math.isclose(d.mean(), 5.0, rel_tol=1e-6)
        assert math.isclose(d.var(), 4.0, rel_tol=1e-6)

    def test_from_cv(self):
        d = make_dist("gamma", mean=5, cv=0.5)
        assert math.isclose(d.mean(), 5.0, rel_tol=1e-6)
        assert math.isclose(d.var(), 6.25, rel_tol=1e-6)

    def test_from_scv(self):
        d = make_dist("gamma", mean=4, scv=0.5)
        assert math.isclose(d.mean(), 4.0, rel_tol=1e-6)
        assert math.isclose(d.var(), 8.0, rel_tol=1e-6)


class TestGammaMeanMode:
    def test_basic(self):
        d = make_dist("gamma", mean=5, mode=3)
        assert math.isclose(d.mean(), 5.0, rel_tol=1e-6)

    def test_mode_close_to_zero(self):
        d = make_dist("gamma", mean=5, mode=0.5)
        assert math.isclose(d.mean(), 5.0, rel_tol=1e-6)

    def test_mode_exceeds_mean_raises(self):
        with pytest.raises(ValueError, match="must be greater than mode"):
            make_dist("gamma", mean=3, mode=5)


class TestGammaQuantiles:
    def test_two_quantiles(self):
        d = make_dist("gamma", q1=2.0, q3=8.0)
        assert math.isclose(d.ppf(0.25), 2.0, rel_tol=1e-4)
        assert math.isclose(d.ppf(0.75), 8.0, rel_tol=1e-4)

    def test_arbitrary_quantiles(self):
        d = make_dist("gamma", quantiles=[(0.1, 1.0), (0.9, 10.0)])
        assert math.isclose(d.ppf(0.1), 1.0, rel_tol=1e-4)
        assert math.isclose(d.ppf(0.9), 10.0, rel_tol=1e-4)

    def test_mean_and_median(self):
        d = make_dist("gamma", mean=5, median=4.5)
        assert math.isclose(d.mean(), 5.0, rel_tol=1e-4)
        assert math.isclose(d.ppf(0.5), 4.5, rel_tol=1e-4)


class TestGammaModeQuantile:
    def test_mode_and_median(self):
        d = make_dist("gamma", mode=3, median=4)
        assert math.isclose(d.ppf(0.5), 4.0, rel_tol=1e-4)

    def test_mode_and_iqr(self):
        d = make_dist("gamma", mode=3, iqr=4)
        q1, q3 = d.ppf(0.25), d.ppf(0.75)
        assert math.isclose(q3 - q1, 4.0, rel_tol=1e-4)


class TestGammaExists:
    def test_feasible(self):
        assert dist_exists("gamma", mean=5, var=3) is True

    def test_negative_mean(self):
        assert dist_exists("gamma", mean=-1, var=3) is False

    def test_zero_variance(self):
        assert dist_exists("gamma", mean=5, var=0) is False


class TestGammaScipy:
    def test_accepts_scipy_object(self):
        from scipy import stats
        d = make_dist(stats.gamma, mean=5, var=3)
        assert math.isclose(d.mean(), 5.0, rel_tol=1e-6)

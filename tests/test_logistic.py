"""Tests for Logistic distribution construction."""

import math
import pytest
from distsfactory import make_dist, dist_exists


class TestLogisticMeanVar:
    def test_basic(self):
        d = make_dist("logistic", mean=0, var=3)
        assert math.isclose(d.mean(), 0.0, abs_tol=1e-6)
        assert math.isclose(d.var(), 3.0, rel_tol=1e-6)

    def test_nonzero_mean(self):
        d = make_dist("logistic", mean=10, var=5)
        assert math.isclose(d.mean(), 10.0, rel_tol=1e-6)
        assert math.isclose(d.var(), 5.0, rel_tol=1e-6)

    def test_from_std(self):
        d = make_dist("logistic", mean=0, std=2)
        assert math.isclose(d.var(), 4.0, rel_tol=1e-6)

    def test_negative_mean(self):
        d = make_dist("logistic", mean=-5, var=10)
        assert math.isclose(d.mean(), -5.0, rel_tol=1e-6)


class TestLogisticQuantiles:
    def test_two_quantiles(self):
        d = make_dist("logistic", q1=2.0, q3=8.0)
        assert math.isclose(d.ppf(0.25), 2.0, rel_tol=1e-6)
        assert math.isclose(d.ppf(0.75), 8.0, rel_tol=1e-6)

    def test_arbitrary_quantiles(self):
        d = make_dist("logistic", quantiles=[(0.1, -5.0), (0.9, 15.0)])
        assert math.isclose(d.ppf(0.1), -5.0, rel_tol=1e-4)
        assert math.isclose(d.ppf(0.9), 15.0, rel_tol=1e-4)

    def test_mean_and_q3(self):
        d = make_dist("logistic", mean=5, q3=8)
        assert math.isclose(d.mean(), 5.0, rel_tol=1e-6)
        assert math.isclose(d.ppf(0.75), 8.0, rel_tol=1e-4)

    def test_mean_and_median_inconsistent_raises(self):
        with pytest.raises(ValueError, match="must equal median"):
            make_dist("logistic", mean=5, median=6)

    def test_mean_and_median_underdetermined_raises(self):
        with pytest.raises(ValueError, match="additional constraint"):
            make_dist("logistic", mean=5, median=5)


class TestLogisticModeIQR:
    def test_basic(self):
        d = make_dist("logistic", mode=5, iqr=4)
        q1, q3 = d.ppf(0.25), d.ppf(0.75)
        assert math.isclose(q3 - q1, 4.0, rel_tol=1e-6)
        # For logistic, mode = mean = median
        assert math.isclose(d.mean(), 5.0, rel_tol=1e-6)


class TestLogisticExists:
    def test_feasible(self):
        assert dist_exists("logistic", mean=0, var=3) is True

    def test_any_mean_works(self):
        assert dist_exists("logistic", mean=-100, var=0.01) is True

    def test_zero_var(self):
        assert dist_exists("logistic", mean=0, var=0) is False

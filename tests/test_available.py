"""Tests for available_distributions with support filters."""

import math
import pytest
from distsfactory import available_distributions


class TestSupportString:
    def test_positive(self):
        names = available_distributions(support="positive")
        assert "gamma" in names
        assert "exponential" in names
        assert "beta" not in names

    def test_unit(self):
        names = available_distributions(support="unit")
        assert names == ["beta"]

    def test_real(self):
        names = available_distributions(support="real")
        assert "normal" in names
        assert "gamma" not in names
        assert "beta" not in names

    def test_integer_nonneg(self):
        names = available_distributions(support="integer_nonneg")
        assert set(names) == {"poisson", "negative_binomial", "geometric"}

    def test_integer_bounded(self):
        names = available_distributions(support="integer_bounded")
        assert "binomial" in names
        assert "discrete_uniform" in names

    def test_unknown_string(self):
        with pytest.raises(ValueError, match="Unknown support string"):
            available_distributions(support="bogus")


class TestSupportTuple:
    def test_positive_via_tuple(self):
        names = available_distributions(support=(0, math.inf))
        assert "gamma" in names
        assert "normal" not in names

    def test_real_via_tuple(self):
        names = available_distributions(support=(-math.inf, math.inf))
        assert "normal" in names

    def test_unit_via_tuple(self):
        names = available_distributions(support=(0, 1))
        assert names == ["beta"]


class TestWithMomentsFilter:
    def test_filters_to_feasible_in_positive(self):
        names = available_distributions(support="positive", mean=5, var=3)
        assert "gamma" in names
        assert "exponential" not in names  # var != mean^2

    def test_filters_to_feasible_in_real(self):
        names = available_distributions(support="real", mean=0, var=2)
        assert "normal" in names
        # TDist requires mean=0 AND var>1, both satisfied here.
        assert "tdist" in names

    def test_no_filter_returns_all(self):
        names = available_distributions()
        # Has at least the well-known suspects
        for n in ("gamma", "beta", "normal", "binomial", "poisson"):
            assert n in names

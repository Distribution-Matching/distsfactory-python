"""Tests for the top-level API: make_dist, dist_exists, available_distributions."""

import pytest
from distsfactory import make_dist, dist_exists, available_distributions


class TestMakeDistErrors:
    def test_unknown_distribution(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            make_dist("pareto", mean=5)

    def test_no_spec(self):
        with pytest.raises(ValueError, match="at least one"):
            make_dist("gamma")

    def test_unsupported_spec(self):
        # Gamma doesn't support mean-only (2 params needed)
        with pytest.raises(ValueError, match="does not support"):
            make_dist("gamma", mean=5)


class TestAvailableDistributions:
    def test_mean_var(self):
        dists = available_distributions(mean=5, var=3)
        assert "gamma" in dists
        assert "logistic" in dists

    def test_exponential_only_when_var_matches(self):
        # mean=5, var=25 -> exponential is feasible (var = mean^2)
        dists = available_distributions(mean=5, var=25)
        assert "exponential" in dists

        # mean=5, var=3 -> exponential is not feasible
        dists = available_distributions(mean=5, var=3)
        assert "exponential" not in dists

    def test_beta_only_in_unit(self):
        # mean=0.5, var=0.05 -> beta feasible
        dists = available_distributions(mean=0.5, var=0.05)
        assert "beta" in dists

        # mean=5, var=3 -> beta not feasible (mean outside [0,1])
        dists = available_distributions(mean=5, var=3)
        assert "beta" not in dists


class TestDistExists:
    def test_returns_bool(self):
        result = dist_exists("gamma", mean=5, var=3)
        assert result is True
        assert isinstance(result, bool)

    def test_false_result(self):
        result = dist_exists("exponential", mean=5, var=3)
        assert result is False


class TestCaseInsensitive:
    def test_uppercase(self):
        d = make_dist("Gamma", mean=5, var=3)
        assert round(d.mean(), 4) == 5.0

    def test_mixed_case(self):
        d = make_dist("BETA", mean=0.5, var=0.05)
        assert round(d.mean(), 4) == 0.5

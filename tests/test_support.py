"""Tests for support-based construction (affine + truncation)."""

import math
import pytest
from distsfactory import make_dist


def _close(a, b, rel=1e-6, abs_tol=1e-10):
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


class TestAffineScale:
    def test_beta_on_arbitrary_interval(self):
        d = make_dist("beta", mean=3.5, var=0.5, support=(2, 7))
        assert _close(d.mean(), 3.5)
        assert _close(d.var(), 0.5)
        lo, hi = d.support()
        assert _close(lo, 2.0)
        assert _close(hi, 7.0)

    def test_uniform_continuous_on_interval(self):
        d = make_dist("uniform", mean=5, var=3, support=(0, 10))
        assert _close(d.mean(), 5.0)
        assert _close(d.var(), 3.0)


class TestAffineShift:
    def test_gamma_shifted(self):
        d = make_dist("gamma", mean=8, var=3, support=(3, math.inf))
        assert _close(d.mean(), 8.0)
        assert _close(d.var(), 3.0)
        lo, hi = d.support()
        assert _close(lo, 3.0)
        assert math.isinf(hi)

    def test_exponential_shifted(self):
        d = make_dist("exponential", mean=5, var=4, support=(3, math.inf))
        # mean=5 with lower bound 3 -> standard mean = 2, var = 4 -> Exponential(2)
        assert _close(d.mean(), 5.0)
        assert _close(d.var(), 4.0)


class TestAffineFlip:
    def test_gamma_flipped(self):
        d = make_dist("gamma", mean=5, var=3, support=(-math.inf, 10))
        assert _close(d.mean(), 5.0)
        assert _close(d.var(), 3.0)
        lo, hi = d.support()
        assert math.isinf(lo)
        assert _close(hi, 10.0)


class TestDiscreteShift:
    def test_binomial_shifted(self):
        d = make_dist("binomial", mean=12, var=1.2, support=range(10, 16))
        assert _close(d.mean(), 12.0)
        assert _close(d.var(), 1.2)

    def test_discrete_uniform_shifted(self):
        d = make_dist("discrete_uniform", mean=5, var=2, support=range(3, 8))
        # support {3,4,5,6,7}, n=4, mean=5, var = 4*6/12 = 2
        assert _close(d.mean(), 5.0)
        assert _close(d.var(), 2.0)


class TestTruncation:
    def test_normal_two_sided(self):
        d = make_dist("normal", mean=0.1, var=0.05, support=(-0.5, 0.5))
        assert _close(d.mean(), 0.1, rel=1e-6)
        assert _close(d.var(), 0.05, rel=1e-6)

    def test_normal_returns_truncnorm(self):
        # scipy has truncnorm natively, so we return that rather than _TruncatedDist.
        d = make_dist("normal", mean=0.1, var=0.05, support=(-0.5, 0.5))
        assert d.dist.name == "truncnorm"

    def test_normal_half_below(self):
        d = make_dist("normal", mean=2.0, var=3.0, support=(0.0, math.inf))
        assert _close(d.mean(), 2.0, rel=1e-6)
        assert _close(d.var(), 3.0, rel=1e-6)

    def test_normal_half_above(self):
        d = make_dist("normal", mean=-2.0, var=3.0, support=(-math.inf, 0.0))
        assert _close(d.mean(), -2.0, rel=1e-6)
        assert _close(d.var(), 3.0, rel=1e-6)

    def test_laplace_two_sided(self):
        d = make_dist("laplace", mean=0.0, var=0.1, support=(-1.0, 1.0))
        assert _close(d.mean(), 0.0, abs_tol=1e-8)
        assert _close(d.var(), 0.1, rel=1e-6)

    def test_laplace_half_below(self):
        d = make_dist("laplace", mean=2.0, var=3.0, support=(0.0, math.inf))
        assert _close(d.mean(), 2.0, rel=1e-6)
        assert _close(d.var(), 3.0, rel=1e-6)

    def test_logistic_two_sided(self):
        d = make_dist("logistic", mean=0.0, var=0.1, support=(-1.0, 1.0))
        assert _close(d.mean(), 0.0, abs_tol=1e-8)
        assert _close(d.var(), 0.1, rel=1e-6)


class TestTruncatedScipyNative:
    """Verify that truncated forms scipy supports natively are returned as
    real scipy frozen distributions, not our _TruncatedDist wrapper.
    """

    def test_truncated_exponential_returns_truncexpon(self):
        from distsfactory._truncation_solvers import _solve_truncated_exponential
        d = _solve_truncated_exponential(0.0, 10.0, 2.0, 3.50262)
        assert d.dist.name == "truncexpon"

    def test_truncated_exponential_no_upper_bound(self):
        # Half-truncated (lo > 0, hi = +inf) returns scipy.expon shifted
        d = make_dist("exponential", mean=5.0, var=9.0, support=(2.0, math.inf))
        assert _close(d.mean(), 5.0)
        assert _close(d.var(), 9.0)

    def test_truncated_exp_inconsistent_var_rejected(self):
        with pytest.raises(ValueError, match="variance is determined"):
            make_dist("exponential", mean=2.0, var=4.0, support=(0.0, 10.0))


class TestSupportErrors:
    def test_no_meanvar_with_support(self):
        with pytest.raises(ValueError, match="support="):
            make_dist("gamma", mean=5, support=(3, math.inf))

    def test_positive_dist_on_negative_lower(self):
        with pytest.raises(ValueError, match=r"lo < 0"):
            make_dist("gamma", mean=5, var=3, support=(-2, 5))

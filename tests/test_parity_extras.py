"""Parity-gap tests for TriangularDist, DiscreteTriangular, truncated TDist,
truncated Poisson, and generic-truncation Gamma/Beta moment matching.

Self-contained — no Julia dependency.
"""

import math
import pytest
from distsfactory import make_dist, partial_dist


def _close(a, b, rel=1e-5, abs_tol=1e-9):
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


# ---------------------------------------------------------------------------
# Continuous Triangular (3-parameter, mode-based)
# ---------------------------------------------------------------------------

class TestTriangular:
    def test_symmetric(self):
        d = make_dist("triangular", mean=0.0, var=1.0, mode=0.0)
        assert _close(d.mean(), 0.0)
        assert _close(d.var(), 1.0)

    def test_asymmetric(self):
        d = make_dist("triangular", mean=5.0, var=2.0, mode=4.0)
        assert _close(d.mean(), 5.0)
        assert _close(d.var(), 2.0)
        lo, hi = d.support()
        # mode should be inside support
        assert lo <= 4.0 <= hi

    def test_infeasible_raises(self):
        # mode far from mean with small variance -> negative discriminant
        with pytest.raises(ValueError, match=r"no real|a <= c <= b"):
            make_dist("triangular", mean=5.0, var=0.1, mode=6.0)


# ---------------------------------------------------------------------------
# Discrete extensions
# ---------------------------------------------------------------------------

class TestDiscreteSymTriangular:
    def test_construct(self):
        d = make_dist("discrete_sym_triangular", mean=5, var=4)
        # n=4 -> var = n*(n+2)/6 = 24/6 = 4
        assert _close(d.mean(), 5.0)
        assert _close(d.var(), 4.0)
        assert d.support() == (1, 9)

    def test_pmf_normalizes(self):
        d = make_dist("discrete_sym_triangular", mean=5, var=4)
        lo, hi = d.support()
        total = sum(float(d.pmf(k)) for k in range(lo, hi + 1))
        assert _close(total, 1.0)

    def test_mode(self):
        d = make_dist("discrete_sym_triangular", mean=5, var=4)
        assert d.mode() == 5

    def test_infeasible_non_integer_n(self):
        # n must be a non-neg integer; var=5 doesn't give one
        with pytest.raises(ValueError, match="half-width"):
            make_dist("discrete_sym_triangular", mean=5, var=5)


class TestDiscreteTriangular:
    def test_construct(self):
        d = make_dist("discrete_triangular", mean=5, var=2, mode=5)
        # mean and var won't be exact (3 integer params), but close
        assert abs(d.mean() - 5) < 1.0
        assert abs(d.var() - 2) < 1.0

    def test_mode_preserved(self):
        d = make_dist("discrete_triangular", mean=5, var=2, mode=4)
        assert d.mode() == 4


# ---------------------------------------------------------------------------
# Truncated TDist (PartialDist + support)
# ---------------------------------------------------------------------------

class TestTruncatedTDist:
    def test_half_below(self):
        spec = partial_dist("tdist", df=5)
        d = make_dist(spec, mean=2.0, var=1.0, support=(0.0, math.inf))
        assert _close(d.mean(), 2.0)
        assert _close(d.var(), 1.0)

    def test_half_above(self):
        spec = partial_dist("tdist", df=5)
        d = make_dist(spec, mean=-2.0, var=1.0, support=(-math.inf, 0.0))
        assert _close(d.mean(), -2.0)
        assert _close(d.var(), 1.0)

    def test_low_df_rejected(self):
        spec = partial_dist("tdist", df=2)
        with pytest.raises(ValueError, match="df > 2"):
            make_dist(spec, mean=2.0, var=1.0, support=(0.0, math.inf))

    def test_two_sided_not_implemented(self):
        spec = partial_dist("tdist", df=5)
        with pytest.raises(NotImplementedError, match="Two-sided"):
            make_dist(spec, mean=0.0, var=0.5, support=(-1.0, 1.0))


# ---------------------------------------------------------------------------
# Truncated Poisson
# ---------------------------------------------------------------------------

class TestTruncatedPoisson:
    def test_mean_consistent_var(self):
        # Truncated Poisson: variance is determined by the mean. The user must
        # supply a var consistent with that (within rtol=1e-3) or get an error.
        # First call directly to discover the consistent variance for mean=2.5
        # on {2,...,10}.
        from distsfactory._distributions import truncated_poisson
        td = truncated_poisson(2, 10, 2.5)
        achieved_var = td.var()
        d = make_dist("poisson", mean=2.5, var=achieved_var, support=range(2, 11))
        assert _close(d.mean(), 2.5)
        assert _close(d.var(), achieved_var)

    def test_inconsistent_var_rejected(self):
        with pytest.raises(ValueError, match="determined by the mean"):
            make_dist("poisson", mean=4.0, var=4.0, support=range(2, 11))

    def test_mean_outside_bounds_rejected(self):
        with pytest.raises(ValueError, match="must be in"):
            make_dist("poisson", mean=1.0, var=1.0, support=range(2, 11))


# ---------------------------------------------------------------------------
# Generic truncation (non-locscale) with actual moment matching
# ---------------------------------------------------------------------------

class TestTruncatedGammaBeta:
    def test_truncated_gamma_matches_moments(self):
        d = make_dist("gamma", mean=3.0, var=1.0, support=(0.0, 10.0))
        assert _close(d.mean(), 3.0, rel=1e-6)
        assert _close(d.var(), 1.0, rel=1e-6)

    def test_truncated_beta_matches_moments(self):
        d = make_dist("beta", mean=0.5, var=0.02, support=(0.2, 0.8))
        assert _close(d.mean(), 0.5, rel=1e-6)
        assert _close(d.var(), 0.02, rel=1e-6)

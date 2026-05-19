"""Tests for real-line continuous distributions."""

import math
import pytest
from distsfactory import make_dist, dist_exists


def _close(a, b, rel=1e-6):
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-12)


class TestNormal:
    def test_mean_var(self):
        d = make_dist("normal", mean=5, var=3)
        assert _close(d.mean(), 5.0)
        assert _close(d.var(), 3.0)

    def test_alias_norm(self):
        d = make_dist("norm", mean=0, var=1)
        assert _close(d.var(), 1.0)

    def test_q1_q3(self):
        d = make_dist("normal", q1=-1.0, q3=1.0)
        assert _close(d.ppf(0.25), -1.0)
        assert _close(d.ppf(0.75), 1.0)
        assert _close(d.mean(), 0.0, rel=1e-9)

    def test_median_iqr(self):
        d = make_dist("normal", median=5.0, iqr=2.0)
        assert _close(d.ppf(0.5), 5.0)
        assert _close(d.ppf(0.75) - d.ppf(0.25), 2.0)

    def test_mode_var(self):
        d = make_dist("normal", mode=3, var=4)
        assert _close(d.mean(), 3.0)
        assert _close(d.var(), 4.0)


class TestLaplace:
    def test_mean_var(self):
        d = make_dist("laplace", mean=2, var=8)
        assert _close(d.mean(), 2.0)
        assert _close(d.var(), 8.0)

    def test_quantiles(self):
        d = make_dist("laplace", q1=-0.5, q3=0.5)
        assert _close(d.ppf(0.25), -0.5)
        assert _close(d.ppf(0.75), 0.5)

    def test_mode_iqr(self):
        d = make_dist("laplace", mode=0.0, iqr=2 * math.log(2))
        assert _close(d.kwds["scale"], 1.0)


class TestLogisticExpanded:
    def test_mean_var(self):
        d = make_dist("logistic", mean=0, var=math.pi ** 2 / 3)
        assert _close(d.kwds["scale"], 1.0)


class TestGumbel:
    def test_mean_var(self):
        d = make_dist("gumbel", mean=5, var=3)
        assert _close(d.mean(), 5.0)
        assert _close(d.var(), 3.0)

    def test_two_quantiles(self):
        d = make_dist("gumbel", quantiles=[(0.1, 1.0), (0.9, 5.0)])
        assert _close(d.ppf(0.1), 1.0)
        assert _close(d.ppf(0.9), 5.0)


class TestCauchy:
    def test_mean_var_rejected(self):
        with pytest.raises(ValueError, match="no finite mean"):
            make_dist("cauchy", mean=0, var=1)

    def test_two_quantiles(self):
        d = make_dist("cauchy", q1=-1.0, q3=1.0)
        # standard Cauchy: ppf(0.25) = -1, ppf(0.75) = 1
        assert _close(d.ppf(0.25), -1.0)
        assert _close(d.ppf(0.75), 1.0)


class TestTDist:
    def test_mean_var(self):
        d = make_dist("tdist", mean=0, var=2)
        # nu = 2*var/(var-1) = 4
        assert _close(d.kwds["df"], 4.0)
        assert _close(d.var(), 2.0)

    def test_mean_nonzero_rejected(self):
        assert not dist_exists("tdist", mean=1, var=2)

    def test_var_le_one_rejected(self):
        assert not dist_exists("tdist", mean=0, var=0.5)


class TestUniformReal:
    def test_mean_var(self):
        d = make_dist("uniform", mean=5, var=3)
        assert _close(d.mean(), 5.0)
        assert _close(d.var(), 3.0)


class TestSymTriangular:
    def test_mean_var(self):
        d = make_dist("sym_triangular", mean=2, var=6)
        assert _close(d.mean(), 2.0)
        assert _close(d.var(), 6.0)

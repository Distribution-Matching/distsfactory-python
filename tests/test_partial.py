"""Tests for ``PartialDist`` — Python analog of Julia's ``@dist`` / ``DistSpec``.

Self-contained; mirrors Julia's `test_partial_dist_from_mean`, `test_partial_normal_fix_sigma`,
`test_partial_dist_from_mean_var`, `test_partial_beta`, etc.
"""

import math
import pytest
from distsfactory import partial_dist, make_dist, PartialDist


def _close(a, b, rel=1e-5, abs_tol=1e-9):
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


class TestGammaPartial:
    def test_pin_shape_solve_scale_from_mean(self):
        spec = partial_dist("gamma", a=3.0)
        d = make_dist(spec, mean=5.0)
        assert _close(d.kwds["a"], 3.0)
        assert _close(d.kwds["scale"], 5.0 / 3.0)
        assert _close(d.mean(), 5.0)

    def test_pin_shape_solve_scale_from_var(self):
        spec = partial_dist("gamma", a=3.0)
        d = make_dist(spec, var=3.0)
        # alpha=3, theta = sqrt(var/alpha) = 1.0
        assert _close(d.kwds["scale"], 1.0)
        assert _close(d.var(), 3.0)

    def test_pin_scale_solve_shape_from_mean(self):
        spec = partial_dist("gamma", scale=2.0)
        d = make_dist(spec, mean=5.0)
        assert _close(d.kwds["scale"], 2.0)
        assert _close(d.kwds["a"], 2.5)

    def test_no_pin_two_free(self):
        spec = partial_dist("gamma")
        d = make_dist(spec, mean=5.0, var=3.0)
        assert _close(d.mean(), 5.0)
        assert _close(d.var(), 3.0)


class TestNormalPartial:
    def test_pin_loc_solve_scale(self):
        spec = partial_dist("normal", loc=3.0)
        d = make_dist(spec, var=4.0)
        assert _close(d.kwds["scale"], 2.0)
        assert _close(d.mean(), 3.0)
        assert _close(d.var(), 4.0)

    def test_pin_scale_solve_loc(self):
        spec = partial_dist("normal", scale=1.0)
        d = make_dist(spec, mean=3.0)
        assert _close(d.kwds["loc"], 3.0)


class TestLogisticPartial:
    def test_pin_loc_solve_scale(self):
        spec = partial_dist("logistic", loc=2.0)
        d = make_dist(spec, var=22.3)
        assert _close(d.mean(), 2.0)
        assert _close(d.var(), 22.3)


class TestBetaPartial:
    def test_pin_alpha_solve_beta(self):
        spec = partial_dist("beta", a=2.0)
        d = make_dist(spec, mean=0.4)
        # mean = a/(a+b) = 0.4 -> b = 3
        assert _close(d.kwds["b"], 3.0)
        assert _close(d.mean(), 0.4)


class TestExponentialPartial:
    def test_no_pin_solve_scale_from_mean(self):
        spec = partial_dist("exponential")
        d = make_dist(spec, mean=3.0)
        assert _close(d.kwds["scale"], 3.0)


class TestPoissonPartial:
    def test_no_pin_solve_from_mean(self):
        spec = partial_dist("poisson")
        d = make_dist(spec, mean=5.0)
        assert _close(d.kwds["mu"], 5.0)


class TestErrors:
    def test_unknown_distribution(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            partial_dist("not_a_dist")

    def test_partial_with_support_rejected(self):
        spec = partial_dist("gamma", a=3.0)
        with pytest.raises(ValueError, match="support"):
            make_dist(spec, mean=5.0, support=(3, math.inf))

    def test_repr_is_useful(self):
        spec = partial_dist("gamma", a=3.0)
        assert "gamma" in repr(spec)
        assert "a=3.0" in repr(spec)

    def test_fixed_params(self):
        spec = partial_dist("gamma", a=3.0)
        assert spec.fixed_params() == {"a": 3.0}
        assert spec.free_params() == ["scale"]


class TestAllFixedValidation:
    """When all parameters are pinned, solve_partial must verify the moments
    rather than silently returning a mismatched distribution.
    """

    def test_mean_mismatch_raises(self):
        spec = partial_dist("gamma", a=2.0, scale=3.0)  # mean=6, var=18
        with pytest.raises(ValueError, match="target_mean"):
            make_dist(spec, mean=10.0)

    def test_var_mismatch_raises(self):
        spec = partial_dist("gamma", a=2.0, scale=3.0)  # var=18
        with pytest.raises(ValueError, match="target_var"):
            make_dist(spec, var=50.0)

    def test_both_match_succeeds(self):
        spec = partial_dist("gamma", a=2.0, scale=3.0)
        d = make_dist(spec, mean=6.0, var=18.0)
        assert math.isclose(d.mean(), 6.0)
        assert math.isclose(d.var(), 18.0)

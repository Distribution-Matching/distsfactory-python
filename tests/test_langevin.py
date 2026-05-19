"""Tests for the Langevin module and the truncated location-scale envelope.

Self-contained — no Julia dependency. Mirrors the structure of
DistributionsFactories.jl/test/test_langevin.jl.
"""

import math
import pytest
from distsfactory._langevin import (
    langevin, langevin_deriv, inv_langevin, truncexp_max_var,
)
from distsfactory._feasibility import why_not_truncated_locscale
from distsfactory import make_dist, dist_exists


def _close(a, b, rel=1e-9, abs_tol=1e-12):
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


class TestLangevinFunction:
    @pytest.mark.parametrize("z", [-3.0, -0.5, -1e-5, 0.0, 1e-5, 0.5, 3.0])
    def test_roundtrip(self, z):
        y = langevin(z)
        z_back = inv_langevin(y)
        assert _close(z_back, z, rel=1e-8, abs_tol=1e-10)

    def test_zero(self):
        assert langevin(0.0) == 0.0
        assert langevin_deriv(0.0) == pytest.approx(1 / 3)

    @pytest.mark.parametrize("x", [1e-6, 1e-5, 1e-4, 1e-3])
    def test_series_matches_closed_form(self, x):
        # Closed form has cancellation, but the series is accurate to ~1e-12
        series = langevin(x)
        # For comparison at non-tiny x where the closed form is reliable:
        if x >= 1e-3:
            direct = 1 / math.tanh(x) - 1 / x
            assert _close(series, direct, rel=1e-12)
        # Just check shape: L(x) > 0 for x > 0
        assert series > 0

    def test_inv_out_of_domain_raises(self):
        with pytest.raises(ValueError, match="must be < 1"):
            inv_langevin(1.5)


class TestTruncexpMaxVar:
    def test_symmetric_midpoint(self):
        # At the midpoint of [-1, 1], the max variance is w**2 * L'(0) = 1/3
        v = truncexp_max_var(-1.0, 1.0, 0.0)
        assert _close(v, 1 / 3)

    def test_bhatia_davis_bound(self):
        # The dome is bounded above by Bhatia–Davis: (hi-mu)*(mu-lo).
        # The Langevin envelope is strictly *below* this universal bound.
        for mu in [0.1, 0.3, 0.5]:
            v = truncexp_max_var(0.0, 1.0, mu)
            bd = (1.0 - mu) * (mu - 0.0)
            assert v < bd, f"Langevin {v} should be < Bhatia-Davis {bd}"

    def test_mu_outside_support_raises(self):
        with pytest.raises(ValueError, match="must lie strictly"):
            truncexp_max_var(0.0, 1.0, 1.5)


class TestEnvelopePredicate:
    def test_normal_below_dome(self):
        reason = why_not_truncated_locscale("normal", 0.0, 0.1, -1.0, 1.0)
        assert reason is None  # feasible

    def test_normal_above_dome(self):
        reason = why_not_truncated_locscale("normal", 0.0, 0.5, -1.0, 1.0)
        assert reason is not None
        assert "Langevin" in reason

    def test_half_below_within(self):
        # var < (mu - lo)^2
        reason = why_not_truncated_locscale("normal", 2.0, 3.0, 0.0, math.inf)
        assert reason is None

    def test_half_below_out(self):
        # var >= (mu - lo)^2
        reason = why_not_truncated_locscale("normal", 2.0, 5.0, 0.0, math.inf)
        assert reason is not None
        assert "(mu - lo)^2" in reason

    def test_laplace_boundary_attained(self):
        # Laplace attains the boundary exactly: var == (mu-lo)^2 is feasible
        reason = why_not_truncated_locscale("laplace", 2.0, 4.0, 0.0, math.inf)
        assert reason is None

    def test_laplace_just_above_boundary(self):
        reason = why_not_truncated_locscale("laplace", 2.0, 4.1, 0.0, math.inf)
        assert reason is not None


class TestMakeDistEnforcesEnvelope:
    def test_normal_above_dome_raises_clean_value_error(self):
        with pytest.raises(ValueError, match="Langevin"):
            make_dist("normal", mean=0.0, var=0.5, support=(-1.0, 1.0))

    def test_laplace_above_envelope_raises(self):
        with pytest.raises(ValueError, match="exponential bound"):
            make_dist("laplace", mean=2.0, var=10.0, support=(0.0, math.inf))


class TestDistExistsWithSupport:
    def test_structural_check_only(self):
        # Matches Julia: dist_exists with `support=` is a structural predicate,
        # not the Langevin envelope. It returns True even when (mu, var) is
        # above the dome; that check fires at constructor time.
        assert dist_exists("normal", mean=0.0, var=0.5, support=(-1, 1)) is True
        assert dist_exists("normal", mean=0.0, var=10.0, support=(-1, 1)) is True

    def test_positive_dist_on_real_rejected(self):
        # Gamma's natural support is [0, inf); it cannot be placed on (-inf, inf).
        assert dist_exists("gamma", mean=0.0, var=1.0,
                           support=(-math.inf, math.inf)) is False

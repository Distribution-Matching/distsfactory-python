"""Regression tests for the 0.2.0 wrapper-accessor surface.

The wrapper classes ``_FlippedDist`` and ``_TruncatedDist`` (used when scipy
lacks a native form for the requested transform) now expose ``.parent`` as
the public name for the underlying distribution. The older ``._inner`` is
retained as a property alias for backwards compatibility but new code should
use ``.parent``.
"""

import math

import pytest

from distsfactory import make_dist
from distsfactory._support import _FlippedDist, _TruncatedDist


def test_truncated_laplace_exposes_parent():
    # Truncated Laplace falls back to _TruncatedDist (scipy has no truncexpon-like
    # native truncated Laplace).
    d = make_dist("laplace", mean=2.0, var=4.0, support=(0.0, math.inf))
    assert isinstance(d, _TruncatedDist)
    assert d.parent is not None
    # Verify the parent is the un-truncated Laplace.
    assert d.parent.dist.name == "laplace"
    # Sanity: the wrapped distribution's mean is the requested target,
    # not the parent's location.
    assert abs(d.mean() - 2.0) < 1e-3


def test_truncated_dist_inner_alias_still_works():
    """._inner is preserved as a property alias for ._inner-using legacy code."""
    d = make_dist("laplace", mean=2.0, var=4.0, support=(0.0, math.inf))
    assert isinstance(d, _TruncatedDist)
    # The property returns the same object as .parent.
    assert d._inner is d.parent


def test_truncated_dist_support_interval():
    """.support_interval gives the (lo, hi) endpoints as a tuple."""
    d = make_dist("laplace", mean=2.0, var=4.0, support=(0.0, math.inf))
    assert isinstance(d, _TruncatedDist)
    lo, hi = d.support_interval
    assert lo == 0.0
    assert hi == math.inf
    # support() method matches.
    assert d.support_interval == (d.lo, d.hi)


def test_flipped_dist_exposes_parent():
    # Gamma flipped onto (-inf, b]. _affine_flip in _support.py returns a
    # _FlippedDist.
    d = make_dist("gamma", mean=5.0, var=3.0, support=(-math.inf, 10.0))
    assert isinstance(d, _FlippedDist)
    assert d.parent is not None
    # The parent is a frozen gamma.
    assert d.parent.dist.name == "gamma"
    # Moments transform: X = b - Y, so E[X] = b - E[Y].
    assert abs(d.mean() - (d.b - d.parent.mean())) < 1e-9
    assert abs(d.var() - d.parent.var()) < 1e-9


def test_flipped_dist_inner_alias_still_works():
    d = make_dist("gamma", mean=5.0, var=3.0, support=(-math.inf, 10.0))
    assert isinstance(d, _FlippedDist)
    assert d._inner is d.parent


def test_flipped_dist_repr_uses_parent_name():
    d = make_dist("gamma", mean=5.0, var=3.0, support=(-math.inf, 10.0))
    assert "parent=" in repr(d)


def test_truncated_dist_repr_uses_parent_name():
    d = make_dist("laplace", mean=2.0, var=4.0, support=(0.0, math.inf))
    assert "parent=" in repr(d)


def test_parent_is_a_frozen_scipy_distribution():
    """The .parent attribute must quack like a frozen scipy distribution so
    downstream code can use it directly."""
    d = make_dist("laplace", mean=2.0, var=4.0, support=(0.0, math.inf))
    p = d.parent
    # Frozen scipy distributions expose .mean(), .var(), .pdf(), .cdf(), .ppf(), .rvs(), .kwds
    assert callable(p.mean)
    assert callable(p.var)
    assert callable(p.pdf)
    assert callable(p.cdf)
    assert callable(p.ppf)
    assert callable(p.rvs)
    assert hasattr(p, "kwds")

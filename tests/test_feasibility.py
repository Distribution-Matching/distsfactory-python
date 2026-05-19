"""Feasibility-region tests: ``dist_exists`` and ``make_dist`` agree on the boundary.

Mirrors `test_mean_var_feasibility_rejections` in DistributionsFactories.jl/test/.

Self-contained — no Julia dependency.
"""

import math
import pytest
from scipy import stats
from distsfactory import make_dist, dist_exists


# Build a feasible (mean, var) pair for Chi by reading off a known Chi(df=4).
_CHI_REF = stats.chi(df=4)
CHI_MEAN, CHI_VAR = _CHI_REF.mean(), _CHI_REF.var()
RAYLEIGH_CV2 = (4 - math.pi) / math.pi


FEASIBLE_CASES = [
    ("normal",            dict(mean=0, var=1)),
    ("laplace",           dict(mean=0, var=2)),
    ("logistic",          dict(mean=0, var=3)),
    ("gumbel",            dict(mean=0, var=2)),
    ("uniform",           dict(mean=5, var=3)),
    ("sym_triangular",    dict(mean=0, var=1)),
    ("tdist",             dict(mean=0, var=2)),
    ("gamma",             dict(mean=5, var=3)),
    ("exponential",       dict(mean=3, var=9)),
    ("erlang",            dict(mean=4, var=4)),
    ("lognormal",         dict(mean=5, var=3)),
    ("weibull",           dict(mean=5, var=3)),
    ("frechet",           dict(mean=5, var=3)),
    ("rayleigh",          dict(mean=5, var=5 ** 2 * RAYLEIGH_CV2)),
    ("chi",               dict(mean=CHI_MEAN, var=CHI_VAR)),
    ("chisq",             dict(mean=4, var=8)),
    ("fdist",             dict(mean=1.5, var=10)),
    ("inverse_gamma",     dict(mean=5, var=3)),
    ("pareto",            dict(mean=5, var=3)),
    ("folded_normal",     dict(mean=2.5, var=1.2)),
    ("beta",              dict(mean=0.5, var=0.05)),
    ("binomial",          dict(mean=5, var=2.5)),
    ("poisson",           dict(mean=5, var=5)),
    ("negative_binomial", dict(mean=5, var=8)),
    ("geometric",         dict(mean=2, var=6)),
    ("discrete_uniform",  dict(mean=5, var=10)),
]


# (name, kwargs, why_label_to_grep) — infeasible cases that must reject.
INFEASIBLE_CASES = [
    # Universal: negative variance
    ("normal",            dict(mean=0, var=-1), "var > 0"),
    ("gamma",             dict(mean=5, var=-1), "var > 0"),
    ("beta",              dict(mean=0.5, var=-1), "var > 0"),
    # Gamma: non-positive mean
    ("gamma",             dict(mean=-1, var=3), "mu > 0"),
    # Exponential: var != mean^2
    ("exponential",       dict(mean=2.5, var=1.5), "var = mu"),
    ("exponential",       dict(mean=2.5, var=20.0), "var = mu"),
    # Beta: mean outside (0,1)
    ("beta",              dict(mean=1.5, var=0.05), "0 < mu < 1"),
    ("beta",              dict(mean=-0.1, var=0.05), "0 < mu < 1"),
    # Beta: variance too large
    ("beta",              dict(mean=0.5, var=0.3), r"var < mu\*\(1-mu\)"),
    # TDist: mean must be 0
    ("tdist",             dict(mean=1, var=2), "mu = 0"),
    # TDist: var must be > 1
    ("tdist",             dict(mean=0, var=0.5), "var > 1"),
    # FDist: mean must be in (1, 2)
    ("fdist",             dict(mean=0.5, var=1), "mu > 1"),
    ("fdist",             dict(mean=2.5, var=1), "mu < 2"),
    # Poisson: var != mean
    ("poisson",           dict(mean=5, var=3), "mu = var"),
    # NegativeBinomial: var must be > mean
    ("negative_binomial", dict(mean=5, var=3), "var > mu"),
    # Binomial: var must be < mean
    ("binomial",          dict(mean=5, var=7), "var < mu"),
    # Geometric: var must equal mean*(1+mean)
    ("geometric",         dict(mean=2, var=3), r"var = mu\*\(1\+mu\)"),
    # Rayleigh: CV must equal sqrt((4-pi)/pi)
    ("rayleigh",          dict(mean=5, var=10), "CV"),
    # Cauchy: no finite moments
    ("cauchy",            dict(mean=0, var=1), "no finite mean"),
]


@pytest.mark.parametrize("name,kw", FEASIBLE_CASES, ids=[c[0] for c in FEASIBLE_CASES])
def test_feasible_constructs(name, kw):
    d = make_dist(name, **kw)
    expected_mean = kw["mean"]
    expected_var = kw["var"]
    assert math.isclose(d.mean(), expected_mean, rel_tol=1e-5, abs_tol=1e-9)
    assert math.isclose(d.var(), expected_var, rel_tol=1e-5, abs_tol=1e-9)


@pytest.mark.parametrize("name,kw", FEASIBLE_CASES, ids=[c[0] for c in FEASIBLE_CASES])
def test_feasible_exists_true(name, kw):
    assert dist_exists(name, **kw) is True


@pytest.mark.parametrize(
    "name,kw,match", INFEASIBLE_CASES, ids=[f"{c[0]}_{c[2][:12]}" for c in INFEASIBLE_CASES]
)
def test_infeasible_raises(name, kw, match):
    with pytest.raises(ValueError, match=match):
        make_dist(name, **kw)


@pytest.mark.parametrize(
    "name,kw,match", INFEASIBLE_CASES, ids=[f"{c[0]}_{c[2][:12]}" for c in INFEASIBLE_CASES]
)
def test_infeasible_exists_false(name, kw, match):
    assert dist_exists(name, **kw) is False

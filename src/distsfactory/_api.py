"""Public API: ``make_dist``, ``dist_exists``, ``available_distributions``."""

import math

from ._registry import resolve_dist, DISTRIBUTIONS, SUPPORT_TYPE
from ._distributions import DIST_HANDLERS
from ._feasibility import exists_mean_var as _exists_mean_var
from ._specs import parse_spec, MeanVarSpec, MeanSpec, VarSpec
from ._support import dist_on_support as _dist_on_support
from ._partial import PartialDist, solve_partial


_SUPPORT_STRINGS = {
    "real", "positive", "unit", "integer_nonneg", "integer_bounded"
}


def _classify_support(support):
    """Classify a tuple/range/string into a natural-support category."""
    if isinstance(support, str):
        if support not in _SUPPORT_STRINGS:
            raise ValueError(
                f"Unknown support string {support!r}. "
                f"Use one of {sorted(_SUPPORT_STRINGS)}."
            )
        return support
    if isinstance(support, range):
        if support.start == 0:
            return "integer_nonneg" if support.stop > 10 ** 9 else "integer_bounded"
        return "integer_bounded"
    if isinstance(support, tuple) and len(support) == 2:
        lo, hi = float(support[0]), float(support[1])
        if math.isinf(lo) and lo < 0 and math.isinf(hi) and hi > 0:
            return "real"
        if lo == 0 and math.isinf(hi):
            return "positive"
        if lo == 0 and hi == 1:
            return "unit"
        return None  # bounded non-unit: no candidates by natural support
    raise ValueError(
        f"Unsupported support type {type(support).__name__}. "
        f"Use a 2-tuple, range, or category string."
    )


def make_dist(dist, support=None, **kwargs):
    """Construct a distribution from partial specifications.

    Returns a frozen ``scipy.stats`` distribution whose moments, quantiles,
    or mode match the given constraints.

    Parameters
    ----------
    dist : str or scipy.stats distribution
        Distribution family. Canonical names (``"gamma"``, ``"beta"``,
        ``"normal"``, ``"weibull"``, ``"binomial"``, …) are accepted along with
        common aliases (case-insensitive). A ``scipy.stats`` distribution
        object is also accepted.
    support : tuple or range, optional
        Target support. Use ``(lo, hi)`` for a continuous interval (either
        endpoint may be ``math.inf`` / ``-math.inf``). Use ``range(a, b+1)``
        for a discrete interval ``{a, …, b}``. When given, the distribution
        is placed on this support via an affine transform or truncation
        (currently restricted to ``mean+var`` specs).
    **kwargs
        Specification keywords. At least one is required.

        Moment keywords: ``mean``, ``var``, ``std``, ``cv``, ``scv``,
        ``second_moment``.

        Quantile keywords: ``median``, ``q1``, ``q3``, ``iqr``,
        ``quantiles`` (list of two ``(p, q)`` tuples).

        Mode keywords: ``mode`` (combine with ``mean``, ``var``, ``iqr``,
        ``median``, ``q1``, ``q3``).

    Returns
    -------
    scipy.stats.rv_frozen
        A frozen distribution with ``pdf``/``pmf``, ``cdf``, ``ppf``, ``rvs``,
        ``mean``, ``var``, etc.

    Examples
    --------
    >>> from distsfactory import make_dist
    >>> round(make_dist("gamma", mean=5, var=3).mean(), 6)
    5.0
    >>> round(make_dist("normal", mean=0, var=4).var(), 6)
    4.0
    >>> round(make_dist("beta", mean=3.5, var=0.5, support=(2, 7)).mean(), 6)
    3.5
    """
    spec = parse_spec(**kwargs)

    # PartialDist branch: solve free parameters from moment constraints.
    if isinstance(dist, PartialDist):
        if support is not None:
            # Special case: PartialDist("tdist", df=ν) + support=(lo, hi)
            # routes to the truncated location-scale Student-t solver.
            if dist.name == "tdist" and "df" in dist.fixed and \
                    isinstance(spec, MeanVarSpec) and isinstance(support, tuple):
                from ._truncation_solvers import solve_truncated_tdist_half
                lo, hi = float(support[0]), float(support[1])
                return solve_truncated_tdist_half(
                    df=dist.fixed["df"], lo=lo, hi=hi,
                    mu=spec.mean, var=spec.var,
                )
            raise ValueError("`support=` is not supported with this PartialDist.")
        if isinstance(spec, MeanVarSpec):
            return solve_partial(dist, target_mean=spec.mean, target_var=spec.var)
        if isinstance(spec, MeanSpec):
            return solve_partial(dist, target_mean=spec.mean)
        if isinstance(spec, VarSpec):
            return solve_partial(dist, target_var=spec.var)
        raise ValueError(
            "PartialDist currently supports only mean, var, or mean+var specifications."
        )

    name, _ = resolve_dist(dist)

    if support is not None:
        if not isinstance(spec, MeanVarSpec):
            raise ValueError(
                "`support=` is currently only supported together with mean+var "
                "(or equivalent dispersion: std, cv, scv, second_moment)."
            )
        return _dist_on_support(dist, spec.mean, spec.var, support)

    handler = DIST_HANDLERS[name]
    spec_type = type(spec)

    if spec_type not in handler.DISPATCH:
        supported = sorted(t.__name__ for t in handler.DISPATCH)
        raise ValueError(
            f"Distribution {name!r} does not support specification type "
            f"{spec_type.__name__}. Supported: {supported}"
        )

    return handler.DISPATCH[spec_type](spec)


def dist_exists(dist, **kwargs):
    """Check whether a distribution can be constructed with the given constraints.

    For mean+variance specs this is a pure predicate (never throws). For other
    specs it falls back to trying to construct.

    Examples
    --------
    >>> from distsfactory import dist_exists
    >>> dist_exists("beta", mean=0.5, var=0.1)
    True
    >>> dist_exists("beta", mean=0.5, var=0.3)
    False
    >>> dist_exists("exponential", mean=2.5, var=6.25)
    True
    >>> dist_exists("exponential", mean=2.5, var=1.5)
    False
    """
    name, _ = resolve_dist(dist)
    spec = parse_spec(**kwargs)

    if isinstance(spec, MeanVarSpec):
        return _exists_mean_var(name, spec.mean, spec.var)

    # For other specs, try construction
    try:
        make_dist(dist, **kwargs)
        return True
    except (ValueError, RuntimeError, AssertionError):
        return False


def available_distributions(support=None, **kwargs):
    """List distributions feasible for the given constraints.

    Mirrors Julia's ``available_distributions(support; kwargs...)``.

    Parameters
    ----------
    support : tuple, range, str, or None
        Optional support filter. Accepts:

        - A 2-tuple ``(lo, hi)`` of endpoints (use ``math.inf`` for unbounded).
          The shape is classified to a natural-support category:
          ``(-inf, inf)`` -> ``"real"``,
          ``(0, inf)``   -> ``"positive"``,
          ``(0, 1)``     -> ``"unit"``,
          bounded otherwise.
        - A ``range`` object: classified as discrete bounded (or non-negative
          unbounded when ``stop`` is ``sys.maxsize``-like).
        - A string: one of ``"real"``, ``"positive"``, ``"unit"``,
          ``"integer_nonneg"``, ``"integer_bounded"``.
        - ``None``: no filter.
    **kwargs
        Specification keywords (same as ``make_dist``). When omitted, returns
        all candidates for the given support (or all distributions if
        ``support`` is also ``None``).

    Returns
    -------
    list of str
        Canonical distribution names that are feasible.

    Examples
    --------
    >>> import math
    >>> from distsfactory import available_distributions
    >>> "gamma" in available_distributions(mean=5, var=3)
    True
    >>> "beta" in available_distributions(support="unit")
    True
    >>> "gamma" in available_distributions(support=(0, math.inf), mean=5, var=3)
    True
    """
    candidates = list(DIST_HANDLERS.keys())
    if support is not None:
        natural = _classify_support(support)
        if natural is not None:
            candidates = [n for n in candidates if SUPPORT_TYPE.get(n) == natural]

    if not kwargs:
        return candidates

    spec = parse_spec(**kwargs)

    if isinstance(spec, MeanVarSpec):
        return [n for n in candidates if _exists_mean_var(n, spec.mean, spec.var)]

    feasible = []
    for name in candidates:
        try:
            make_dist(name, **kwargs)
            feasible.append(name)
        except (ValueError, RuntimeError, AssertionError):
            pass
    return feasible

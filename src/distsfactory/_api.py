"""Public API: ``make_dist``, ``dist_exists``, ``available_distributions``."""

from ._registry import resolve_dist, DISTRIBUTIONS, SUPPORT_TYPE
from ._distributions import DIST_HANDLERS
from ._feasibility import exists_mean_var as _exists_mean_var
from ._specs import parse_spec, MeanVarSpec


def make_dist(dist, **kwargs):
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
    >>> round(make_dist("weibull", mean=2, var=1).mean(), 4)
    2.0
    """
    name, _ = resolve_dist(dist)
    handler = DIST_HANDLERS[name]
    spec = parse_spec(**kwargs)
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

    Parameters
    ----------
    support : str or None
        Optional natural-support filter. One of ``"real"``, ``"positive"``,
        ``"unit"``, ``"integer_nonneg"``, ``"integer_bounded"``. When set,
        restricts the candidate pool to distributions with that natural
        support.
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
    >>> from distsfactory import available_distributions
    >>> "gamma" in available_distributions(mean=5, var=3)
    True
    >>> "beta" in available_distributions(support="unit")
    True
    """
    candidates = list(DIST_HANDLERS.keys())
    if support is not None:
        candidates = [n for n in candidates if SUPPORT_TYPE.get(n) == support]

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

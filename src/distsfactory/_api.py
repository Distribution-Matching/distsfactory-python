"""Public API: make_dist, dist_exists, available_distributions."""

from ._registry import resolve_dist, DISTRIBUTIONS, SUPPORT_TYPE
from ._distributions import DIST_HANDLERS
from ._specs import parse_spec, MeanVarSpec


def make_dist(dist, **kwargs):
    """Construct a distribution from partial specifications.

    Returns a frozen ``scipy.stats`` distribution whose moments or quantiles
    match the given constraints.

    Parameters
    ----------
    dist : str or scipy.stats distribution
        Distribution family.  Strings like ``"gamma"``, ``"beta"``,
        ``"exponential"``, ``"logistic"`` are recognised (case-insensitive).
        A ``scipy.stats`` distribution object is also accepted.
    **kwargs
        Specification keywords.  At least one is required.

        Moment keywords:
            mean, var, std, cv, scv, second_moment

        Quantile keywords:
            median, q1, q3, iqr, quantiles (list of two (p, q) tuples)

        Mode keywords:
            mode

        These can be combined (e.g. ``mean=5, var=3`` or ``q1=10, q3=30``).

    Returns
    -------
    scipy.stats.rv_frozen
        A frozen distribution object with ``pdf``, ``cdf``, ``ppf``,
        ``rvs``, ``mean``, ``var``, etc.

    Examples
    --------
    >>> from distsfactory import make_dist
    >>> d = make_dist("gamma", mean=5, var=3)
    >>> round(d.mean(), 6)
    5.0
    >>> d = make_dist("exponential", mean=3)
    >>> round(d.mean(), 6)
    3.0
    >>> d = make_dist("beta", mean=0.4, var=0.02)
    >>> round(d.mean(), 6)
    0.4
    """
    name, _ = resolve_dist(dist)
    handler = DIST_HANDLERS[name]
    spec = parse_spec(**kwargs)
    spec_type = type(spec)

    if spec_type not in handler.DISPATCH:
        supported = [t.__name__ for t in handler.DISPATCH]
        raise ValueError(
            f"Distribution {name!r} does not support specification type "
            f"{spec_type.__name__}. Supported: {supported}"
        )

    return handler.DISPATCH[spec_type](spec)


def dist_exists(dist, **kwargs):
    """Check whether a distribution can be constructed with the given constraints.

    Currently supports mean+variance feasibility checks.

    Parameters
    ----------
    dist : str or scipy.stats distribution
        Distribution family.
    **kwargs
        Specification keywords (same as ``make_dist``).

    Returns
    -------
    bool
        ``True`` if the distribution is feasible, ``False`` otherwise.

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
    handler = DIST_HANDLERS[name]
    spec = parse_spec(**kwargs)

    if isinstance(spec, MeanVarSpec):
        return handler.exists_mean_var(spec.mean, spec.var)

    # For specs without a feasibility check, try to construct
    try:
        make_dist(dist, **kwargs)
        return True
    except (ValueError, RuntimeError):
        return False


def available_distributions(**kwargs):
    """List distributions that can be constructed with the given constraints.

    When ``mean`` and ``var`` (or equivalent) are given, returns only those
    distributions for which the moment combination is feasible.

    Parameters
    ----------
    **kwargs
        Specification keywords (same as ``make_dist``).  At least ``mean``
        and one dispersion measure are required.

    Returns
    -------
    list of str
        Canonical distribution names that are feasible.

    Examples
    --------
    >>> from distsfactory import available_distributions
    >>> "gamma" in available_distributions(mean=5, var=3)
    True
    """
    spec = parse_spec(**kwargs)

    if isinstance(spec, MeanVarSpec):
        return [
            name for name, handler in DIST_HANDLERS.items()
            if handler.exists_mean_var(spec.mean, spec.var)
        ]

    # For other specs, try construction
    feasible = []
    for name in DIST_HANDLERS:
        try:
            make_dist(name, **kwargs)
            feasible.append(name)
        except (ValueError, RuntimeError):
            pass
    return feasible

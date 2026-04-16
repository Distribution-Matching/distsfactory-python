"""Registry mapping distribution names to scipy.stats distribution objects."""

from scipy import stats

# Canonical name -> scipy.stats continuous distribution
DISTRIBUTIONS = {
    "gamma": stats.gamma,
    "exponential": stats.expon,
    "logistic": stats.logistic,
    "beta": stats.beta,
}

# Aliases (common alternative names)
ALIASES = {
    "exp": "exponential",
    "expon": "exponential",
}

# Support classification for each distribution
SUPPORT_TYPE = {
    "gamma": "positive",
    "exponential": "positive",
    "logistic": "real",
    "beta": "unit",
}


def resolve_dist(dist):
    """Resolve a distribution argument to (canonical_name, scipy_dist).

    Parameters
    ----------
    dist : str or scipy.stats distribution
        Distribution name (e.g. "gamma") or a scipy.stats distribution object.

    Returns
    -------
    name : str
        Canonical distribution name.
    scipy_dist : scipy.stats.rv_continuous
        The scipy distribution object.
    """
    if isinstance(dist, str):
        name = dist.lower()
        name = ALIASES.get(name, name)
        if name not in DISTRIBUTIONS:
            raise ValueError(
                f"Unknown distribution: {dist!r}. "
                f"Available: {sorted(DISTRIBUTIONS.keys())}"
            )
        return name, DISTRIBUTIONS[name]

    # Assume it's a scipy.stats distribution object
    for name, scipy_dist in DISTRIBUTIONS.items():
        if dist is scipy_dist or getattr(dist, 'name', None) == scipy_dist.name:
            return name, scipy_dist

    raise ValueError(
        f"Unrecognized scipy distribution: {dist}. "
        f"Available: {sorted(DISTRIBUTIONS.keys())}"
    )

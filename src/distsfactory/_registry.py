"""Registry mapping canonical distribution names to scipy.stats objects.

Canonical names are lowercase strings matching the lowercase Julia type name
where reasonable (e.g. ``"gamma"``, ``"inverse_gamma"``, ``"negative_binomial"``).
The registry tracks:

- ``DISTRIBUTIONS`` — name -> scipy.stats distribution object (for ``make_dist``
  with a string argument and for round-tripping when a user passes a scipy
  object directly).
- ``SUPPORT_TYPE`` — name -> natural-support classification (``"real"``,
  ``"positive"``, ``"unit"``, ``"integer_nonneg"``, ``"integer_bounded"``).
  Mirrors `_natural_support` in `src/support.jl`.
- ``ALIASES`` — alternate spellings users may type.
"""

from scipy import stats


# ---------------------------------------------------------------------------
# Canonical name -> scipy.stats distribution
# ---------------------------------------------------------------------------
# Notes on parameterization:
#   - "geometric" uses scipy.nbinom(1, p) so the support is {0,1,...}, matching
#     Julia's Geometric. scipy.stats.geom is shifted to {1,2,...} and would
#     give the wrong mean.
#   - "frechet" maps to scipy.invweibull (Frechet is "inverse Weibull").
#   - "sym_triangular" maps to scipy.triang with c=0.5 (symmetric case).
#   - "erlang" reuses scipy.gamma — Erlang is Gamma with integer shape.

DISTRIBUTIONS = {
    # Real-line continuous
    "normal":            stats.norm,
    "tdist":             stats.t,
    "logistic":          stats.logistic,
    "laplace":           stats.laplace,
    "gumbel":            stats.gumbel_r,
    "cauchy":            stats.cauchy,
    "sym_triangular":    stats.triang,
    "triangular":        stats.triang,
    "uniform":           stats.uniform,
    # Positive continuous
    "gamma":             stats.gamma,
    "erlang":            stats.gamma,
    "exponential":       stats.expon,
    "lognormal":         stats.lognorm,
    "weibull":           stats.weibull_min,
    "frechet":           stats.invweibull,
    "chi":               stats.chi,
    "chisq":             stats.chi2,
    "rayleigh":          stats.rayleigh,
    "fdist":             stats.f,
    "inverse_gamma":     stats.invgamma,
    "pareto":            stats.pareto,
    "folded_normal":     stats.foldnorm,
    # Unit-interval continuous
    "beta":              stats.beta,
    # Discrete
    "binomial":          stats.binom,
    "poisson":           stats.poisson,
    "negative_binomial": stats.nbinom,
    "geometric":         stats.nbinom,  # parameterized as nbinom(1, p)
    "discrete_uniform":  stats.randint,
    # Extensions (not in scipy.stats — see _extensions.py)
    "discrete_sym_triangular": None,
    "discrete_triangular":     None,
}


# Common alternate names users might type
ALIASES = {
    "exp":            "exponential",
    "expon":          "exponential",
    "norm":           "normal",
    "gaussian":       "normal",
    "log_normal":     "lognormal",
    "log-normal":     "lognormal",
    "chi2":           "chisq",
    "chi_squared":    "chisq",
    "chisquare":      "chisq",
    "f":              "fdist",
    "f_dist":         "fdist",
    "invgamma":       "inverse_gamma",
    "inv_gamma":      "inverse_gamma",
    "nbinom":         "negative_binomial",
    "neg_binomial":   "negative_binomial",
    "discrete_unif":  "discrete_uniform",
    "discrete-unif":  "discrete_uniform",
    "randint":        "discrete_uniform",
    "t":              "tdist",
    "student_t":      "tdist",
    "students_t":     "tdist",
    "weibull_min":    "weibull",
    "invweibull":     "frechet",
    "gumbel_r":       "gumbel",
    "triang":         "sym_triangular",
    "symtriangular":  "sym_triangular",
    "foldnorm":       "folded_normal",
    "folded-normal":  "folded_normal",
}


# Natural support classification (mirrors `_natural_support` in src/support.jl)
SUPPORT_TYPE = {
    # Real-line
    "normal":            "real",
    "tdist":             "real",
    "logistic":          "real",
    "laplace":           "real",
    "gumbel":            "real",
    "cauchy":            "real",
    "sym_triangular":    "real",
    "triangular":        "real",
    "uniform":           "real",
    # Positive
    "gamma":             "positive",
    "erlang":            "positive",
    "exponential":       "positive",
    "lognormal":         "positive",
    "weibull":           "positive",
    "frechet":           "positive",
    "chi":               "positive",
    "chisq":             "positive",
    "rayleigh":          "positive",
    "fdist":             "positive",
    "inverse_gamma":     "positive",
    "pareto":            "positive",
    "folded_normal":     "positive",
    # Unit
    "beta":              "unit",
    # Discrete bounded
    "binomial":          "integer_bounded",
    "discrete_uniform":  "integer_bounded",
    "discrete_sym_triangular": "integer_bounded",
    "discrete_triangular":     "integer_bounded",
    # Discrete unbounded
    "poisson":           "integer_nonneg",
    "negative_binomial": "integer_nonneg",
    "geometric":         "integer_nonneg",
}


# Canonical "tunable" scipy params per distribution. Matches the parameter set
# the Julia Distributions.jl version exposes for that family. ``loc`` is omitted
# for families whose natural support has a finite start (positive, unit) because
# scipy adds it as a free shift parameter that doesn't exist in the canonical
# parameterization.
CANONICAL_PARAMS = {
    # Real-line
    "normal":            ("loc", "scale"),
    "tdist":             ("df", "loc", "scale"),
    "logistic":          ("loc", "scale"),
    "laplace":           ("loc", "scale"),
    "gumbel":            ("loc", "scale"),
    "cauchy":            ("loc", "scale"),
    "sym_triangular":    ("loc", "scale"),
    "uniform":           ("loc", "scale"),
    # Positive (no loc by default — scipy adds it but it's not in Julia's
    # canonical form)
    "gamma":             ("a", "scale"),
    "erlang":            ("a", "scale"),
    "exponential":       ("scale",),
    "lognormal":         ("s", "scale"),
    "weibull":           ("c", "scale"),
    "frechet":           ("c", "scale"),
    "chi":               ("df",),
    "chisq":             ("df",),
    "rayleigh":          ("scale",),
    "fdist":             ("dfn", "dfd"),
    "inverse_gamma":     ("a", "scale"),
    "pareto":            ("b", "scale"),
    "folded_normal":     ("c", "scale"),
    # Unit-interval
    "beta":              ("a", "b"),
    # Discrete (loc handled separately for shifted ranges)
    "binomial":          ("n", "p"),
    "poisson":           ("mu",),
    "negative_binomial": ("n", "p"),
    "geometric":         ("p",),         # we parameterize as nbinom(1, p)
    "discrete_uniform":  ("low", "high"),
}


def resolve_dist(dist):
    """Resolve a distribution argument to ``(canonical_name, scipy_dist)``.

    Accepts a canonical name (``"gamma"``), an alias (``"exp"``), or a
    ``scipy.stats`` distribution object. Case-insensitive for strings. For
    extension types not backed by scipy (e.g. ``discrete_triangular``), the
    returned scipy object is ``None`` — callers that need the scipy object
    should special-case extension names.
    """
    if isinstance(dist, str):
        name = dist.lower().replace("-", "_")
        name = ALIASES.get(name, name)
        if name not in DISTRIBUTIONS:
            raise ValueError(
                f"Unknown distribution: {dist!r}. "
                f"Available: {sorted(DISTRIBUTIONS.keys())}"
            )
        return name, DISTRIBUTIONS[name]

    # scipy.stats distribution object: match by `name` attribute (lower).
    target = getattr(dist, "name", None)
    if target is not None:
        for canon, scipy_dist in DISTRIBUTIONS.items():
            if scipy_dist is None:
                continue
            if dist is scipy_dist or scipy_dist.name == target:
                return canon, scipy_dist

    raise ValueError(
        f"Unrecognized scipy distribution: {dist}. "
        f"Available: {sorted(DISTRIBUTIONS.keys())}"
    )

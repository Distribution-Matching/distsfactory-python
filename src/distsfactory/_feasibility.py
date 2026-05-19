"""Per-distribution feasibility predicates for moment-based construction.

Mirrors the architecture of `dist_exists.jl` in DistributionsFactories.jl:
- `why_not_mean_var(name, mu, var)` returns ``None`` when feasible, otherwise a
  short human-readable reason string.
- `exists_mean_var(name, mu, var)` is the pure boolean predicate.
- `require_mean_var(name, mu, var)` raises ``ValueError`` carrying the reason
  string when infeasible; used by constructors before they compute parameters.

Per-distribution rules match the Julia conditions in `_why_not_dist_from_mean_var`.
"""

import math
from scipy.special import gamma as _gamma_fn


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _why_not_positive_var(name, var):
    if not (var > 0):
        return f"{name}: the condition var > 0 is not satisfied"
    return None


# ---------------------------------------------------------------------------
# Per-distribution rules
# ---------------------------------------------------------------------------

def _why_not_normal(mu, var):
    return _why_not_positive_var("Normal", var)


def _why_not_uniform(mu, var):
    return _why_not_positive_var("Uniform", var)


def _why_not_logistic(mu, var):
    return _why_not_positive_var("Logistic", var)


def _why_not_laplace(mu, var):
    return _why_not_positive_var("Laplace", var)


def _why_not_gumbel(mu, var):
    return _why_not_positive_var("Gumbel", var)


def _why_not_sym_triangular(mu, var):
    return _why_not_positive_var("SymTriangular", var)


def _why_not_tdist(mu, var):
    r = _why_not_positive_var("TDist", var)
    if r is not None:
        return r
    if not math.isclose(mu, 0.0, abs_tol=1e-12):
        return "TDist: the condition mu = 0 is not satisfied"
    if not (var > 1):
        return "TDist: the condition var > 1 is not satisfied"
    return None


def _why_not_cauchy(mu, var):
    return ("Cauchy: distribution has no finite mean or variance; "
            "construct via quantiles instead")


def _why_not_exponential(mu, var):
    r = _why_not_positive_var("Exponential", var)
    if r is not None:
        return r
    if not (mu > 0):
        return "Exponential: the condition mu > 0 is not satisfied"
    if not math.isclose(var, mu * mu, rel_tol=1e-10):
        return "Exponential: the condition var = mu^2 is not satisfied"
    return None


def _why_not_gamma(mu, var):
    r = _why_not_positive_var("Gamma", var)
    if r is not None:
        return r
    if not (mu > 0):
        return "Gamma: the condition mu > 0 is not satisfied"
    return None


def _why_not_erlang(mu, var):
    return _why_not_gamma(mu, var)  # same rules, k just rounds to int


def _why_not_lognormal(mu, var):
    r = _why_not_positive_var("LogNormal", var)
    if r is not None:
        return r
    if not (mu > 0):
        return "LogNormal: the condition mu > 0 is not satisfied"
    return None


def _why_not_weibull(mu, var):
    r = _why_not_positive_var("Weibull", var)
    if r is not None:
        return r
    if not (mu > 0):
        return "Weibull: the condition mu > 0 is not satisfied"
    return None


def _why_not_frechet(mu, var):
    r = _why_not_positive_var("Frechet", var)
    if r is not None:
        return r
    if not (mu > 0):
        return "Frechet: the condition mu > 0 is not satisfied"
    return None


def _why_not_chisq(mu, var):
    r = _why_not_positive_var("Chisq", var)
    if r is not None:
        return r
    if not (mu > 0):
        return "Chisq: the condition mu > 0 is not satisfied"
    if not float(mu).is_integer():
        return "Chisq: the condition mu in N is not satisfied"
    if not math.isclose(var, 2 * mu, rel_tol=1e-10):
        return "Chisq: the condition var = 2*mu is not satisfied"
    return None


def _why_not_chi(mu, var):
    r = _why_not_positive_var("Chi", var)
    if r is not None:
        return r
    if not (mu > 0):
        return "Chi: the condition mu > 0 is not satisfied"
    # mu = sqrt(2) * Gamma((k+1)/2) / Gamma(k/2) where k = mu^2 + var
    k = mu * mu + var
    if not (k > 0):
        return "Chi: derived k <= 0"
    try:
        predicted = math.sqrt(2.0) * _gamma_fn((k + 1) / 2) / _gamma_fn(k / 2)
    except (OverflowError, ValueError):
        return "Chi: cannot evaluate gamma function at derived k"
    if not math.isclose(predicted, mu, rel_tol=1e-8, abs_tol=1e-10):
        return "Chi: the moment identity is not satisfied"
    return None


def _why_not_rayleigh(mu, var):
    r = _why_not_positive_var("Rayleigh", var)
    if r is not None:
        return r
    if not (mu > 0):
        return "Rayleigh: the condition mu > 0 is not satisfied"
    target_cv = math.sqrt((4 - math.pi) / math.pi)
    actual_cv = math.sqrt(var) / mu
    if not math.isclose(actual_cv, target_cv, rel_tol=1e-8):
        return "Rayleigh: the condition CV = sqrt((4-pi)/pi) is not satisfied"
    return None


def _why_not_fdist(mu, var):
    r = _why_not_positive_var("FDist", var)
    if r is not None:
        return r
    if not (mu > 1):
        return "FDist: the condition mu > 1 is not satisfied"
    if not (mu < 2):
        return "FDist: the condition mu < 2 is not satisfied"
    bound = mu * mu * (mu - 1) / (2 - mu)
    if not (var > bound):
        return f"FDist: the condition var > mu^2*(mu-1)/(2-mu) = {bound} is not satisfied"
    return None


def _why_not_inverse_gamma(mu, var):
    r = _why_not_positive_var("InverseGamma", var)
    if r is not None:
        return r
    if not (mu > 0):
        return "InverseGamma: the condition mu > 0 is not satisfied"
    return None


def _why_not_pareto(mu, var):
    r = _why_not_positive_var("Pareto", var)
    if r is not None:
        return r
    if not (mu > 0):
        return "Pareto: the condition mu > 0 is not satisfied"
    cv2 = var / (mu * mu)
    alpha = 1 + math.sqrt(1 + 1 / cv2)
    if not (alpha > 2):
        return "Pareto: derived alpha must be > 2 (variance infinite otherwise)"
    return None


def _why_not_folded_normal(mu, var):
    r = _why_not_positive_var("FoldedNormal", var)
    if r is not None:
        return r
    if not (mu > 0):
        return "FoldedNormal: the condition mu > 0 is not satisfied"
    return None


def _why_not_beta(mu, var):
    r = _why_not_positive_var("Beta", var)
    if r is not None:
        return r
    if not (0 < mu < 1):
        return "Beta: the condition 0 < mu < 1 is not satisfied"
    if not (var < mu * (1 - mu)):
        return "Beta: the condition var < mu*(1-mu) is not satisfied"
    return None


def _why_not_binomial(mu, var):
    r = _why_not_positive_var("Binomial", var)
    if r is not None:
        return r
    if not (var < mu):
        return "Binomial: the condition var < mu is not satisfied"
    if not (mu > 0):
        return "Binomial: the condition mu > 0 is not satisfied"
    n_raw = mu * mu / (mu - var)
    if not math.isclose(n_raw, round(n_raw), rel_tol=1e-8, abs_tol=1e-8):
        return f"Binomial: mu^2/(mu-var) must be a positive integer (got n ~ {n_raw})"
    return None


def _why_not_poisson(mu, var):
    r = _why_not_positive_var("Poisson", var)
    if r is not None:
        return r
    if not math.isclose(mu, var, rel_tol=1e-10):
        return "Poisson: the condition mu = var is not satisfied"
    return None


def _why_not_negative_binomial(mu, var):
    r = _why_not_positive_var("NegativeBinomial", var)
    if r is not None:
        return r
    if not (mu > 0):
        return "NegativeBinomial: the condition mu > 0 is not satisfied"
    if not (var > mu):
        return "NegativeBinomial: the condition var > mu is not satisfied"
    return None


def _why_not_geometric(mu, var):
    r = _why_not_positive_var("Geometric", var)
    if r is not None:
        return r
    if not (mu > 0):
        return "Geometric: the condition mu > 0 is not satisfied"
    if not math.isclose(var, mu * (1 + mu), rel_tol=1e-10):
        return "Geometric: the condition var = mu*(1+mu) is not satisfied"
    return None


def _why_not_discrete_uniform(mu, var):
    r = _why_not_positive_var("DiscreteUniform", var)
    if r is not None:
        return r
    n_raw = -1 + math.sqrt(1 + 12 * var)
    if not (math.isclose(n_raw, round(n_raw), abs_tol=1e-8) and round(n_raw) >= 0):
        return f"DiscreteUniform: b-a must be a non-negative integer (got n ~ {n_raw})"
    n = round(n_raw)
    a_raw = mu - n / 2
    if not math.isclose(a_raw, round(a_raw), abs_tol=1e-8):
        return f"DiscreteUniform: lower bound must be an integer (got a ~ {a_raw})"
    return None


# ---------------------------------------------------------------------------
# Dispatch table (canonical-name -> rule)
# ---------------------------------------------------------------------------

_RULES = {
    "normal":            _why_not_normal,
    "uniform":           _why_not_uniform,
    "logistic":          _why_not_logistic,
    "laplace":           _why_not_laplace,
    "gumbel":            _why_not_gumbel,
    "sym_triangular":    _why_not_sym_triangular,
    "tdist":             _why_not_tdist,
    "cauchy":            _why_not_cauchy,
    "exponential":       _why_not_exponential,
    "gamma":             _why_not_gamma,
    "erlang":            _why_not_erlang,
    "lognormal":         _why_not_lognormal,
    "weibull":           _why_not_weibull,
    "frechet":           _why_not_frechet,
    "chisq":             _why_not_chisq,
    "chi":               _why_not_chi,
    "rayleigh":          _why_not_rayleigh,
    "fdist":             _why_not_fdist,
    "inverse_gamma":     _why_not_inverse_gamma,
    "pareto":            _why_not_pareto,
    "folded_normal":     _why_not_folded_normal,
    "beta":              _why_not_beta,
    "binomial":          _why_not_binomial,
    "poisson":           _why_not_poisson,
    "negative_binomial": _why_not_negative_binomial,
    "geometric":         _why_not_geometric,
    "discrete_uniform":  _why_not_discrete_uniform,
}


def why_not_mean_var(name, mu, var):
    """Return ``None`` if `(mu, var)` is feasible for `name`, else a reason string."""
    rule = _RULES.get(name)
    if rule is None:
        return f"{name}: distribution not supported"
    return rule(mu, var)


def exists_mean_var(name, mu, var):
    """Boolean predicate. ``True`` iff `(mu, var)` is feasible for `name`."""
    return why_not_mean_var(name, mu, var) is None


def require_mean_var(name, mu, var):
    """Raise ``ValueError`` with the feasibility reason when infeasible."""
    reason = why_not_mean_var(name, mu, var)
    if reason is not None:
        raise ValueError(reason)

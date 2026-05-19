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

from ._langevin import truncexp_max_var


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


def _why_not_discrete_sym_triangular(mu, var):
    r = _why_not_positive_var("DiscreteSymTriangular", var)
    if r is not None:
        return r
    if not math.isclose(mu, round(mu), abs_tol=1e-8):
        return f"DiscreteSymTriangular: mu must be an integer (got {mu})"
    n_raw = -1 + math.sqrt(1 + 6 * var)
    if not (math.isclose(n_raw, round(n_raw), abs_tol=1e-8) and round(n_raw) >= 0):
        return (f"DiscreteSymTriangular: half-width n = -1 + sqrt(1+6*var) "
                f"must be a non-negative integer (got n ~ {n_raw})")
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
    "discrete_sym_triangular": _why_not_discrete_sym_triangular,
}


def why_not_mean_var(name, mu, var):
    """Return ``None`` if `(mu, var)` is feasible for `name`, else a reason string."""
    rule = _RULES.get(name)
    if rule is None:
        return f"{name}: distribution not supported"
    return rule(mu, var)


# ---------------------------------------------------------------------------
# Truncated location-scale feasibility (Langevin envelope)
# ---------------------------------------------------------------------------
# For Normal / Laplace / Logistic truncated to [lo, hi], the feasibility region
# is the truncated-exponential dome:
#   - Bounded:    var < sigma2_max(mu) per the inverse-Langevin envelope.
#   - Half-below: var < (mu - lo)^2 (the exponential bound).
#   - Half-above: var < (hi - mu)^2.
# Mirrors `_why_not_truncexp_envelope` and `_why_not_half_trunc_exp_envelope`
# in DistributionsFactories.jl/src/dist_exists.jl.

_TRUNC_LOCSCALE_FAMILIES = {"normal", "laplace", "logistic"}


def _why_not_truncated_locscale(name, mu, var, lo, hi):
    """Feasibility reason (or None) for ``name`` truncated to ``[lo, hi]``."""
    label = name.capitalize()
    if not (var > 0):
        return f"Truncated {label}: var > 0 is not satisfied"

    lo_finite = math.isfinite(lo)
    hi_finite = math.isfinite(hi)

    if lo_finite and hi_finite:
        if not (lo < mu < hi):
            return f"Truncated {label}: mu must be in ({lo}, {hi})"
        sigma2_max = truncexp_max_var(lo, hi, mu)
        # Tolerance: the envelope is approached (Normal/Logistic) or attained
        # (Laplace) at the boundary. 1e-8 keeps the boundary itself feasible
        # without admitting clearly-infeasible interiors. Matches Julia.
        if var > sigma2_max * (1 + 1e-8):
            return (f"Truncated {label}: var = {var} exceeds the Langevin "
                    f"feasibility envelope sigma2_max ~= {sigma2_max:.6g} at "
                    f"mu = {mu} on [{lo}, {hi}]. The Normal/Laplace/Logistic "
                    f"families share this truncated-exponential upper bound.")
        return None

    if lo_finite and not hi_finite:
        if not (mu > lo):
            return f"Truncated {label}: mu must be > lo = {lo}"
        gap = mu - lo
        # Laplace attains the boundary; Normal/Logistic only approach it. Apply
        # a small slack on the comparison so the Laplace boundary case is
        # admitted.
        if name == "laplace":
            if var > gap ** 2 * (1 + 1e-10):
                return (f"Truncated Laplace on [{lo}, inf): var must be "
                        f"<= (mu - lo)^2 = {gap ** 2} (exponential bound, attained)")
        else:
            if var >= gap ** 2:
                return (f"Truncated {label} on [{lo}, inf): var must be "
                        f"< (mu - lo)^2 = {gap ** 2} (exponential-tail bound)")
        return None

    if not lo_finite and hi_finite:
        if not (mu < hi):
            return f"Truncated {label}: mu must be < hi = {hi}"
        gap = hi - mu
        if name == "laplace":
            if var > gap ** 2 * (1 + 1e-10):
                return (f"Truncated Laplace on (-inf, {hi}]: var must be "
                        f"<= (hi - mu)^2 = {gap ** 2} (exponential bound, attained)")
        else:
            if var >= gap ** 2:
                return (f"Truncated {label} on (-inf, {hi}]: var must be "
                        f"< (hi - mu)^2 = {gap ** 2} (exponential-tail bound)")
        return None

    # Both endpoints infinite: untruncated, defer to the family's own predicate.
    return _RULES[name](mu, var)


def why_not_mean_var_on_support(name, mu, var, lo, hi):
    """Feasibility reason for ``name`` placed on ``[lo, hi]`` (structural).

    Mirrors Julia's `_dist_exists_on_support`. This is a **structural** check:
    it verifies that the family can be placed on the requested support (via
    affine transform or truncation) and standardizes moments back to the
    natural support for the base feasibility predicate.

    The Langevin envelope dome for Truncated{Normal/Laplace/Logistic} is
    intentionally **not** applied here — Julia only applies the envelope when
    you pass a ``Truncated{<:Normal}`` instance, not a ``Type + support=``.
    The constructor path (``make_dist``) does apply it for clean errors via
    ``why_not_truncated_locscale``.
    """
    from ._registry import SUPPORT_TYPE
    natural = SUPPORT_TYPE.get(name)
    if natural is None:
        return f"{name}: distribution not supported"

    if natural == "real":
        # Real-line family on any support: structural check only.
        return _RULES[name](mu, var)

    if natural == "positive":
        if math.isfinite(lo) and math.isinf(hi):
            return _RULES[name](mu - lo, var)
        if math.isinf(lo) and math.isfinite(hi):
            return _RULES[name](hi - mu, var)
        if math.isfinite(lo) and math.isfinite(hi):
            if lo < 0:
                return (f"Cannot place {name!r} (natural [0, inf)) on "
                        f"[{lo}, {hi}] with lo < 0")
            return _RULES[name](mu, var)
        return f"Cannot place {name!r} (natural [0, inf)) on (-inf, inf)"

    if natural == "unit":
        if math.isfinite(lo) and math.isfinite(hi):
            w = hi - lo
            return _RULES[name]((mu - lo) / w, var / w ** 2)
        return f"Cannot place {name!r} (natural [0, 1]) on unbounded interval"

    # Discrete: deferred to constructor for now.
    return None


def why_not_truncated_locscale(name, mu, var, lo, hi):
    """Langevin envelope feasibility for an *explicitly* truncated locscale dist.

    Use this when the caller is building a truncated Normal/Laplace/Logistic
    (rather than placing the type on an interval). Mirrors Julia's
    ``_why_not_dist_from_mean_var(d::Truncated{<:Normal}, ...)``.
    """
    if name not in _TRUNC_LOCSCALE_FAMILIES:
        return f"{name}: Langevin envelope only applies to Normal/Laplace/Logistic"
    return _why_not_truncated_locscale(name, mu, var, lo, hi)


def exists_mean_var(name, mu, var):
    """Boolean predicate. ``True`` iff `(mu, var)` is feasible for `name`."""
    return why_not_mean_var(name, mu, var) is None


def require_mean_var(name, mu, var):
    """Raise ``ValueError`` with the feasibility reason when infeasible."""
    reason = why_not_mean_var(name, mu, var)
    if reason is not None:
        raise ValueError(reason)


def exists_mean_var_on_support(name, mu, var, lo, hi):
    """Boolean predicate. ``True`` iff ``name`` is feasible on ``[lo, hi]``."""
    return why_not_mean_var_on_support(name, mu, var, lo, hi) is None


def require_mean_var_on_support(name, mu, var, lo, hi):
    """Raise ``ValueError`` with the feasibility reason when infeasible on a support."""
    reason = why_not_mean_var_on_support(name, mu, var, lo, hi)
    if reason is not None:
        raise ValueError(reason)

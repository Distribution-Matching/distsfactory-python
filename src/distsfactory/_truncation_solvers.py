"""Solvers for truncated location-scale families.

Mirrors the structure of `_solve_truncated_unit` / `_solve_truncated_mean_var`
in DistributionsFactories.jl. Given a target ``(lo, hi, mu, var)`` and a base
family (Normal/Laplace/Logistic), find the parent ``(loc, scale)`` whose
truncation to ``[lo, hi]`` matches the requested moments.

We standardize to ``[-0.5, 0.5]`` for numerical stability, then 2D Newton on
``(loc, log scale)`` with quadrature-evaluated truncated moments.
"""

import math
import numpy as np
from scipy import stats
from scipy.integrate import quad


_FAMILIES = {
    "normal": stats.norm,
    "laplace": stats.laplace,
    "logistic": stats.logistic,
}


def _truncated_moments(family, loc, scale, lo, hi):
    """Return ``(mean, var)`` of ``family(loc, scale)`` truncated to ``[lo, hi]``."""
    rv = family(loc=loc, scale=scale)
    Z = rv.cdf(hi) - rv.cdf(lo)
    if Z <= 0:
        return float("nan"), float("nan")
    m, _ = quad(lambda x: x * rv.pdf(x), lo, hi)
    m2, _ = quad(lambda x: x ** 2 * rv.pdf(x), lo, hi)
    mean = m / Z
    var = m2 / Z - mean ** 2
    return mean, var


def _solve_canonical(family, mu_std, var_std, maxiter=200, tol=1e-10, h=1e-7):
    """Solve the canonical 2D problem on ``[-0.5, 0.5]``.

    Returns ``(loc_std, scale_std)`` such that ``family(loc, scale)``
    truncated to ``[-0.5, 0.5]`` has mean ``mu_std`` and variance ``var_std``.
    """
    if not (-0.5 < mu_std < 0.5):
        raise ValueError("standardized mean must be in (-0.5, 0.5)")
    if not (var_std > 0):
        raise ValueError("standardized variance must be > 0")

    x = np.array([float(mu_std), math.log(math.sqrt(var_std))])
    converged = False
    for _ in range(maxiter):
        m, v = _truncated_moments(family, x[0], math.exp(x[1]), -0.5, 0.5)
        F = np.array([m - mu_std, v - var_std])
        if np.max(np.abs(F)) < tol:
            converged = True
            break

        J = np.empty((2, 2))
        for j in range(2):
            xp = x.copy()
            xp[j] += h
            mp, vp = _truncated_moments(family, xp[0], math.exp(xp[1]), -0.5, 0.5)
            Fp = np.array([mp - mu_std, vp - var_std])
            J[:, j] = (Fp - F) / h

        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            break

        # Damped step
        step = 1.0
        for _ in range(20):
            x_new = x + step * dx
            mn, vn = _truncated_moments(family, x_new[0], math.exp(x_new[1]), -0.5, 0.5)
            Fn = np.array([mn - mu_std, vn - var_std])
            if np.all(np.isfinite(Fn)) and np.max(np.abs(Fn)) < np.max(np.abs(F)):
                break
            step *= 0.5
        x = x + step * dx

    if not converged:
        raise RuntimeError(
            f"Truncated {family.name}: 2D Newton did not converge after {maxiter} "
            f"iterations (target mu={mu_std}, var={var_std})"
        )
    return x[0], math.exp(x[1])


def solve_truncated_locscale(name, lo, hi, mu, var):
    """Match ``(mu, var)`` for a location-scale family truncated to ``[lo, hi]``.

    Dispatches between three solvers:

    - **Two-sided** (both ``lo`` and ``hi`` finite): standardize to ``[-0.5, 0.5]``,
      2D Newton on the unit interval, un-standardize.
    - **Half-truncated** (one endpoint infinite): solve in user coordinates with
      a generic 2D Newton on ``(loc, log scale)``.

    Returns:

    - For ``"normal"``: a frozen ``scipy.stats.truncnorm`` (scipy has native
      support; this gives users a first-class scipy distribution).
    - For ``"laplace"`` / ``"logistic"``: a ``_TruncatedDist`` wrapper (scipy
      has no native truncated form for these).
    """
    family = _FAMILIES[name]

    if math.isfinite(lo) and math.isfinite(hi):
        c = (lo + hi) / 2
        w = hi - lo
        loc_std, scale_std = _solve_canonical(family, (mu - c) / w, var / w ** 2)
        loc = c + w * loc_std
        scale = w * scale_std
    else:
        loc, scale = _solve_truncated_locscale_halfinf(family, lo, hi, mu, var)

    if name == "normal":
        # Scipy's truncnorm uses standardized bounds: a = (lo - loc) / scale.
        a_std = (lo - loc) / scale if math.isfinite(lo) else -math.inf
        b_std = (hi - loc) / scale if math.isfinite(hi) else math.inf
        return stats.truncnorm(a=a_std, b=b_std, loc=loc, scale=scale)

    from ._support import _TruncatedDist
    return _TruncatedDist(family(loc=loc, scale=scale), lo, hi)


def _solve_truncated_locscale_halfinf(family, lo, hi, mu_t, var_t,
                                       maxiter=200, tol=1e-10, h=1e-7):
    """Half-truncated location-scale solver. Returns ``(parent_loc, parent_scale)``.

    Solves in user coordinates because there's no two-sided affine standardization
    when one endpoint is infinite. Initial guess: ``(mu_t, sqrt(var_t))`` — the
    untruncated parent moments, which is good when the truncation tail mass is
    modest.
    """
    x = np.array([float(mu_t), math.log(math.sqrt(var_t))])

    def F(x_):
        m, v = _truncated_moments(family, x_[0], math.exp(x_[1]), lo, hi)
        return np.array([m - mu_t, v - var_t])

    converged = False
    for _ in range(maxiter):
        Fx = F(x)
        if np.all(np.isfinite(Fx)) and np.max(np.abs(Fx)) < tol:
            converged = True
            break
        J = np.empty((2, 2))
        for j in range(2):
            xp = x.copy()
            xp[j] += h
            J[:, j] = (F(xp) - Fx) / h
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break
        step = 1.0
        for _ in range(20):
            x_new = x + step * dx
            Fn = F(x_new)
            if np.all(np.isfinite(Fn)) and np.max(np.abs(Fn)) < np.max(np.abs(Fx)):
                break
            step *= 0.5
        x = x + step * dx

    if not converged:
        raise RuntimeError(
            f"Truncated {family.name}: half-truncated 2D Newton did not converge "
            f"on [{lo}, {hi}] (target mu={mu_t}, var={var_t})"
        )
    return x[0], math.exp(x[1])


def solve_truncated_tdist_half(df, lo, hi, mu, var, maxiter=200, tol=1e-10, h=1e-7):
    """Half-truncated location-scale Student-t.

    For ``lo`` finite and ``hi = +inf``: find ``(loc, scale)`` such that
    ``mu + scale * TDist(df)`` truncated to ``[lo, +inf)`` has mean ``mu``
    and variance ``var``. The reflection case (``hi`` finite, ``lo = -inf``)
    is handled by flipping. ``df`` must be > 2.
    """
    from scipy.integrate import quad

    if not (df > 2):
        raise ValueError(f"Truncated TDist requires df > 2 (got {df})")

    lo_finite = math.isfinite(lo)
    hi_finite = math.isfinite(hi)
    if lo_finite and hi_finite:
        raise NotImplementedError(
            "Two-sided Truncated{TDist} factory is not implemented "
            "(Julia tracks this as a known gap as well)."
        )
    if not lo_finite and not hi_finite:
        # No truncation: untruncated location-scale t.
        scale = math.sqrt(var * (df - 2) / df)
        return stats.t(df=df, loc=mu, scale=scale)

    # Reflection: solve the lower-truncated problem then flip.
    if hi_finite and not lo_finite:
        # Y = -X; truncated(t, -inf, hi) <-> reflected lower-truncated at -hi.
        d_ref = solve_truncated_tdist_half(df, -hi, math.inf, -mu, var,
                                            maxiter=maxiter, tol=tol, h=h)
        # d_ref is _TruncatedDist; reflect its parent.
        parent_ref = d_ref.parent
        loc_ref, scale_ref = parent_ref.kwds["loc"], parent_ref.kwds["scale"]
        parent = stats.t(df=df, loc=-loc_ref, scale=scale_ref)
        from ._support import _TruncatedDist
        return _TruncatedDist(parent, -math.inf, hi)

    # Lower-truncated, hi = +inf
    sigma_bar = math.sqrt(var)
    z = (mu - lo) / sigma_bar  # canonical target mean on [0, +inf)
    # Canonical: parent mu_std + sigma_std*t(df) truncated to [0, +inf), match (z, 1).

    x = np.array([0.0, 0.0])  # (mu_std, log sigma_std)

    def moments(x_):
        mu_std = x_[0]
        sigma_std = math.exp(x_[1])
        # Work in standard-t coords: substitute u = (X-mu_std)/sigma_std,
        # truncation X >= 0 -> u >= -mu_std/sigma_std.
        tlo = -mu_std / sigma_std
        Z, _ = quad(lambda t: stats.t.pdf(t, df), tlo, math.inf)
        if Z <= 0:
            return float("nan"), float("nan")
        m1, _ = quad(lambda t: (mu_std + sigma_std * t) * stats.t.pdf(t, df), tlo, math.inf)
        m2, _ = quad(lambda t: (mu_std + sigma_std * t) ** 2 * stats.t.pdf(t, df), tlo, math.inf)
        m = m1 / Z
        v = m2 / Z - m ** 2
        return m, v

    converged = False
    for _ in range(maxiter):
        m, v = moments(x)
        F = np.array([m - z, v - 1.0])
        if not np.all(np.isfinite(F)):
            raise RuntimeError(
                f"Truncated TDist half-below: residual non-finite at df={df}, x={x}"
            )
        if np.max(np.abs(F)) < tol:
            converged = True
            break

        J = np.empty((2, 2))
        for j in range(2):
            xp = x.copy()
            xp[j] += h
            mp, vp = moments(xp)
            J[:, j] = (np.array([mp - z, vp - 1.0]) - F) / h

        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            break

        step = 1.0
        for _ in range(20):
            x_new = x + step * dx
            mn, vn = moments(x_new)
            Fn = np.array([mn - z, vn - 1.0])
            if np.all(np.isfinite(Fn)) and np.max(np.abs(Fn)) < np.max(np.abs(F)):
                break
            step *= 0.5
        x = x + step * dx

    if not converged:
        raise RuntimeError(
            f"Truncated TDist half-below: 2D Newton did not converge for df={df}, target z={z}"
        )

    mu_std, sigma_std = x[0], math.exp(x[1])
    parent_loc = lo + sigma_bar * mu_std
    parent_scale = sigma_bar * sigma_std
    parent = stats.t(df=df, loc=parent_loc, scale=parent_scale)
    from ._support import _TruncatedDist
    return _TruncatedDist(parent, lo, math.inf)


def solve_truncated_generic(name, lo, hi, mu, var, maxiter=200, tol=1e-10, h=1e-7):
    """Solve a truncated distribution for moments via Newton on the canonical params.

    Dispatches by the number of free parameters in the family's canonical set:

    - **1 free param** (Exponential, Rayleigh, Chisq, …): the truncated variance is
      determined by the mean, so we 1D-solve for the parent's free param to match
      ``mu`` and check that the achieved variance matches the requested ``var``
      to a relative tolerance.
    - **2 free params** (Gamma, Beta, LogNormal, …): 2D Newton on ``(p1, p2)`` to
      match ``(mu, var)`` directly.

    For Exponential we return a native ``scipy.stats.truncexpon`` frozen instead
    of a ``_TruncatedDist`` wrapper.
    """
    from scipy.integrate import quad
    from ._distributions import DIST_HANDLERS

    if name == "exponential":
        return _solve_truncated_exponential(lo, hi, mu, var, maxiter=maxiter, tol=tol)

    handler = DIST_HANDLERS[name]
    # Build an initial parent matching (mu, var) on the natural support; we'll
    # iterate from there. Use the canonical from_mean_var when feasible.
    try:
        seed = handler.from_mean_var(mu, var)
    except (ArithmeticError, ValueError, RuntimeError):
        # If the target falls outside the natural-support feasibility region,
        # we still try with a seed that's known-valid for the family.
        seed = handler.from_mean_var(mu if mu > 0 else 1.0, max(var, 1e-3))

    # Use the family's canonical parameters; fall back to "all kwds minus loc"
    # for cases not in the table.
    from ._registry import CANONICAL_PARAMS
    if name in CANONICAL_PARAMS:
        free_kwds = [k for k in CANONICAL_PARAMS[name] if k in seed.kwds]
    else:
        free_kwds = [k for k in seed.kwds if k != "loc"]
    x = np.array([float(seed.kwds[k]) for k in free_kwds])

    if len(free_kwds) == 1:
        return _solve_truncated_generic_1d(
            handler, seed, free_kwds[0], x[0], lo, hi, mu, var, maxiter=maxiter, tol=tol
        )

    def build(params_vec):
        kwds = dict(seed.kwds)
        for k, v in zip(free_kwds, params_vec):
            kwds[k] = float(v)
        return seed.dist(*seed.args, **kwds)

    def moments(params_vec):
        d = build(params_vec)
        Z = d.cdf(hi) - d.cdf(lo)
        if Z <= 0:
            return float("nan"), float("nan")
        m, _  = quad(lambda x_: x_ * d.pdf(x_), lo, hi)
        m2, _ = quad(lambda x_: x_ ** 2 * d.pdf(x_), lo, hi)
        mean = m / Z
        v = m2 / Z - mean ** 2
        return mean, v

    converged = False
    for _ in range(maxiter):
        m, v = moments(x)
        F = np.array([m - mu, v - var])
        if np.all(np.isfinite(F)) and np.max(np.abs(F)) < tol:
            converged = True
            break

        J = np.empty((2, 2))
        for j in range(2):
            xp = x.copy()
            xp[j] += h
            mp, vp = moments(xp)
            J[:, j] = (np.array([mp - mu, vp - var]) - F) / h

        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            break

        step = 1.0
        for _ in range(20):
            x_new = x + step * dx
            mn, vn = moments(x_new)
            Fn = np.array([mn - mu, vn - var])
            if np.all(np.isfinite(Fn)) and np.max(np.abs(Fn)) < np.max(np.abs(F)):
                break
            step *= 0.5
        x = x + step * dx

    if not converged:
        raise RuntimeError(
            f"Truncated {name}: 2D Newton did not converge after {maxiter} "
            f"iterations (target mean={mu}, var={var}); last residual was {F}"
        )

    from ._support import _TruncatedDist
    return _TruncatedDist(build(x), lo, hi)


def _solve_truncated_exponential(lo, hi, mu, var, maxiter=200, tol=1e-10):
    """Solve a truncated Exponential and return a native ``scipy.stats.truncexpon``.

    The truncated exponential has one free parameter (the parent scale theta), so
    variance is determined by the mean. If the caller supplied a ``var`` that
    disagrees with the achieved variance, surface a clear error.
    """
    from scipy.optimize import brentq
    from scipy.integrate import quad

    if lo < 0:
        raise ValueError(
            f"Truncated Exponential: lower bound must be >= 0 (got lo={lo})"
        )
    # Parent: stats.expon(loc=lo, scale=theta). Truncated support is [lo, hi] where
    # hi may be +inf (no truncation above) or finite. Shift to origin by lo.
    shift = lo
    width = (hi - lo) if math.isfinite(hi) else math.inf
    mu_shifted = mu - lo

    if mu_shifted <= 0:
        raise ValueError(
            f"Truncated Exponential: mu must be > lo (got mu={mu}, lo={lo})"
        )
    if math.isfinite(width) and mu_shifted >= width:
        raise ValueError(
            f"Truncated Exponential on [{lo}, {hi}]: mu must be in ({lo}, {hi})"
        )

    def truncated_mean(theta):
        # Mean of Exp(scale=theta) truncated to [0, width]; closed form when finite.
        if not math.isfinite(width):
            return theta  # ordinary exponential
        r = width / theta
        # Z = 1 - exp(-r). m1 = theta * (1 - (r+1)*exp(-r)). mean = m1 / Z.
        Z = 1 - math.exp(-r)
        m1 = theta * (1 - (r + 1) * math.exp(-r))
        return m1 / Z

    def residual(theta):
        return truncated_mean(theta) - mu_shifted

    # Bracket: theta in (0, large). For large theta, truncated_mean -> width/2.
    lo_t = mu_shifted * 0.01
    hi_t = max(mu_shifted * 100, width if math.isfinite(width) else mu_shifted * 100)
    while residual(lo_t) > 0 and lo_t > 1e-12:
        lo_t /= 2
    while residual(hi_t) < 0:
        hi_t *= 2
    theta = brentq(residual, lo_t, hi_t, xtol=tol)

    # Build the scipy truncexpon and check variance consistency
    if math.isfinite(width):
        b_std = width / theta
        d = stats.truncexpon(b=b_std, loc=shift, scale=theta)
    else:
        d = stats.expon(loc=shift, scale=theta)
    achieved_var = float(d.var())
    if not math.isclose(achieved_var, var, rel_tol=1e-3, abs_tol=1e-9):
        raise ValueError(
            f"Truncated Exponential on [{lo}, {hi}]: variance is determined by "
            f"the mean; achieved var={achieved_var:.6g}, requested var={var:.6g}"
        )
    return d


def _solve_truncated_generic_1d(handler, seed, free_kwd, x0, lo, hi, mu, var,
                                 maxiter=200, tol=1e-10):
    """1D solver for truncated 1-parameter families (Rayleigh, Chisq, etc.).

    Solves for the single free param to match ``mu`` (variance is determined),
    then checks the achieved variance against the requested ``var``.
    """
    from scipy.optimize import brentq
    from scipy.integrate import quad
    from ._support import _TruncatedDist

    def build(p):
        kwds = dict(seed.kwds)
        kwds[free_kwd] = float(p)
        return seed.dist(*seed.args, **kwds)

    def truncated_mean(p):
        d = build(p)
        Z = d.cdf(hi) - d.cdf(lo)
        if Z <= 0:
            return float("nan")
        m, _ = quad(lambda x: x * d.pdf(x), lo, hi)
        return m / Z

    def residual(p):
        return truncated_mean(p) - mu

    # Find a bracket geometrically outward from x0
    lo_p = x0 * 0.01
    hi_p = x0 * 100
    while residual(lo_p) > 0 and lo_p > 1e-12:
        lo_p /= 2
    while residual(hi_p) < 0:
        hi_p *= 2
    sol = brentq(residual, lo_p, hi_p, xtol=tol)

    parent = build(sol)
    td = _TruncatedDist(parent, lo, hi)
    achieved_var = td.var()
    if not math.isclose(achieved_var, var, rel_tol=1e-3, abs_tol=1e-9):
        family_label = type(parent.dist).__name__
        raise ValueError(
            f"Truncated 1-param family ({family_label}) on [{lo}, {hi}]: "
            f"variance is determined by the mean; achieved var={achieved_var:.6g}, "
            f"requested var={var:.6g}"
        )
    return td

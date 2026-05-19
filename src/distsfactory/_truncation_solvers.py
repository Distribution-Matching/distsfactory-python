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

    Standardize to ``[-0.5, 0.5]``, solve, un-standardize.
    """
    family = _FAMILIES[name]
    c = (lo + hi) / 2
    w = hi - lo
    loc_std, scale_std = _solve_canonical(family, (mu - c) / w, var / w ** 2)
    loc = c + w * loc_std
    scale = w * scale_std
    # Return a TruncatedDist wrapping the parent
    from ._support import _TruncatedDist
    return _TruncatedDist(family(loc=loc, scale=scale), lo, hi)


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
        parent_ref = d_ref._inner
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
    """Solve a non-location-scale family truncated to ``[lo, hi]`` for moments.

    Uses 2D Newton on the parent's *canonical* parameters (e.g. Gamma's
    ``(a, scale)``, Beta's ``(a, b)``) — works in user coordinates because
    there's no closed-form affine standardization for these families.
    """
    from scipy.integrate import quad
    from ._distributions import DIST_HANDLERS

    handler = DIST_HANDLERS[name]
    # Build an initial parent matching (mu, var) on the natural support; we'll
    # iterate from there. Use the canonical from_mean_var when feasible.
    try:
        seed = handler.from_mean_var(mu, var)
    except Exception:
        # If the target falls outside the natural-support feasibility region,
        # we still try with a seed that's known-valid for the family.
        seed = handler.from_mean_var(mu if mu > 0 else 1.0, max(var, 1e-3))

    free_kwds = [k for k in seed.kwds if k != "loc"]  # the canonical params
    x = np.array([float(seed.kwds[k]) for k in free_kwds])

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

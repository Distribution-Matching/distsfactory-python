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

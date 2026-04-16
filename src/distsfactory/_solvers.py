"""Numerical solvers for parameter fitting."""

import numpy as np
from scipy.optimize import brentq


def find_root_1d(f, x0=0.0, bracket=None, **kwargs):
    """Find a root of a scalar function.

    If a bracket [a, b] is given (where f(a) and f(b) have opposite signs),
    uses Brent's method.  Otherwise, searches outward from x0 to find a
    bracket automatically, then applies Brent's method.
    """
    if bracket is not None:
        return brentq(f, bracket[0], bracket[1], **kwargs)

    # Auto-bracket: expand geometrically from x0
    a, b = x0 - 1.0, x0 + 1.0
    for _ in range(60):
        fa, fb = f(a), f(b)
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
            return brentq(f, a, b, **kwargs)
        a *= 2.0
        b *= 2.0
    raise RuntimeError(f"Could not bracket a root starting from x0={x0}")


def newton_2d(F, x0, maxiter=200, tol=1e-10, h=1e-7):
    """Damped Newton iteration for a 2D system F(x) = 0.

    Parameters
    ----------
    F : callable
        Takes a length-2 array, returns a length-2 array of residuals.
    x0 : array-like
        Initial guess (length 2).
    maxiter : int
        Maximum iterations.
    tol : float
        Convergence tolerance on max(|F|).
    h : float
        Step size for finite-difference Jacobian.

    Returns
    -------
    x : ndarray
        Solution vector (length 2).
    """
    x = np.asarray(x0, dtype=float)
    for _ in range(maxiter):
        Fx = np.asarray(F(x), dtype=float)
        if np.max(np.abs(Fx)) < tol:
            return x

        # Numerical Jacobian
        J = np.empty((2, 2))
        for j in range(2):
            xp = x.copy()
            xp[j] += h
            J[:, j] = (np.asarray(F(xp)) - Fx) / h

        try:
            dx = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            raise RuntimeError("Singular Jacobian in Newton iteration")

        # Damped step
        step = 1.0
        for _ in range(20):
            x_new = x - step * dx
            Fx_new = np.asarray(F(x_new), dtype=float)
            if np.max(np.abs(Fx_new)) < np.max(np.abs(Fx)):
                break
            step *= 0.5
        x = x - step * dx

    raise RuntimeError(
        f"Newton iteration did not converge after {maxiter} iterations "
        f"(residual: {np.max(np.abs(F(x)))})"
    )

"""Partial distribution specification — Python analog of Julia's ``@dist`` / ``DistSpec``.

Users pin some scipy parameters and leave others to be solved from moment
constraints:

>>> from distsfactory import partial_dist, make_dist
>>> spec = partial_dist("gamma", a=3.0)        # pin shape, solve scale
>>> d = make_dist(spec, mean=5.0)
>>> round(d.mean(), 6)
5.0
>>> round(d.kwds["scale"], 4)
1.6667

The solver is generic: identify free scipy kwds, build the frozen dist as a
function of those kwds, and 1D-/2D-Newton on ``(mean, var)`` to match the
requested moments. No per-distribution code is needed beyond the family's
``mean()`` / ``var()`` implementations (which scipy provides).
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ._registry import resolve_dist, CANONICAL_PARAMS


def _canonical_params(name):
    """Return the canonical tunable parameter set for a distribution.

    Pulled from ``CANONICAL_PARAMS`` in the registry; matches the
    Julia parameterization (e.g. Gamma has (a, scale), not (a, loc, scale)).
    """
    if name not in CANONICAL_PARAMS:
        raise ValueError(f"No canonical params known for {name!r}")
    return CANONICAL_PARAMS[name]


@dataclass(frozen=True)
class PartialDist:
    """A scipy-frozen-like spec with some parameters fixed and some free.

    Use ``partial_dist(...)`` to construct one. Pass to ``make_dist(spec, mean=...,
    var=...)`` to solve the free parameters from moment constraints.
    """

    name: str
    fixed: Dict[str, float]

    def __repr__(self):
        fixed_str = ", ".join(f"{k}={v}" for k, v in self.fixed.items())
        return f"PartialDist({self.name!r}, {fixed_str})"

    def free_params(self):
        """Return the list of scipy kwd names still to be solved.

        Drawn from the canonical parameter set for this distribution
        (matches the Julia parameterization). ``loc`` on positive-support
        families is intentionally excluded from this list — to free it,
        use the ``support=`` keyword on ``make_dist`` instead.
        """
        return [n for n in _canonical_params(self.name) if n not in self.fixed]

    def fixed_params(self):
        """Return a dict of the fixed parameters."""
        return dict(self.fixed)


def partial_dist(name, **fixed):
    """Construct a ``PartialDist`` pinning some scipy parameters.

    Parameters
    ----------
    name : str
        Canonical distribution name (or alias). See ``DISTRIBUTIONS``.
    **fixed
        Scipy parameter names to pin (e.g. ``a=3.0`` for gamma's shape,
        ``loc=0.0`` for normal's mean).
    """
    # Validate the name eagerly.
    resolve_dist(name)
    return PartialDist(name=name.lower(), fixed=dict(fixed))


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def _build(scipy_dist, fixed, free_names, x):
    """Build a frozen dist from fixed + free kwd values."""
    kwds = dict(fixed)
    for name, val in zip(free_names, x):
        kwds[name] = float(val)
    # Discrete distributions don't take scale: drop it if it slipped in.
    return scipy_dist(**kwds)


def _moments(scipy_dist, fixed, free_names, x):
    try:
        d = _build(scipy_dist, fixed, free_names, x)
        return float(d.mean()), float(d.var())
    except Exception:
        return float("nan"), float("nan")


def _solve_one_free(scipy_dist, fixed, free_name, target_mean, target_var,
                     maxiter=200, tol=1e-10):
    """1 free parameter, possibly overdetermined by (mean, var)."""
    from scipy.optimize import brentq

    # Build target functions of x (a scalar).
    def mean_residual(x):
        m, _ = _moments(scipy_dist, fixed, [free_name], [x])
        return m - target_mean

    def var_residual(x):
        _, v = _moments(scipy_dist, fixed, [free_name], [x])
        return v - target_var

    # Strategy: if variance is given, try var first (more sensitive to scale
    # params); else solve from mean. Then verify the other.
    primary, secondary = (var_residual, mean_residual) if target_var is not None \
        else (mean_residual, var_residual)

    sol = _bracketed_brentq(primary, x0=1.0)
    if sol is None:
        # try mean instead if we started with var
        sol = _bracketed_brentq(secondary, x0=1.0)
    if sol is None:
        raise RuntimeError(
            f"Could not solve PartialDist({scipy_dist.name}, fixed={fixed}) "
            f"for free parameter {free_name!r}"
        )

    d = _build(scipy_dist, fixed, [free_name], [sol])
    # Tolerance check: if both mean and var were requested, ensure both match.
    if target_var is not None and target_mean is not None:
        if not math.isclose(float(d.mean()), target_mean, rel_tol=1e-4, abs_tol=1e-8):
            raise ValueError(
                f"PartialDist({scipy_dist.name}, fixed={fixed}): cannot satisfy "
                f"both mean={target_mean} and var={target_var} with 1 free parameter"
            )
    return d


def _bracketed_brentq(f, x0=1.0):
    """Expand a bracket from x0 outward until f changes sign, then brentq."""
    from scipy.optimize import brentq

    # First try a small positive bracket (scipy params are usually >0).
    candidates = [(1e-6, 1e3), (1e-9, 1e6), (-1e6, 1e6)]
    for a, b in candidates:
        try:
            fa, fb = f(a), f(b)
        except Exception:
            continue
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
            return brentq(f, a, b)
    return None


def _solve_two_free(scipy_dist, fixed, free_names, target_mean, target_var,
                     maxiter=200, tol=1e-10, h=1e-7):
    """2 free parameters from (mean, var) — 2D damped Newton on the free vars."""
    # Initial guess: equal split based on target moments. Use log-space when
    # the parameter looks positive-only (a heuristic from the free name).
    x = np.array([1.0, 1.0])

    def F(x_):
        m, v = _moments(scipy_dist, fixed, free_names, x_)
        return np.array([m - target_mean, v - target_var])

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
            Fp = F(xp)
            J[:, j] = (Fp - Fx) / h

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
            f"PartialDist({scipy_dist.name}, fixed={fixed}): 2D Newton did not "
            f"converge for free parameters {free_names} (target mean={target_mean}, "
            f"var={target_var}); last residual was {F(x)}"
        )

    return _build(scipy_dist, fixed, free_names, x)


def solve_partial(spec, target_mean=None, target_var=None):
    """Solve a ``PartialDist`` for the free parameters given moment targets."""
    name, scipy_dist = resolve_dist(spec.name)
    free_names = spec.free_params()

    if not free_names:
        # All params fixed: build and check.
        d = scipy_dist(**spec.fixed)
        if target_mean is not None and not math.isclose(
            float(d.mean()), target_mean, rel_tol=1e-6, abs_tol=1e-9
        ):
            raise ValueError(
                f"PartialDist({name}, {spec.fixed}) has mean {d.mean()} "
                f"but target_mean is {target_mean}"
            )
        return d

    if len(free_names) == 1:
        return _solve_one_free(scipy_dist, spec.fixed, free_names[0],
                               target_mean, target_var)

    if len(free_names) == 2:
        if target_mean is None or target_var is None:
            raise ValueError(
                f"PartialDist with 2 free parameters {free_names} needs both "
                f"mean and var (got mean={target_mean}, var={target_var})"
            )
        return _solve_two_free(scipy_dist, spec.fixed, free_names,
                                target_mean, target_var)

    raise ValueError(
        f"PartialDist has {len(free_names)} free parameters ({free_names}); "
        f"max 2 supported"
    )

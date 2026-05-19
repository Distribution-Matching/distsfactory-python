"""Langevin feasibility envelope for truncated location-scale families.

Port of `src/langevin.jl` in DistributionsFactories.jl. For any continuous,
unimodal, location-scale family with exponential tails (Normal, Laplace,
Logistic) truncated to a bounded interval ``[a, b]``, the set of achievable
``(mean, variance)`` pairs forms a dome whose upper envelope coincides with the
truncated-exponential family. The maximum variance at mean ``mu`` on ``[a, b]``
is

    sigma2_max(mu) = w^2 * L'(L^{-1}((c - mu) / w))

where ``c = (a + b) / 2``, ``w = (b - a) / 2``, and ``L`` is the Langevin
function ``L(x) = coth(x) - 1/x``.

A truncated Normal / Laplace / Logistic with mean ``mu`` and variance ``var``
on ``[a, b]`` exists iff ``a < mu < b`` and ``var < sigma2_max(mu)``.

For half-truncated cases (lo finite, hi = +inf), the exponential-tail bound
gives ``var < (mu - lo)^2``; the Laplace boundary is *attained* (not just
approached) when the parent location is past the truncation point.
"""

import math


def langevin(x):
    """Langevin function ``L(x) = coth(x) - 1/x``.

    Uses a Maclaurin series for ``|x| < 1e-4`` to avoid catastrophic
    cancellation: both ``coth(x)`` and ``1/x`` individually blow up like
    ``1/x`` while their difference is ``O(x)``.
    """
    if abs(x) < 1e-4:
        return x / 3 - x ** 3 / 45 + 2 * x ** 5 / 945 - x ** 7 / 4725
    return 1.0 / math.tanh(x) - 1.0 / x


def langevin_deriv(x):
    """Derivative ``L'(x) = 1/x**2 - 1/sinh(x)**2``.

    Uses a Maclaurin series for ``|x| < 1e-3`` to avoid the same kind of
    catastrophic cancellation near zero.
    """
    if abs(x) < 1e-3:
        return 1 / 3 - x ** 2 / 15 + 2 * x ** 4 / 189 - x ** 6 / 675
    s = math.sinh(x)
    return 1.0 / x ** 2 - 1.0 / (s * s)


def inv_langevin(y):
    """Inverse Langevin function ``L^{-1}(y)`` on ``y in (-1, 1)``.

    Initial guess: Cohen's (1991) [3/2] Pade approximant ``y(3 - y**2)/(1 - y**2)``,
    which is exact to ``O(y**3)`` near zero and captures the +/-1 pole structure
    at the domain edges. Plain Newton refinement; converges in well under
    50 iterations across the entire domain.
    """
    if not (abs(y) < 1):
        raise ValueError(f"inv_langevin: |y| must be < 1 (got {y})")
    if y == 0:
        return 0.0
    z = y * (3 - y ** 2) / (1 - y ** 2)
    for _ in range(50):
        f = langevin(z) - y
        if abs(f) < 1e-14:
            return z
        z -= f / langevin_deriv(z)
    return z


def truncexp_max_var(a, b, mu):
    """Upper-envelope variance at mean ``mu`` on ``[a, b]``.

    Shared feasibility boundary for truncated Normal / Laplace / Logistic:
    a distribution in any of these families with mean ``mu`` and variance
    ``var`` exists iff ``var < truncexp_max_var(a, b, mu)``.
    """
    if not (a < b):
        raise ValueError(f"truncexp_max_var: require a < b (got a={a}, b={b})")
    if not (a < mu < b):
        raise ValueError(f"truncexp_max_var: mu must lie strictly in ({a}, {b}) (got {mu})")
    c = (a + b) / 2
    w = (b - a) / 2
    z = inv_langevin((c - mu) / w)
    return w ** 2 * langevin_deriv(z)

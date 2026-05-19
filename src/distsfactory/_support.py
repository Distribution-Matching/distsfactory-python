"""Construct distributions on arbitrary supports.

Mirrors ``src/support.jl`` in DistributionsFactories.jl. Given a target support,
chooses between:

- **Affine transform** when the requested support has the same shape as the
  distribution's natural support (e.g. Gamma on ``[a, ∞)``, Beta on ``[a, b]``).
- **Truncation** when the requested support is strictly contained in the
  natural one (e.g. Normal on ``[a, b]``).

Continuous supports are passed as a 2-tuple ``(lo, hi)``; either endpoint may
be ``math.inf`` or ``-math.inf``. Discrete supports are passed as a ``range``
(``range(a, b+1)`` for the inclusive integer interval ``{a, …, b}``).
"""

import math
from scipy import stats

from ._registry import SUPPORT_TYPE, resolve_dist
from ._distributions import DIST_HANDLERS


def _support_endpoints(support):
    """Return ``(lo, hi, is_discrete)`` from a tuple or range support spec."""
    if isinstance(support, range):
        return float(support.start), float(support.stop - 1), True
    if isinstance(support, tuple) and len(support) == 2:
        return float(support[0]), float(support[1]), False
    raise ValueError(
        f"Unsupported `support` type: {type(support).__name__}. "
        f"Use a 2-tuple (a, b) or a range(a, b+1)."
    )


def _requested_support_shape(lo, hi):
    if math.isinf(lo) and lo < 0 and math.isinf(hi) and hi > 0:
        return "real"
    if math.isfinite(lo) and math.isinf(hi) and hi > 0:
        return "half_right"
    if math.isinf(lo) and lo < 0 and math.isfinite(hi):
        return "half_left"
    if math.isfinite(lo) and math.isfinite(hi):
        return "bounded"
    raise ValueError(f"Invalid support endpoints: ({lo}, {hi})")


# ---------------------------------------------------------------------------
# Affine wrappers
# ---------------------------------------------------------------------------

class _FlippedDist:
    """Frozen-like wrapper for ``b - X`` where ``X`` is a frozen scipy dist.

    Implements the methods we expect from a frozen scipy distribution
    (``mean``, ``var``, ``std``, ``pdf``, ``cdf``, ``ppf``, ``rvs``, ``support``).
    Used for placing positive-support distributions on ``(-∞, b]``.
    """

    def __init__(self, b, inner):
        self.b = float(b)
        self._inner = inner

    def __repr__(self):
        return f"FlippedDist(b={self.b}, inner={self._inner})"

    def mean(self):
        return self.b - self._inner.mean()

    def var(self):
        return self._inner.var()

    def std(self):
        return self._inner.std()

    def pdf(self, x):
        return self._inner.pdf(self.b - x)

    def cdf(self, x):
        return 1.0 - self._inner.cdf(self.b - x)

    def sf(self, x):
        return self._inner.cdf(self.b - x)

    def ppf(self, q):
        return self.b - self._inner.ppf(1.0 - q)

    def rvs(self, size=None, random_state=None):
        return self.b - self._inner.rvs(size=size, random_state=random_state)

    def support(self):
        lo, hi = self._inner.support()
        return (self.b - hi, self.b - lo)


def _affine_shift(name, mu, var, a):
    """Standardize to natural support, build, then shift by ``+a``."""
    mu_std = mu - a
    return _build_standard(name, mu_std, var, loc=a)


def _affine_flip(name, mu, var, b):
    """Standardize, build, then flip about ``b`` (yielding support ``(-∞, b]``)."""
    mu_std = b - mu
    inner = _build_standard(name, mu_std, var, loc=0.0)
    return _FlippedDist(b=b, inner=inner)


def _affine_scale(name, mu, var, a, b):
    """Standardize unit-support distribution to ``[0, 1]``, then scale to ``[a, b]``."""
    w = b - a
    mu_std = (mu - a) / w
    var_std = var / w ** 2
    handler = DIST_HANDLERS[name]
    # Build on unit support, then re-parameterize via loc/scale.
    # For Beta and Uniform, scipy frozen accepts loc/scale.
    base = handler.from_mean_var(mu_std, var_std)
    kwds = dict(base.kwds)
    kwds["loc"] = a
    kwds["scale"] = w
    return base.dist(*base.args, **kwds)


def _build_standard(name, mu_std, var_std, loc=0.0):
    """Build a frozen distribution on its natural support, then apply scipy loc.

    For positive-support distributions, scipy accepts ``loc`` to shift the
    support. We build at ``loc=0`` and then re-freeze with the desired loc.
    """
    handler = DIST_HANDLERS[name]
    base = handler.from_mean_var(mu_std, var_std)
    if loc == 0.0:
        return base
    kwds = dict(base.kwds)
    kwds["loc"] = kwds.get("loc", 0.0) + loc
    return base.dist(*base.args, **kwds)


# ---------------------------------------------------------------------------
# Discrete affine shift
# ---------------------------------------------------------------------------

def _discrete_affine_shift(name, mu, var, a):
    """Build a discrete distribution shifted to start at ``a``.

    Only meaningful for ``Binomial`` / ``DiscreteUniform`` (bounded discrete).
    """
    handler = DIST_HANDLERS[name]
    base = handler.from_mean_var(mu - a, var)
    kwds = dict(base.kwds)
    kwds["loc"] = kwds.get("loc", 0.0) + a
    return base.dist(*base.args, **kwds)


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

class _TruncatedDist:
    """Frozen-like wrapper representing the truncation of ``inner`` to ``[lo, hi]``.

    Useful when scipy lacks a dedicated truncated form for the family. The
    underlying ``inner`` distribution is used as-is for the un-truncated pdf;
    we renormalize by ``inner.cdf(hi) - inner.cdf(lo)``.
    """

    def __init__(self, inner, lo, hi):
        self._inner = inner
        self.lo = float(lo)
        self.hi = float(hi)
        self._discrete = hasattr(inner, "pmf")
        if self._discrete:
            lo_i = int(math.ceil(self.lo))
            hi_i = int(math.floor(self.hi))
            cdf_at_lo_minus_1 = inner.cdf(lo_i - 1) if lo_i > 0 else 0.0
            self._Z = inner.cdf(hi_i) - cdf_at_lo_minus_1
        else:
            self._Z = inner.cdf(hi) - inner.cdf(lo)
        if not (self._Z > 0):
            raise ValueError(
                f"Truncated distribution: truncation mass is zero on [{lo}, {hi}]"
            )

    def __repr__(self):
        return f"TruncatedDist(inner={self._inner}, lo={self.lo}, hi={self.hi})"

    def pdf(self, x):
        x = _scalar_or_array(x)
        in_support = (x >= self.lo) & (x <= self.hi)
        if self._discrete:
            out = self._inner.pmf(x) / self._Z
        else:
            out = self._inner.pdf(x) / self._Z
        return _where_zero(in_support, out)

    def pmf(self, x):
        if not self._discrete:
            raise AttributeError("pmf not defined for continuous distribution")
        return self.pdf(x)

    def cdf(self, x):
        x = _scalar_or_array(x)
        clipped = _clip(x, self.lo, self.hi)
        return (self._inner.cdf(clipped) - self._inner.cdf(self.lo)) / self._Z

    def sf(self, x):
        return 1.0 - self.cdf(x)

    def ppf(self, q):
        import numpy as np
        q = np.asarray(q)
        F_lo = self._inner.cdf(self.lo)
        return self._inner.ppf(F_lo + q * self._Z)

    def rvs(self, size=None, random_state=None):
        # Inverse-CDF sampling from a uniform on [F(lo), F(hi)].
        import numpy as np
        U = stats.uniform.rvs(size=size, random_state=random_state)
        F_lo = self._inner.cdf(self.lo)
        return self._inner.ppf(F_lo + U * self._Z)

    def mean(self):
        if self._discrete:
            lo_i = int(math.ceil(self.lo))
            hi_i = int(math.floor(self.hi))
            m = 0.0
            for k in range(lo_i, hi_i + 1):
                m += k * self._inner.pmf(k) / self._Z
            return m
        from scipy.integrate import quad
        m, _ = quad(lambda x: x * self._inner.pdf(x), self.lo, self.hi)
        return m / self._Z

    def var(self):
        if self._discrete:
            lo_i = int(math.ceil(self.lo))
            hi_i = int(math.floor(self.hi))
            m, m2 = 0.0, 0.0
            for k in range(lo_i, hi_i + 1):
                pk = self._inner.pmf(k) / self._Z
                m += k * pk
                m2 += k * k * pk
            return m2 - m * m
        from scipy.integrate import quad
        m, _ = quad(lambda x: x * self._inner.pdf(x), self.lo, self.hi)
        m2, _ = quad(lambda x: x ** 2 * self._inner.pdf(x), self.lo, self.hi)
        return m2 / self._Z - (m / self._Z) ** 2

    def std(self):
        return math.sqrt(self.var())

    def support(self):
        return (self.lo, self.hi)


def _scalar_or_array(x):
    import numpy as np
    return np.asarray(x)


def _clip(x, lo, hi):
    import numpy as np
    return np.minimum(np.maximum(x, lo), hi)


def _where_zero(mask, vals):
    import numpy as np
    return np.where(mask, vals, 0.0)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def dist_on_support(dist, mu, var, support):
    """Construct ``dist`` with the given moments on the given support.

    Decides between affine transform and truncation by comparing the
    requested support to the distribution's natural support.
    """
    name, _ = resolve_dist(dist)
    natural = SUPPORT_TYPE.get(name)
    if natural is None:
        raise ValueError(f"{name!r}: no natural support classification known")

    lo, hi, is_discrete = _support_endpoints(support)

    if is_discrete:
        return _dispatch_discrete(name, mu, var, lo, hi)

    requested = _requested_support_shape(lo, hi)

    if natural == "real":
        if requested == "real":
            return DIST_HANDLERS[name].from_mean_var(mu, var)
        # truncation
        return _truncate_real(name, mu, var, lo, hi)

    if natural == "positive":
        if requested == "half_right":
            return _affine_shift(name, mu, var, lo)
        if requested == "half_left":
            return _affine_flip(name, mu, var, hi)
        if requested == "bounded":
            if lo < 0:
                raise ValueError(
                    f"Cannot place {name!r} (natural [0, ∞)) on [{lo}, {hi}] with lo < 0"
                )
            return _truncate_positive(name, mu, var, lo, hi)
        raise ValueError(
            f"Cannot place {name!r} (natural [0, ∞)) on (-∞, ∞)"
        )

    if natural == "unit":
        if requested == "bounded":
            return _affine_scale(name, mu, var, lo, hi)
        raise ValueError(
            f"Cannot place {name!r} (natural [0, 1]) on an unbounded interval"
        )

    raise ValueError(f"Unsupported combination: {name!r} on ({lo}, {hi})")


def _dispatch_discrete(name, mu, var, lo, hi):
    natural = SUPPORT_TYPE[name]
    if natural == "integer_bounded":
        return _discrete_affine_shift(name, mu, var, lo)
    if natural == "integer_nonneg":
        if name == "poisson":
            from ._distributions import truncated_poisson
            td = truncated_poisson(lo, hi, mu)
            # Variance is determined by mean on a bounded support — if the
            # caller passed a var that disagrees, surface it (Julia does the
            # same; rtol=1e-3 there).
            achieved_var = td.var()
            if not math.isclose(achieved_var, var, rel_tol=1e-3, abs_tol=1e-9):
                raise ValueError(
                    f"Truncated Poisson on [{int(lo)}, {int(hi)}]: variance is "
                    f"determined by the mean; achieved var={achieved_var:.6g}, "
                    f"requested var={var:.6g}"
                )
            return td
        raise NotImplementedError(
            f"{name!r} on a bounded discrete range is not yet supported"
        )
    raise ValueError(f"{name!r} does not have discrete support")


def _truncate_real(name, mu, var, lo, hi):
    """Truncate a real-line distribution to ``[lo, hi]`` and match moments."""
    from ._truncation_solvers import solve_truncated_locscale, solve_truncated_generic
    if name in ("normal", "laplace", "logistic"):
        return solve_truncated_locscale(name, lo, hi, mu, var)
    return solve_truncated_generic(name, lo, hi, mu, var)


def _truncate_positive(name, mu, var, lo, hi):
    """Truncate a positive-support distribution to ``[lo, hi]`` and match moments."""
    from ._truncation_solvers import solve_truncated_generic
    return solve_truncated_generic(name, lo, hi, mu, var)

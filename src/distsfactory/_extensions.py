"""Distribution types that scipy.stats does not provide natively.

Mirrors the Julia ``src/extensions/`` modules:

- ``DiscreteSymmetricTriangular(mu, n)`` — integer-valued symmetric triangular
  on ``{mu-n, ..., mu+n}``.
- ``DiscreteTriangular(a, b, c)`` — integer-valued asymmetric triangular on
  ``{a, ..., b}`` with mode at ``c``.

Both expose a frozen-scipy-like surface (``pmf``, ``cdf``, ``ppf``, ``mean``,
``var``, ``std``, ``rvs``, ``support``) so they work with the rest of the
package.
"""

import math
import numpy as np


class DiscreteSymmetricTriangularDist:
    """Integer symmetric triangular on ``{mu-n, ..., mu+n}``.

    PMF: ``P(mu+k) = (n + 1 - |k|) / (n + 1)**2`` for ``k in {-n, ..., n}``.
    Mean ``= mu``; var ``= n*(n+2)/6``.
    """

    def __init__(self, mu, n):
        if n < 0 or not float(n).is_integer():
            raise ValueError("DiscreteSymmetricTriangular: n must be a non-negative integer")
        if not float(mu).is_integer():
            raise ValueError("DiscreteSymmetricTriangular: mu must be an integer")
        self.mu = int(mu)
        self.n = int(n)

    def __repr__(self):
        return f"DiscreteSymmetricTriangularDist(mu={self.mu}, n={self.n})"

    def support(self):
        return (self.mu - self.n, self.mu + self.n)

    def mean(self):
        return float(self.mu)

    def var(self):
        return self.n * (self.n + 2) / 6

    def std(self):
        return math.sqrt(self.var())

    def mode(self):
        return self.mu

    def pmf(self, x):
        scalar = np.isscalar(x)
        x_arr = np.atleast_1d(np.asarray(x))
        out = np.zeros_like(x_arr, dtype=float)
        for i, xi in np.ndenumerate(x_arr):
            if float(xi).is_integer() and self.mu - self.n <= xi <= self.mu + self.n:
                k = int(xi) - self.mu
                out[i] = (self.n + 1 - abs(k)) / (self.n + 1) ** 2
        return float(out[0]) if scalar else out

    def pdf(self, x):
        return self.pmf(x)

    def cdf(self, x):
        scalar = np.isscalar(x)
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.zeros_like(x_arr, dtype=float)
        for i, xi in np.ndenumerate(x_arr):
            if xi < self.mu - self.n:
                out[i] = 0.0
            elif xi >= self.mu + self.n:
                out[i] = 1.0
            else:
                k_max = math.floor(xi) - self.mu
                s = 0.0
                for k in range(-self.n, k_max + 1):
                    s += (self.n + 1 - abs(k)) / (self.n + 1) ** 2
                out[i] = s
        return float(out[0]) if scalar else out

    def ppf(self, p):
        scalar = np.isscalar(p)
        p_arr = np.atleast_1d(np.asarray(p, dtype=float))
        out = np.zeros_like(p_arr, dtype=float)
        for i, pi in np.ndenumerate(p_arr):
            if pi <= 0:
                out[i] = self.mu - self.n
            elif pi >= 1:
                out[i] = self.mu + self.n
            else:
                s = 0.0
                result = self.mu + self.n
                for k in range(-self.n, self.n + 1):
                    s += (self.n + 1 - abs(k)) / (self.n + 1) ** 2
                    if s >= pi:
                        result = self.mu + k
                        break
                out[i] = result
        return float(out[0]) if scalar else out

    def rvs(self, size=None, random_state=None):
        rng = np.random.default_rng(random_state)
        u = rng.uniform(size=size)
        return self.ppf(u)


class DiscreteTriangularDist:
    """Integer asymmetric triangular on ``{a, ..., b}`` with mode ``c``.

    PMF: two linear ramps meeting at ``c``. Normalizer ``Z = (b - a + 2) / 2``.
    """

    def __init__(self, a, b, c):
        if not (a <= c <= b):
            raise ValueError(
                f"DiscreteTriangular: must satisfy a <= c <= b (got a={a}, b={b}, c={c})"
            )
        for name, val in (("a", a), ("b", b), ("c", c)):
            if not float(val).is_integer():
                raise ValueError(f"DiscreteTriangular: {name} must be an integer")
        self.a = int(a)
        self.b = int(b)
        self.c = int(c)
        self._Z = (self.b - self.a + 2) / 2

    def __repr__(self):
        return f"DiscreteTriangularDist(a={self.a}, b={self.b}, c={self.c})"

    def support(self):
        return (self.a, self.b)

    def mode(self):
        return self.c

    def _pmf_scalar(self, k):
        if not (self.a <= k <= self.b):
            return 0.0
        if k <= self.c:
            return (k - self.a + 1) / (self.c - self.a + 1) / self._Z
        return (self.b - k + 1) / (self.b - self.c + 1) / self._Z

    def pmf(self, x):
        scalar = np.isscalar(x)
        x_arr = np.atleast_1d(np.asarray(x))
        out = np.zeros_like(x_arr, dtype=float)
        for i, xi in np.ndenumerate(x_arr):
            if float(xi).is_integer():
                out[i] = self._pmf_scalar(int(xi))
        return float(out[0]) if scalar else out

    def pdf(self, x):
        return self.pmf(x)

    def cdf(self, x):
        scalar = np.isscalar(x)
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.zeros_like(x_arr, dtype=float)
        for i, xi in np.ndenumerate(x_arr):
            if xi < self.a:
                out[i] = 0.0
            elif xi >= self.b:
                out[i] = 1.0
            else:
                k_max = math.floor(xi)
                out[i] = sum(self._pmf_scalar(k) for k in range(self.a, k_max + 1))
        return float(out[0]) if scalar else out

    def ppf(self, p):
        scalar = np.isscalar(p)
        p_arr = np.atleast_1d(np.asarray(p, dtype=float))
        out = np.zeros_like(p_arr, dtype=float)
        for i, pi in np.ndenumerate(p_arr):
            if pi <= 0:
                out[i] = self.a
            elif pi >= 1:
                out[i] = self.b
            else:
                s = 0.0
                result = self.b
                for k in range(self.a, self.b + 1):
                    s += self._pmf_scalar(k)
                    if s >= pi:
                        result = k
                        break
                out[i] = result
        return float(out[0]) if scalar else out

    def mean(self):
        return sum(k * self._pmf_scalar(k) for k in range(self.a, self.b + 1))

    def var(self):
        m = self.mean()
        return sum((k - m) ** 2 * self._pmf_scalar(k) for k in range(self.a, self.b + 1))

    def std(self):
        return math.sqrt(self.var())

    def rvs(self, size=None, random_state=None):
        rng = np.random.default_rng(random_state)
        u = rng.uniform(size=size)
        return self.ppf(u)

"""Distribution-specific moment/quantile/mode handlers.

One handler class per distribution family. Each handler exposes:

- ``from_*`` constructors returning a frozen ``scipy.stats`` distribution.
- ``DISPATCH`` mapping each supported spec dataclass to a builder lambda.
- (Where applicable) per-family quantile helpers.

The feasibility predicates live in ``_feasibility.py`` and are called from the
public ``dist_exists`` / ``make_dist`` paths; some constructors call
``require_mean_var`` to raise early with the same reason a feasibility check
would have surfaced.
"""

import math
import numpy as np
from scipy import stats
from scipy.special import gamma as _gammafn, gammaln

from ._solvers import find_root_1d, newton_2d
from ._feasibility import require_mean_var
from ._extensions import DiscreteSymmetricTriangularDist, DiscreteTriangularDist
from ._specs import (
    MeanVarSpec, MeanSpec, VarSpec, QuantileSpec, TwoQuantileSpec,
    MeanQuantileSpec, MeanModeSpec, ModeVarSpec, ModeQuantileSpec, ModeIQRSpec,
    ModeSpec, MeanVarModeSpec,
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _two_quantiles_location_scale(scipy_family, p1, q1, p2, q2):
    """Solve for (loc, scale) of a location-scale family from two quantiles.

    Uses ``scipy_family.ppf`` on the standard parameterization.
    """
    if not (0 < p1 < 1 and 0 < p2 < 1):
        raise ValueError("p values must be in (0, 1)")
    if p1 == p2:
        raise ValueError("p1 and p2 must be distinct")
    z1 = scipy_family.ppf(p1)
    z2 = scipy_family.ppf(p2)
    scale = (q2 - q1) / (z2 - z1)
    if scale <= 0:
        raise ValueError("Quantile specification implies non-positive scale")
    loc = q1 - scale * z1
    return loc, scale


# ===========================================================================
# Real-line continuous
# ===========================================================================

# ---- Normal ---------------------------------------------------------------
class NormalDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("normal", mu, var)
        return stats.norm(loc=mu, scale=math.sqrt(var))

    @staticmethod
    def from_two_quantiles(p1, q1, p2, q2):
        loc, scale = _two_quantiles_location_scale(stats.norm, p1, q1, p2, q2)
        return stats.norm(loc=loc, scale=scale)

    @staticmethod
    def from_mode_var(mode, var):
        return stats.norm(loc=mode, scale=math.sqrt(var))

    @staticmethod
    def from_mode_iqr(mode, iqr):
        sigma = iqr / (2 * stats.norm.ppf(0.75))
        return stats.norm(loc=mode, scale=sigma)

    DISPATCH = {
        MeanVarSpec:      lambda s: NormalDist.from_mean_var(s.mean, s.var),
        TwoQuantileSpec:  lambda s: NormalDist.from_two_quantiles(s.p1, s.q1, s.p2, s.q2),
        ModeVarSpec:      lambda s: NormalDist.from_mode_var(s.mode, s.var),
        ModeIQRSpec:      lambda s: NormalDist.from_mode_iqr(s.mode, s.iqr),
    }


# ---- Laplace --------------------------------------------------------------
class LaplaceDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("laplace", mu, var)
        # var = 2*b^2
        b = math.sqrt(var / 2)
        return stats.laplace(loc=mu, scale=b)

    @staticmethod
    def from_two_quantiles(p1, q1, p2, q2):
        loc, scale = _two_quantiles_location_scale(stats.laplace, p1, q1, p2, q2)
        return stats.laplace(loc=loc, scale=scale)

    @staticmethod
    def from_mode_iqr(mode, iqr):
        # Laplace: IQR = 2*b*ln(2)
        b = iqr / (2 * math.log(2))
        return stats.laplace(loc=mode, scale=b)

    DISPATCH = {
        MeanVarSpec:      lambda s: LaplaceDist.from_mean_var(s.mean, s.var),
        TwoQuantileSpec:  lambda s: LaplaceDist.from_two_quantiles(s.p1, s.q1, s.p2, s.q2),
        ModeIQRSpec:      lambda s: LaplaceDist.from_mode_iqr(s.mode, s.iqr),
    }


# ---- Logistic -------------------------------------------------------------
class LogisticDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("logistic", mu, var)
        # var = scale^2 * pi^2 / 3
        s = math.sqrt(3 * var / math.pi ** 2)
        return stats.logistic(loc=mu, scale=s)

    @staticmethod
    def from_two_quantiles(p1, q1, p2, q2):
        # Logistic quantile is loc + scale * log(p/(1-p))
        z1 = math.log(p1 / (1 - p1))
        z2 = math.log(p2 / (1 - p2))
        s = (q2 - q1) / (z2 - z1)
        mu = q1 - s * z1
        return stats.logistic(loc=mu, scale=s)

    @staticmethod
    def from_mode_iqr(mode, iqr):
        s = iqr / (2 * math.log(3))
        return stats.logistic(loc=mode, scale=s)

    @staticmethod
    def from_mean_quantile(mu, p, q):
        z = math.log(p / (1 - p))
        if math.isclose(z, 0.0, abs_tol=1e-12):
            if math.isclose(mu, q, rel_tol=1e-6):
                raise ValueError(
                    "Logistic mean and median are always equal; "
                    "need an additional constraint to determine scale"
                )
            raise ValueError(
                f"Logistic mean must equal median, but got mean={mu}, median={q}"
            )
        s = (q - mu) / z
        if s <= 0:
            raise ValueError(
                f"Cannot construct Logistic with mean={mu} and "
                f"quantile({p})={q}: scale would be non-positive"
            )
        return stats.logistic(loc=mu, scale=s)

    DISPATCH = {
        MeanVarSpec:      lambda s: LogisticDist.from_mean_var(s.mean, s.var),
        TwoQuantileSpec:  lambda s: LogisticDist.from_two_quantiles(s.p1, s.q1, s.p2, s.q2),
        ModeIQRSpec:      lambda s: LogisticDist.from_mode_iqr(s.mode, s.iqr),
        MeanQuantileSpec: lambda s: LogisticDist.from_mean_quantile(s.mean, s.p, s.q),
    }


# ---- Gumbel ---------------------------------------------------------------
# scipy.gumbel_r: pdf = exp(-(x-loc)/scale - exp(-(x-loc)/scale)) / scale
# mean = loc + scale * gamma_euler, var = scale^2 * pi^2 / 6
class GumbelDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("gumbel", mu, var)
        beta = math.sqrt(6 * var / math.pi ** 2)
        loc = mu - beta * np.euler_gamma
        return stats.gumbel_r(loc=loc, scale=beta)

    @staticmethod
    def from_two_quantiles(p1, q1, p2, q2):
        loc, scale = _two_quantiles_location_scale(stats.gumbel_r, p1, q1, p2, q2)
        return stats.gumbel_r(loc=loc, scale=scale)

    DISPATCH = {
        MeanVarSpec:     lambda s: GumbelDist.from_mean_var(s.mean, s.var),
        TwoQuantileSpec: lambda s: GumbelDist.from_two_quantiles(s.p1, s.q1, s.p2, s.q2),
    }


# ---- Cauchy ---------------------------------------------------------------
class CauchyDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("cauchy", mu, var)
        raise AssertionError("unreachable: cauchy always rejects mean-var")

    @staticmethod
    def from_two_quantiles(p1, q1, p2, q2):
        loc, scale = _two_quantiles_location_scale(stats.cauchy, p1, q1, p2, q2)
        return stats.cauchy(loc=loc, scale=scale)

    DISPATCH = {
        MeanVarSpec:     lambda s: CauchyDist.from_mean_var(s.mean, s.var),
        TwoQuantileSpec: lambda s: CauchyDist.from_two_quantiles(s.p1, s.q1, s.p2, s.q2),
    }


# ---- TDist ----------------------------------------------------------------
# Standard t with df = nu. mean=0 if nu>1, var = nu/(nu-2) if nu>2.
class TDistDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("tdist", mu, var)
        # mu must be 0 and var > 1. Then nu = 2*var/(var-1).
        nu = 2 * var / (var - 1)
        return stats.t(df=nu)

    DISPATCH = {
        MeanVarSpec: lambda s: TDistDist.from_mean_var(s.mean, s.var),
    }


# ---- Uniform --------------------------------------------------------------
# scipy.uniform(loc, scale) on [loc, loc+scale]. var = scale^2/12.
class UniformDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("uniform", mu, var)
        half_width = math.sqrt(3 * var)
        a = mu - half_width
        scale = 2 * half_width
        return stats.uniform(loc=a, scale=scale)

    DISPATCH = {
        MeanVarSpec: lambda s: UniformDist.from_mean_var(s.mean, s.var),
    }


# ---- Symmetric Triangular --------------------------------------------------
# SymTriangularDist(mu, s) on Julia: support [mu-s, mu+s], var = s^2/6
# scipy.triang(c, loc, scale): support [loc, loc+scale], mode at loc+c*scale
class SymTriangularDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("sym_triangular", mu, var)
        s = math.sqrt(6 * var)
        return stats.triang(c=0.5, loc=mu - s, scale=2 * s)

    DISPATCH = {
        MeanVarSpec: lambda s: SymTriangularDist.from_mean_var(s.mean, s.var),
    }


# ---- (Asymmetric) Triangular ------------------------------------------------
# TriangularDist(a, b, c) on Julia: support [a, b], mode at c.
# mean = (a+b+c)/3, var = (a^2 + b^2 + c^2 - ab - ac - bc) / 18.
# scipy.triang(c, loc, scale): support [loc, loc+scale], mode at loc+c*scale.
class TriangularDist:
    @staticmethod
    def from_mean_var_mode(mu, var, c):
        if not (var > 0):
            raise ValueError("Triangular: var must be > 0")
        S = 3 * mu - c
        ab = (S ** 2 + c ** 2 - c * S - 18 * var) / 3
        disc = S ** 2 - 4 * ab
        if disc < 0:
            raise ValueError(
                f"Triangular: no real (a, b) for (mu={mu}, var={var}, mode={c}); "
                f"discriminant={disc}"
            )
        sqrt_disc = math.sqrt(disc)
        a = (S - sqrt_disc) / 2
        b = (S + sqrt_disc) / 2
        if not (a <= c <= b):
            raise ValueError(
                f"Triangular: solved (a={a}, b={b}) does not satisfy a <= c <= b"
            )
        scale = b - a
        c_scipy = (c - a) / scale
        return stats.triang(c=c_scipy, loc=a, scale=scale)

    DISPATCH = {
        MeanVarModeSpec: lambda s: TriangularDist.from_mean_var_mode(s.mean, s.var, s.mode),
    }


# ===========================================================================
# Positive continuous
# ===========================================================================

# ---- Gamma ----------------------------------------------------------------
# scipy.gamma(a, scale). mean=a*scale, var=a*scale^2.
class GammaDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("gamma", mu, var)
        alpha = mu ** 2 / var
        theta = var / mu
        return stats.gamma(a=alpha, scale=theta)

    @staticmethod
    def from_mean_mode(mu, mode):
        if mu <= mode:
            raise ValueError(
                f"Gamma mean ({mu}) must be greater than mode ({mode})"
            )
        theta = mu - mode
        alpha = mu / theta
        return stats.gamma(a=alpha, scale=theta)

    @staticmethod
    def from_mode_var(mode, var):
        if mode <= 0:
            raise ValueError(f"Gamma mode must be positive, got {mode}")
        def f(log_alpha):
            alpha = math.exp(log_alpha) + 1
            theta = math.sqrt(var / alpha)
            return (alpha - 1) * theta - mode
        log_alpha = find_root_1d(f, x0=0.0)
        alpha = math.exp(log_alpha) + 1
        theta = math.sqrt(var / alpha)
        return stats.gamma(a=alpha, scale=theta)

    @staticmethod
    def from_mode_quantile(mode, p, q):
        if mode <= 0:
            raise ValueError(f"Gamma mode must be positive, got {mode}")
        def f(log_alpha):
            alpha = math.exp(log_alpha) + 1
            theta = mode / (alpha - 1)
            return stats.gamma.ppf(p, a=alpha, scale=theta) - q
        log_alpha = find_root_1d(f, x0=0.0)
        alpha = math.exp(log_alpha) + 1
        theta = mode / (alpha - 1)
        return stats.gamma(a=alpha, scale=theta)

    @staticmethod
    def from_mode_iqr(mode, iqr):
        if mode <= 0:
            raise ValueError(f"Gamma mode must be positive, got {mode}")
        def f(log_alpha):
            alpha = math.exp(log_alpha) + 1
            theta = mode / (alpha - 1)
            d = stats.gamma(a=alpha, scale=theta)
            return d.ppf(0.75) - d.ppf(0.25) - iqr
        log_alpha = find_root_1d(f, x0=0.0)
        alpha = math.exp(log_alpha) + 1
        theta = mode / (alpha - 1)
        return stats.gamma(a=alpha, scale=theta)

    @staticmethod
    def from_two_quantiles(p1, q1, p2, q2):
        if not (q1 > 0 and q2 > 0):
            raise ValueError("Gamma: quantiles must be > 0")
        r = q2 / q1
        def f(log_alpha):
            alpha = math.exp(log_alpha)
            z1 = stats.gamma.ppf(p1, a=alpha, scale=1.0)
            z2 = stats.gamma.ppf(p2, a=alpha, scale=1.0)
            return z2 / z1 - r
        log_alpha = find_root_1d(f, x0=0.0)
        alpha = math.exp(log_alpha)
        theta = q1 / stats.gamma.ppf(p1, a=alpha, scale=1.0)
        return stats.gamma(a=alpha, scale=theta)

    @staticmethod
    def from_mean_quantile(mu, p, q):
        if mu <= 0:
            raise ValueError("Gamma: mean must be > 0")
        def f(log_alpha):
            alpha = math.exp(log_alpha)
            theta = mu / alpha
            return stats.gamma.ppf(p, a=alpha, scale=theta) - q
        log_alpha = find_root_1d(f, x0=0.0)
        alpha = math.exp(log_alpha)
        theta = mu / alpha
        return stats.gamma(a=alpha, scale=theta)

    DISPATCH = {
        MeanVarSpec:      lambda s: GammaDist.from_mean_var(s.mean, s.var),
        MeanModeSpec:     lambda s: GammaDist.from_mean_mode(s.mean, s.mode),
        ModeVarSpec:      lambda s: GammaDist.from_mode_var(s.mode, s.var),
        ModeQuantileSpec: lambda s: GammaDist.from_mode_quantile(s.mode, s.p, s.q),
        ModeIQRSpec:      lambda s: GammaDist.from_mode_iqr(s.mode, s.iqr),
        TwoQuantileSpec:  lambda s: GammaDist.from_two_quantiles(s.p1, s.q1, s.p2, s.q2),
        MeanQuantileSpec: lambda s: GammaDist.from_mean_quantile(s.mean, s.p, s.q),
    }


# ---- Erlang ---------------------------------------------------------------
# Erlang is Gamma with integer shape. Round to nearest integer.
class ErlangDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("erlang", mu, var)
        k = round(mu ** 2 / var)
        if k < 1:
            k = 1
        theta = var / mu
        return stats.gamma(a=k, scale=theta)

    DISPATCH = {
        MeanVarSpec: lambda s: ErlangDist.from_mean_var(s.mean, s.var),
    }


# ---- Exponential ----------------------------------------------------------
class ExponentialDist:
    @staticmethod
    def from_mean(mu):
        if mu <= 0:
            raise ValueError(f"Exponential mean must be positive, got {mu}")
        return stats.expon(scale=mu)

    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("exponential", mu, var)
        return stats.expon(scale=mu)

    @staticmethod
    def from_var(var):
        if var <= 0:
            raise ValueError("Exponential: var must be > 0")
        return stats.expon(scale=math.sqrt(var))

    @staticmethod
    def from_quantile(p, q):
        if not (0 < p < 1):
            raise ValueError("p must be in (0,1)")
        if not (q > 0):
            raise ValueError("Exponential: quantile must be > 0")
        theta = -q / math.log(1 - p)
        return stats.expon(scale=theta)

    DISPATCH = {
        MeanVarSpec:  lambda s: ExponentialDist.from_mean_var(s.mean, s.var),
        MeanSpec:     lambda s: ExponentialDist.from_mean(s.mean),
        VarSpec:      lambda s: ExponentialDist.from_var(s.var),
        QuantileSpec: lambda s: ExponentialDist.from_quantile(s.p, s.q),
    }


# ---- LogNormal ------------------------------------------------------------
# scipy.lognorm(s=sigma_log, scale=exp(mu_log)).
# mean = exp(mu + sigma^2/2), var = (exp(sigma^2)-1) * exp(2*mu+sigma^2)
class LogNormalDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("lognormal", mu, var)
        sigma_log = math.sqrt(math.log(var / mu ** 2 + 1))
        mu_log = math.log(mu ** 2 / math.sqrt(var + mu ** 2))
        return stats.lognorm(s=sigma_log, scale=math.exp(mu_log))

    @staticmethod
    def from_two_quantiles(p1, q1, p2, q2):
        if not (q1 > 0 and q2 > 0):
            raise ValueError("LogNormal: quantiles must be > 0")
        # log(X) is Normal — reduce to Normal location-scale on log-quantiles.
        loc, scale = _two_quantiles_location_scale(
            stats.norm, p1, math.log(q1), p2, math.log(q2)
        )
        return stats.lognorm(s=scale, scale=math.exp(loc))

    @staticmethod
    def from_mean_quantile(mu, p, q):
        if mu <= 0:
            raise ValueError("LogNormal: mu must be > 0")
        if q <= 0:
            raise ValueError("LogNormal: quantile must be > 0")
        z_p = stats.norm.ppf(p)
        # log(mu) - sigma^2/2 + sigma * z_p = log(q)
        def f(sigma):
            return math.log(mu) - sigma ** 2 / 2 + sigma * z_p - math.log(q)
        sigma_sol = find_root_1d(f, x0=0.5)
        mu_log = math.log(mu) - sigma_sol ** 2 / 2
        return stats.lognorm(s=sigma_sol, scale=math.exp(mu_log))

    DISPATCH = {
        MeanVarSpec:      lambda s: LogNormalDist.from_mean_var(s.mean, s.var),
        TwoQuantileSpec:  lambda s: LogNormalDist.from_two_quantiles(s.p1, s.q1, s.p2, s.q2),
        MeanQuantileSpec: lambda s: LogNormalDist.from_mean_quantile(s.mean, s.p, s.q),
    }


# ---- Weibull --------------------------------------------------------------
# scipy.weibull_min(c=k, scale=lambda). Mean = lambda*Gamma(1+1/k).
def _solve_evt_shape(mu, var, positive=True):
    """Solve x/Beta(1/x, 1/x) = (1 + CV^2)/2 for x.

    Mirrors the Julia `_solve_evt_shape`. Positive root gives Weibull's k;
    negative root (negated) gives Frechet's alpha.
    """
    cv = math.sqrt(var) / mu
    target = (1 + cv ** 2) / 2

    def f(x):
        # x/Beta(1/x, 1/x) = x * Gamma(2/x) / Gamma(1/x)^2
        # scipy's gamma handles negative non-integer arguments (signed), which
        # we need for Frechet's negative-shape branch.
        if x == 0:
            return -target
        return x * _gammafn(2.0 / x) / _gammafn(1.0 / x) ** 2 - target

    cv2 = cv ** 2
    if 0 < cv2 < 1:
        if positive:
            lo, hi = 1.0 / cv, (cv2 + 1) / (2 * cv2)
        else:
            lo, hi = min(-math.sqrt(2 * math.pi), -1.0 / cv), -2 * (1 + cv2) / cv2
        if f(lo) * f(hi) > 0:
            raise RuntimeError(
                f"_solve_evt_shape: no sign change in bracket [{lo}, {hi}] "
                f"for CV^2={cv2}, positive={positive}"
            )
        return find_root_1d(f, bracket=sorted([lo, hi]))
    if math.isclose(cv2, 1.0, abs_tol=1e-12):
        return 1.0 if positive else find_root_1d(f, x0=-math.sqrt(7))
    # cv2 > 1
    if positive:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = -2.0, -2 * (1 + cv2) / cv2
    # Bisect down until sign change
    for _ in range(200):
        tmp = (lo + hi) / 2
        ftmp = f(tmp)
        if ftmp == 0:
            return tmp
        if ftmp < 0:
            hi = tmp
        else:
            lo = tmp
            break
    if f(lo) * f(hi) > 0:
        raise RuntimeError(
            f"_solve_evt_shape: bisection failed for CV^2={cv2}, positive={positive}"
        )
    return find_root_1d(f, bracket=sorted([lo, hi]))


def _gamma_func(x):
    return math.exp(gammaln(x))


class WeibullDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("weibull", mu, var)
        k = _solve_evt_shape(mu, var, positive=True)
        lam = mu / _gamma_func(1 + 1 / k)
        return stats.weibull_min(c=k, scale=lam)

    @staticmethod
    def from_two_quantiles(p1, q1, p2, q2):
        if not (q1 > 0 and q2 > 0):
            raise ValueError("Weibull: quantiles must be > 0")
        # q = lam*(-log(1-p))^(1/k). Eliminate lam: ratio.
        r = q2 / q1
        L = math.log(1 - p2) / math.log(1 - p1)
        if not (L > 0 and L != 1):
            raise ValueError("Weibull: degenerate p values")
        k = math.log(L) / math.log(r)
        if k <= 0:
            raise ValueError("Weibull: quantile spec implies non-positive shape")
        lam = q1 / (-math.log(1 - p1)) ** (1 / k)
        return stats.weibull_min(c=k, scale=lam)

    DISPATCH = {
        MeanVarSpec:     lambda s: WeibullDist.from_mean_var(s.mean, s.var),
        TwoQuantileSpec: lambda s: WeibullDist.from_two_quantiles(s.p1, s.q1, s.p2, s.q2),
    }


# ---- Frechet --------------------------------------------------------------
# scipy.invweibull(c=alpha, scale=s). Mean = s*Gamma(1 - 1/alpha) for alpha>1.
class FrechetDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("frechet", mu, var)
        alpha = -1 * _solve_evt_shape(mu, var, positive=False)
        s = mu / _gamma_func(1 - 1 / alpha)
        return stats.invweibull(c=alpha, scale=s)

    DISPATCH = {
        MeanVarSpec: lambda s: FrechetDist.from_mean_var(s.mean, s.var),
    }


# ---- Chisq ----------------------------------------------------------------
class ChisqDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("chisq", mu, var)
        return stats.chi2(df=mu)

    @staticmethod
    def from_mean(mu):
        if mu <= 0 or not float(mu).is_integer():
            raise ValueError("Chisq: mean must be a positive integer")
        return stats.chi2(df=mu)

    @staticmethod
    def from_var(var):
        # var = 2*k -> k = var/2
        k = var / 2
        if k <= 0 or not float(k).is_integer():
            raise ValueError("Chisq: var/2 must be a positive integer")
        return stats.chi2(df=k)

    DISPATCH = {
        MeanVarSpec: lambda s: ChisqDist.from_mean_var(s.mean, s.var),
        MeanSpec:    lambda s: ChisqDist.from_mean(s.mean),
        VarSpec:     lambda s: ChisqDist.from_var(s.var),
    }


# ---- Chi ------------------------------------------------------------------
class ChiDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("chi", mu, var)
        # Julia: nu = mu^2 + var (only valid if the moment identity holds)
        return stats.chi(df=mu ** 2 + var)

    DISPATCH = {
        MeanVarSpec: lambda s: ChiDist.from_mean_var(s.mean, s.var),
    }


# ---- Rayleigh -------------------------------------------------------------
class RayleighDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("rayleigh", mu, var)
        sigma = math.sqrt(2 / math.pi) * mu
        return stats.rayleigh(scale=sigma)

    @staticmethod
    def from_mean(mu):
        if mu <= 0:
            raise ValueError("Rayleigh: mean must be > 0")
        sigma = mu / math.sqrt(math.pi / 2)
        return stats.rayleigh(scale=sigma)

    @staticmethod
    def from_var(var):
        if var <= 0:
            raise ValueError("Rayleigh: var must be > 0")
        # var = sigma^2 * (4-pi)/2 -> sigma = sqrt(2*var/(4-pi))
        sigma = math.sqrt(2 * var / (4 - math.pi))
        return stats.rayleigh(scale=sigma)

    @staticmethod
    def from_mode(mode):
        if mode <= 0:
            raise ValueError("Rayleigh: mode must be > 0")
        return stats.rayleigh(scale=mode)  # mode == sigma

    @staticmethod
    def from_quantile(p, q):
        if not (0 < p < 1):
            raise ValueError("p must be in (0,1)")
        if q <= 0:
            raise ValueError("Rayleigh: quantile must be > 0")
        sigma = q / math.sqrt(-2 * math.log(1 - p))
        return stats.rayleigh(scale=sigma)

    DISPATCH = {
        MeanVarSpec:  lambda s: RayleighDist.from_mean_var(s.mean, s.var),
        MeanSpec:     lambda s: RayleighDist.from_mean(s.mean),
        VarSpec:      lambda s: RayleighDist.from_var(s.var),
        ModeSpec:     lambda s: RayleighDist.from_mode(s.mode),
        QuantileSpec: lambda s: RayleighDist.from_quantile(s.p, s.q),
    }


# ---- FDist ----------------------------------------------------------------
class FDistDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("fdist", mu, var)
        v2 = 2 * mu / (mu - 1)
        v1 = 2 * mu ** 2 * (v2 - 2) / (var * (v2 - 4) - 2 * mu ** 2)
        return stats.f(dfn=v1, dfd=v2)

    DISPATCH = {
        MeanVarSpec: lambda s: FDistDist.from_mean_var(s.mean, s.var),
    }


# ---- Inverse Gamma --------------------------------------------------------
# scipy.invgamma(a, scale=beta). Mean = beta/(a-1).
class InverseGammaDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("inverse_gamma", mu, var)
        alpha = (mu ** 2 + 2 * var) / var
        beta = mu * (alpha - 1)
        return stats.invgamma(a=alpha, scale=beta)

    DISPATCH = {
        MeanVarSpec: lambda s: InverseGammaDist.from_mean_var(s.mean, s.var),
    }


# ---- Pareto ---------------------------------------------------------------
# scipy.pareto(b=alpha, scale=theta). pdf(x) = alpha * theta^alpha / x^(alpha+1) for x>=theta.
# Julia's Pareto(alpha, theta) has the same parameterization (support [theta, inf)).
# mean = alpha*theta/(alpha-1) for alpha>1.
class ParetoDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("pareto", mu, var)
        cv2 = var / mu ** 2
        alpha = 1 + math.sqrt(1 + 1 / cv2)
        theta = mu * (alpha - 1) / alpha
        return stats.pareto(b=alpha, scale=theta)

    @staticmethod
    def from_two_quantiles(p1, q1, p2, q2):
        if not (q1 > 0 and q2 > 0):
            raise ValueError("Pareto: quantiles must be > 0")
        r = q2 / q1
        L = (1 - p2) / (1 - p1)
        if L == 1:
            raise ValueError("Pareto: degenerate p values")
        alpha = -math.log(L) / math.log(r)
        if alpha <= 0:
            raise ValueError("Pareto: quantile spec implies non-positive shape")
        theta = q1 * (1 - p1) ** (1 / alpha)
        return stats.pareto(b=alpha, scale=theta)

    DISPATCH = {
        MeanVarSpec:     lambda s: ParetoDist.from_mean_var(s.mean, s.var),
        TwoQuantileSpec: lambda s: ParetoDist.from_two_quantiles(s.p1, s.q1, s.p2, s.q2),
    }


# ---- Folded Normal --------------------------------------------------------
# scipy.foldnorm(c=mu/sigma, scale=sigma) — parent Normal(mu, sigma) folded at 0.
# Mean of folded: sigma*sqrt(2/pi)*exp(-mu^2/(2sigma^2)) + mu*erf(mu/(sigma*sqrt(2)))
# Var = mu^2 + sigma^2 - mean^2
def _folded_normal_moments(mu_p, sigma_p):
    m = sigma_p * math.sqrt(2 / math.pi) * math.exp(-mu_p ** 2 / (2 * sigma_p ** 2)) \
        + mu_p * math.erf(mu_p / (sigma_p * math.sqrt(2)))
    v = mu_p ** 2 + sigma_p ** 2 - m ** 2
    return m, v


def _solve_folded_normal(target_mu, target_var, maxiter=200, tol=1e-10):
    x = np.array([target_mu, math.log(math.sqrt(target_var))], dtype=float)

    def F(x):
        mu_p = x[0]
        sigma_p = math.exp(x[1])
        m, v = _folded_normal_moments(mu_p, sigma_p)
        return np.array([m - target_mu, v - target_var])

    h = 1e-7
    for _ in range(maxiter):
        Fx = F(x)
        if np.max(np.abs(Fx)) < tol:
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
            try:
                F_new = F(x_new)
                if np.all(np.isfinite(F_new)) and np.max(np.abs(F_new)) < np.max(np.abs(Fx)):
                    break
            except Exception:
                pass
            step *= 0.5
        x = x + step * dx
    return x[0], math.exp(x[1])


class FoldedNormalDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("folded_normal", mu, var)
        mu_p, sigma_p = _solve_folded_normal(float(mu), float(var))
        c = mu_p / sigma_p
        return stats.foldnorm(c=c, scale=sigma_p)

    DISPATCH = {
        MeanVarSpec: lambda s: FoldedNormalDist.from_mean_var(s.mean, s.var),
    }


# ===========================================================================
# Unit-interval continuous
# ===========================================================================

# ---- Beta -----------------------------------------------------------------
class BetaDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("beta", mu, var)
        S = mu * (1 - mu) / var - 1
        alpha = mu * S
        beta = (1 - mu) * S
        return stats.beta(a=alpha, b=beta)

    @staticmethod
    def from_mean_mode(mu, mode):
        if math.isclose(mu, mode, abs_tol=1e-12):
            raise ValueError(
                "Beta mean and mode cannot be equal (symmetric case is underdetermined)"
            )
        alpha = mu * (2 * mode - 1) / (mode - mu)
        beta = alpha * (1 - mu) / mu
        if alpha <= 1 or beta <= 1:
            raise ValueError(
                f"Beta from mean={mu}, mode={mode} gives alpha={alpha:.4f}, "
                f"beta={beta:.4f}; both must be > 1 for mode to exist"
            )
        return stats.beta(a=alpha, b=beta)

    @staticmethod
    def from_two_quantiles(p1, q1, p2, q2):
        # Initial guess via normal approximation
        z1 = stats.norm.ppf(p1)
        z2 = stats.norm.ppf(p2)
        sigma_est = max((q2 - q1) / (z2 - z1), 1e-4)
        mu_est = max(0.01, min(0.99, q1 - sigma_est * z1))
        v_est = min(sigma_est ** 2, mu_est * (1 - mu_est) * 0.99)
        S0 = mu_est * (1 - mu_est) / v_est - 1
        alpha0 = max(0.5, mu_est * S0)
        beta0 = max(0.5, (1 - mu_est) * S0)

        def F(x):
            a, b = math.exp(x[0]), math.exp(x[1])
            return np.array([
                stats.beta.ppf(p1, a, b) - q1,
                stats.beta.ppf(p2, a, b) - q2,
            ])

        x = newton_2d(F, [math.log(alpha0), math.log(beta0)])
        alpha, beta = math.exp(x[0]), math.exp(x[1])
        return stats.beta(a=alpha, b=beta)

    @staticmethod
    def from_mean_quantile(mu, p, q):
        def f(log_S):
            S = math.exp(log_S)
            return stats.beta.ppf(p, mu * S, (1 - mu) * S) - q
        log_S = find_root_1d(f, x0=1.0)
        S = math.exp(log_S)
        return stats.beta(a=mu * S, b=(1 - mu) * S)

    DISPATCH = {
        MeanVarSpec:      lambda s: BetaDist.from_mean_var(s.mean, s.var),
        MeanModeSpec:     lambda s: BetaDist.from_mean_mode(s.mean, s.mode),
        TwoQuantileSpec:  lambda s: BetaDist.from_two_quantiles(s.p1, s.q1, s.p2, s.q2),
        MeanQuantileSpec: lambda s: BetaDist.from_mean_quantile(s.mean, s.p, s.q),
    }


# ===========================================================================
# Discrete
# ===========================================================================

# ---- Binomial -------------------------------------------------------------
class BinomialDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("binomial", mu, var)
        p = 1 - var / mu
        n = round(mu / p)
        return stats.binom(n=n, p=p)

    DISPATCH = {
        MeanVarSpec: lambda s: BinomialDist.from_mean_var(s.mean, s.var),
    }


# ---- Poisson --------------------------------------------------------------
class PoissonDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("poisson", mu, var)
        return stats.poisson(mu=mu)

    @staticmethod
    def from_mean(mu):
        if mu <= 0:
            raise ValueError("Poisson: mean must be > 0")
        return stats.poisson(mu=mu)

    @staticmethod
    def from_var(var):
        return PoissonDist.from_mean(var)  # var = mean for Poisson

    DISPATCH = {
        MeanVarSpec: lambda s: PoissonDist.from_mean_var(s.mean, s.var),
        MeanSpec:    lambda s: PoissonDist.from_mean(s.mean),
        VarSpec:     lambda s: PoissonDist.from_var(s.var),
    }


# ---- Truncated Poisson ----------------------------------------------------
# Like Julia's `dist_from_mean(Truncated{<:Poisson}, μ̄)`: solve for λ such that
# the truncated mean equals the target. Returned as a _TruncatedPoisson wrapper
# (scipy has no native truncated-Poisson form).
def _truncated_poisson_moments(lam, lo, hi):
    """Direct summation: mean & var of Poisson(λ) truncated to [lo, hi]."""
    lo_i = int(math.ceil(lo))
    hi_i = int(math.floor(hi))
    Z = stats.poisson.cdf(hi_i, lam) - (
        stats.poisson.cdf(lo_i - 1, lam) if lo_i > 0 else 0.0
    )
    if Z <= 0:
        return float("nan"), float("nan")
    m = 0.0
    m2 = 0.0
    for k in range(lo_i, hi_i + 1):
        pk = stats.poisson.pmf(k, lam) / Z
        m += k * pk
        m2 += k * k * pk
    return m, m2 - m * m


def truncated_poisson(lo, hi, mean):
    """Construct a Poisson on integers ``[lo, hi]`` with the given truncated mean.

    The Poisson rate λ is solved by 1D root-finding such that the truncated
    mean matches.
    """
    from scipy.optimize import brentq

    if not (lo < mean < hi):
        raise ValueError(f"Truncated Poisson: mean must be in ({lo}, {hi})")
    if not (float(lo).is_integer() and float(hi).is_integer()):
        raise ValueError("Truncated Poisson: lo and hi must be integers")

    def residual(lam):
        return _truncated_poisson_moments(lam, lo, hi)[0] - mean

    lo_lam = max(mean / 4, 1e-3)
    hi_lam = max(mean * 4, lo_lam + 1.0)
    while residual(lo_lam) > 0 and lo_lam > 1e-8:
        lo_lam /= 2
    while residual(hi_lam) < 0:
        hi_lam *= 2

    lam = brentq(residual, lo_lam, hi_lam)
    parent = stats.poisson(mu=lam)
    from ._support import _TruncatedDist
    return _TruncatedDist(parent, lo, hi)


# ---- Negative Binomial ----------------------------------------------------
# scipy.nbinom(n=r, p). Support {0,1,...}. Mean = r*(1-p)/p, Var = r*(1-p)/p^2.
class NegativeBinomialDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("negative_binomial", mu, var)
        p = mu / var
        r = mu ** 2 / (var - mu)
        return stats.nbinom(n=r, p=p)

    DISPATCH = {
        MeanVarSpec: lambda s: NegativeBinomialDist.from_mean_var(s.mean, s.var),
    }


# ---- Geometric ------------------------------------------------------------
# Julia's Geometric(p) is on {0,1,2,...}, mean = (1-p)/p. We use nbinom(1, p)
# (NOT scipy.geom — that one is on {1,2,...} with mean 1/p).
class GeometricDist:
    @staticmethod
    def from_mean(mu):
        if mu <= 0:
            raise ValueError("Geometric: mean must be > 0")
        p = 1 / (1 + mu)
        return stats.nbinom(n=1, p=p)

    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("geometric", mu, var)
        return GeometricDist.from_mean(mu)

    @staticmethod
    def from_var(var):
        # var = mu*(1+mu) -> mu = (-1 + sqrt(1+4*var))/2
        if var <= 0:
            raise ValueError("Geometric: var must be > 0")
        mu = (-1 + math.sqrt(1 + 4 * var)) / 2
        return GeometricDist.from_mean(mu)

    @staticmethod
    def from_quantile(p, q):
        if not (0 < p < 1):
            raise ValueError("p must be in (0,1)")
        if q < 0 or not float(q).is_integer():
            raise ValueError("Geometric: quantile must be a non-negative integer")
        # P(X <= k) = 1 - (1-prob)^(k+1) -> prob = 1 - (1-p)^(1/(q+1))
        prob = 1 - (1 - p) ** (1 / (q + 1))
        if not (0 < prob < 1):
            raise ValueError("Geometric: degenerate quantile spec")
        return stats.nbinom(n=1, p=prob)

    DISPATCH = {
        MeanSpec:     lambda s: GeometricDist.from_mean(s.mean),
        MeanVarSpec:  lambda s: GeometricDist.from_mean_var(s.mean, s.var),
        VarSpec:      lambda s: GeometricDist.from_var(s.var),
        QuantileSpec: lambda s: GeometricDist.from_quantile(s.p, s.q),
    }


# ---- Discrete Symmetric Triangular ---------------------------------------
# Var = n(n+2)/6; closed-form recovery of n from var.
class DiscreteSymTriangularDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("discrete_sym_triangular", mu, var)
        n = round(-1 + math.sqrt(1 + 6 * var))
        return DiscreteSymmetricTriangularDist(mu=int(round(mu)), n=n)

    DISPATCH = {
        MeanVarSpec: lambda s: DiscreteSymTriangularDist.from_mean_var(s.mean, s.var),
    }


# ---- Discrete Triangular -------------------------------------------------
# 3 integer params (a, b, c). mean+var alone is underdetermined; we use the
# (mean, var, mode) factory like Julia's `dist_from_mean_var_mode`.
class DiscreteTriDist:
    @staticmethod
    def from_mean_var_mode(mu, var, c):
        # Solve the *continuous* triangular for (a, b), round, then search a
        # +/- 1 neighborhood for the integer triple whose moments are closest.
        cont_dist = TriangularDist.from_mean_var_mode(mu, var, c)
        loc = cont_dist.kwds["loc"]
        scale = cont_dist.kwds["scale"]
        c_scipy = cont_dist.kwds["c"]
        a_cont, b_cont = loc, loc + scale
        c_int = int(round(c))
        a0 = int(round(a_cont))
        b0 = int(round(b_cont))

        best = None
        best_err = float("inf")
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                a_try = min(a0 + da, c_int)
                b_try = max(b0 + db, c_int)
                if not (a_try <= c_int <= b_try):
                    continue
                d = DiscreteTriangularDist(a=a_try, b=b_try, c=c_int)
                err = ((d.mean() - mu) / max(abs(mu), 1.0)) ** 2 + \
                      ((d.var() - var) / max(var, 1.0)) ** 2
                if err < best_err:
                    best_err = err
                    best = d
        if best is None:
            raise ValueError(
                f"DiscreteTriangular: no integer triple satisfies "
                f"(mean={mu}, var={var}, mode={c})"
            )
        return best

    DISPATCH = {
        MeanVarModeSpec: lambda s: DiscreteTriDist.from_mean_var_mode(s.mean, s.var, s.mode),
    }


# ---- Discrete Uniform -----------------------------------------------------
# scipy.randint(low, high) on {low, low+1, ..., high-1}. Julia's DiscreteUniform(a,b)
# is on {a,...,b} inclusive — we pass high = b + 1.
class DiscreteUniformDist:
    @staticmethod
    def from_mean_var(mu, var):
        require_mean_var("discrete_uniform", mu, var)
        n = round(-1 + math.sqrt(1 + 12 * var))
        a = round(mu - n / 2)
        b = a + n
        return stats.randint(low=a, high=b + 1)

    DISPATCH = {
        MeanVarSpec: lambda s: DiscreteUniformDist.from_mean_var(s.mean, s.var),
    }


# ===========================================================================
# Master dispatch table: canonical name -> handler class
# ===========================================================================

DIST_HANDLERS = {
    # Real-line continuous
    "normal":            NormalDist,
    "laplace":           LaplaceDist,
    "logistic":          LogisticDist,
    "gumbel":            GumbelDist,
    "cauchy":            CauchyDist,
    "tdist":             TDistDist,
    "uniform":           UniformDist,
    "sym_triangular":    SymTriangularDist,
    "triangular":        TriangularDist,
    # Positive continuous
    "gamma":             GammaDist,
    "erlang":            ErlangDist,
    "exponential":       ExponentialDist,
    "lognormal":         LogNormalDist,
    "weibull":           WeibullDist,
    "frechet":           FrechetDist,
    "chi":               ChiDist,
    "chisq":             ChisqDist,
    "rayleigh":          RayleighDist,
    "fdist":             FDistDist,
    "inverse_gamma":     InverseGammaDist,
    "pareto":            ParetoDist,
    "folded_normal":     FoldedNormalDist,
    # Unit-interval continuous
    "beta":              BetaDist,
    # Discrete
    "binomial":          BinomialDist,
    "poisson":           PoissonDist,
    "negative_binomial": NegativeBinomialDist,
    "geometric":         GeometricDist,
    "discrete_uniform":  DiscreteUniformDist,
    "discrete_sym_triangular": DiscreteSymTriangularDist,
    "discrete_triangular":     DiscreteTriDist,
}

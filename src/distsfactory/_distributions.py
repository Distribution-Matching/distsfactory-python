"""Distribution-specific moment and quantile matching.

Each distribution provides:
- from_*  functions that return a frozen scipy.stats distribution
- exists_mean_var  that checks feasibility for given mean and variance
"""

import math
import numpy as np
from scipy import stats
from ._solvers import find_root_1d, newton_2d
from ._specs import (
    MeanVarSpec, MeanSpec, VarSpec, QuantileSpec, TwoQuantileSpec,
    MeanQuantileSpec, MeanModeSpec, ModeVarSpec, ModeQuantileSpec, ModeIQRSpec,
)


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------
# scipy parameterization: gamma(a, loc, scale) where a=shape, scale=theta
# mean = a * scale, var = a * scale^2

class GammaDist:
    @staticmethod
    def from_mean_var(mu, var):
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
        # Solve: (alpha - 1) * sqrt(var / alpha) == mode
        def f(log_alpha):
            alpha = math.exp(log_alpha) + 1  # ensure alpha > 1
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
            alpha = math.exp(log_alpha) + 1  # ensure alpha > 1
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
        # mean = alpha * theta, so theta = mu / alpha
        # Work in log-space to keep alpha > 0
        def f(log_alpha):
            alpha = math.exp(log_alpha)
            theta = mu / alpha
            return stats.gamma.ppf(p, a=alpha, scale=theta) - q
        log_alpha = find_root_1d(f, x0=0.0)
        alpha = math.exp(log_alpha)
        theta = mu / alpha
        return stats.gamma(a=alpha, scale=theta)

    @staticmethod
    def exists_mean_var(mu, var):
        return mu > 0 and var > 0

    DISPATCH = {
        MeanVarSpec: lambda s: GammaDist.from_mean_var(s.mean, s.var),
        MeanModeSpec: lambda s: GammaDist.from_mean_mode(s.mean, s.mode),
        ModeVarSpec: lambda s: GammaDist.from_mode_var(s.mode, s.var),
        ModeQuantileSpec: lambda s: GammaDist.from_mode_quantile(s.mode, s.p, s.q),
        ModeIQRSpec: lambda s: GammaDist.from_mode_iqr(s.mode, s.iqr),
        TwoQuantileSpec: lambda s: GammaDist.from_two_quantiles(s.p1, s.q1, s.p2, s.q2),
        MeanQuantileSpec: lambda s: GammaDist.from_mean_quantile(s.mean, s.p, s.q),
    }


# ---------------------------------------------------------------------------
# Exponential
# ---------------------------------------------------------------------------
# scipy parameterization: expon(loc, scale) where scale = mean

class ExponentialDist:
    @staticmethod
    def from_mean(mu):
        if mu <= 0:
            raise ValueError(f"Exponential mean must be positive, got {mu}")
        return stats.expon(scale=mu)

    @staticmethod
    def from_mean_var(mu, var):
        if not math.isclose(var, mu ** 2, rel_tol=1e-6):
            raise ValueError(
                f"Exponential variance must equal mean^2. "
                f"Got mean={mu}, var={var}, expected var={mu**2}"
            )
        return stats.expon(scale=mu)

    @staticmethod
    def from_var(var):
        mu = math.sqrt(var)
        return stats.expon(scale=mu)

    @staticmethod
    def from_quantile(p, q):
        theta = -q / math.log(1 - p)
        return stats.expon(scale=theta)

    @staticmethod
    def exists_mean_var(mu, var):
        return mu > 0 and var > 0 and math.isclose(var, mu ** 2, rel_tol=1e-6)

    DISPATCH = {
        MeanVarSpec: lambda s: ExponentialDist.from_mean_var(s.mean, s.var),
        MeanSpec: lambda s: ExponentialDist.from_mean(s.mean),
        VarSpec: lambda s: ExponentialDist.from_var(s.var),
        QuantileSpec: lambda s: ExponentialDist.from_quantile(s.p, s.q),
    }


# ---------------------------------------------------------------------------
# Logistic
# ---------------------------------------------------------------------------
# scipy parameterization: logistic(loc, scale) where loc=mu, scale=s
# mean = loc, var = s^2 * pi^2 / 3

class LogisticDist:
    @staticmethod
    def from_mean_var(mu, var):
        s = math.sqrt(3 * var / math.pi ** 2)
        return stats.logistic(loc=mu, scale=s)

    @staticmethod
    def from_two_quantiles(p1, q1, p2, q2):
        # Logistic is location-scale: quantile(p) = mu + s * log(p/(1-p))
        z1 = math.log(p1 / (1 - p1))
        z2 = math.log(p2 / (1 - p2))
        s = (q2 - q1) / (z2 - z1)
        mu = q1 - s * z1
        return stats.logistic(loc=mu, scale=s)

    @staticmethod
    def from_mode_iqr(mode, iqr):
        # For Logistic: mode = mu, IQR = 2 * s * ln(3)
        s = iqr / (2 * math.log(3))
        return stats.logistic(loc=mode, scale=s)

    @staticmethod
    def from_mean_quantile(mu, p, q):
        # mu + s * log(p/(1-p)) = q  =>  s = (q - mu) / log(p/(1-p))
        z = math.log(p / (1 - p))
        if math.isclose(z, 0.0, abs_tol=1e-12):
            # p=0.5 means quantile is the median, which equals the mean for Logistic.
            # mean + median alone can't determine scale.
            if math.isclose(mu, q, rel_tol=1e-6):
                raise ValueError(
                    "Logistic mean and median are always equal; "
                    "need an additional constraint to determine scale"
                )
            else:
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

    @staticmethod
    def from_quantile(p, q):
        # Single quantile: underdetermined for 2 params, but if we assume
        # mean = mode = q when p = 0.5 (median), we can at least handle that.
        # For a single non-median quantile, we cannot determine both params.
        if math.isclose(p, 0.5, abs_tol=1e-12):
            # median = loc for logistic; scale is undetermined, use default
            raise ValueError(
                "Single quantile for Logistic requires additional constraints "
                "(need at least two constraints for a 2-parameter distribution)"
            )
        raise ValueError(
            "Single quantile for Logistic requires additional constraints"
        )

    @staticmethod
    def exists_mean_var(mu, var):
        return var > 0  # any real mean works

    DISPATCH = {
        MeanVarSpec: lambda s: LogisticDist.from_mean_var(s.mean, s.var),
        TwoQuantileSpec: lambda s: LogisticDist.from_two_quantiles(s.p1, s.q1, s.p2, s.q2),
        ModeIQRSpec: lambda s: LogisticDist.from_mode_iqr(s.mode, s.iqr),
        MeanQuantileSpec: lambda s: LogisticDist.from_mean_quantile(s.mean, s.p, s.q),
    }


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------
# scipy parameterization: beta(a, b) on [0, 1]
# mean = a/(a+b), var = a*b / ((a+b)^2 * (a+b+1))

class BetaDist:
    @staticmethod
    def from_mean_var(mu, var):
        S = mu * (1 - mu) / var - 1
        if S <= 0:
            raise ValueError(
                f"Beta variance too large: var={var} >= mu*(1-mu)={mu*(1-mu)}"
            )
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
        from scipy.stats import norm
        z1 = norm.ppf(p1)
        z2 = norm.ppf(p2)
        mu_est = (q1 * z2 - q2 * z1) / (z2 - z1)
        sigma_est = (q2 - q1) / (z2 - z1)
        var_est = sigma_est ** 2
        # Clamp to valid Beta range
        mu_est = max(0.01, min(0.99, mu_est))
        var_est = min(var_est, mu_est * (1 - mu_est) * 0.9)
        S0 = mu_est * (1 - mu_est) / var_est - 1
        alpha0 = max(0.5, mu_est * S0)
        beta0 = max(0.5, (1 - mu_est) * S0)

        def F(x):
            a, b = math.exp(x[0]), math.exp(x[1])
            r1 = stats.beta.ppf(p1, a, b) - q1
            r2 = stats.beta.ppf(p2, a, b) - q2
            return np.array([r1, r2])

        x = newton_2d(F, [math.log(alpha0), math.log(beta0)])
        alpha, beta = math.exp(x[0]), math.exp(x[1])
        return stats.beta(a=alpha, b=beta)

    @staticmethod
    def from_mean_quantile(mu, p, q):
        # S = alpha + beta; alpha = mu*S, beta = (1-mu)*S
        def f(log_S):
            S = math.exp(log_S)
            a = mu * S
            b = (1 - mu) * S
            return stats.beta.ppf(p, a, b) - q
        log_S = find_root_1d(f, x0=1.0)
        S = math.exp(log_S)
        alpha = mu * S
        beta = (1 - mu) * S
        return stats.beta(a=alpha, b=beta)

    @staticmethod
    def exists_mean_var(mu, var):
        return 0 < mu < 1 and 0 < var < mu * (1 - mu)

    DISPATCH = {
        MeanVarSpec: lambda s: BetaDist.from_mean_var(s.mean, s.var),
        MeanModeSpec: lambda s: BetaDist.from_mean_mode(s.mean, s.mode),
        TwoQuantileSpec: lambda s: BetaDist.from_two_quantiles(s.p1, s.q1, s.p2, s.q2),
        MeanQuantileSpec: lambda s: BetaDist.from_mean_quantile(s.mean, s.p, s.q),
    }


# ---------------------------------------------------------------------------
# Master dispatch table: dist_name -> handler class
# ---------------------------------------------------------------------------

DIST_HANDLERS = {
    "gamma": GammaDist,
    "exponential": ExponentialDist,
    "logistic": LogisticDist,
    "beta": BetaDist,
}

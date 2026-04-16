"""Specification types and parser for moment/quantile constraints."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MeanVarSpec:
    mean: float
    var: float


@dataclass(frozen=True)
class MeanSpec:
    mean: float


@dataclass(frozen=True)
class VarSpec:
    var: float


@dataclass(frozen=True)
class QuantileSpec:
    p: float
    q: float


@dataclass(frozen=True)
class TwoQuantileSpec:
    p1: float
    q1: float
    p2: float
    q2: float


@dataclass(frozen=True)
class MeanQuantileSpec:
    mean: float
    p: float
    q: float


@dataclass(frozen=True)
class MeanModeSpec:
    mean: float
    mode: float


@dataclass(frozen=True)
class ModeVarSpec:
    mode: float
    var: float


@dataclass(frozen=True)
class ModeQuantileSpec:
    mode: float
    p: float
    q: float


@dataclass(frozen=True)
class ModeIQRSpec:
    mode: float
    iqr: float


def parse_spec(
    *,
    mean: Optional[float] = None,
    var: Optional[float] = None,
    std: Optional[float] = None,
    cv: Optional[float] = None,
    scv: Optional[float] = None,
    second_moment: Optional[float] = None,
    median: Optional[float] = None,
    q1: Optional[float] = None,
    q3: Optional[float] = None,
    iqr: Optional[float] = None,
    quantiles: Optional[list] = None,
    mode: Optional[float] = None,
):
    """Parse keyword arguments into a typed specification object.

    Parameters
    ----------
    mean : float, optional
        Target mean.
    var : float, optional
        Target variance.
    std : float, optional
        Target standard deviation (converted to variance).
    cv : float, optional
        Coefficient of variation (requires mean).
    scv : float, optional
        Squared coefficient of variation (requires mean).
    second_moment : float, optional
        E[X^2] (requires mean to compute variance).
    median : float, optional
        Target median (p=0.5 quantile).
    q1 : float, optional
        First quartile (p=0.25).
    q3 : float, optional
        Third quartile (p=0.75).
    iqr : float, optional
        Interquartile range.
    quantiles : list of (p, q) tuples, optional
        Two arbitrary quantile constraints.
    mode : float, optional
        Target mode.

    Returns
    -------
    spec : one of the Spec dataclasses
    """
    # Resolve variance from alternative dispersion measures
    if var is None and std is not None:
        var = std ** 2
    elif var is None and mean is not None and cv is not None:
        var = (cv * mean) ** 2
    elif var is None and mean is not None and scv is not None:
        var = scv * mean ** 2
    elif var is None and mean is not None and second_moment is not None:
        var = second_moment - mean ** 2

    # Mode-based specs
    if mode is not None:
        if mean is not None and var is None:
            return MeanModeSpec(mean, mode)
        if var is not None:
            return ModeVarSpec(mode, var)
        if median is not None:
            return ModeQuantileSpec(mode, 0.5, median)
        if q1 is not None:
            return ModeQuantileSpec(mode, 0.25, q1)
        if q3 is not None:
            return ModeQuantileSpec(mode, 0.75, q3)
        if iqr is not None:
            return ModeIQRSpec(mode, iqr)

    # Quantile-based specs
    if quantiles is not None:
        if len(quantiles) != 2:
            raise ValueError("quantiles must be a list of 2 (p, q) tuples")
        (p1, q1_val), (p2, q2_val) = quantiles
        return TwoQuantileSpec(p1, q1_val, p2, q2_val)
    if q1 is not None and q3 is not None:
        return TwoQuantileSpec(0.25, q1, 0.75, q3)
    if median is not None and iqr is not None:
        return TwoQuantileSpec(0.25, median - iqr / 2, 0.75, median + iqr / 2)
    if mean is not None and median is not None:
        return MeanQuantileSpec(mean, 0.5, median)
    if mean is not None and q1 is not None:
        return MeanQuantileSpec(mean, 0.25, q1)
    if mean is not None and q3 is not None:
        return MeanQuantileSpec(mean, 0.75, q3)
    if median is not None:
        return QuantileSpec(0.5, median)
    if q1 is not None:
        return QuantileSpec(0.25, q1)
    if q3 is not None:
        return QuantileSpec(0.75, q3)

    # Moment-based specs
    if mean is not None and var is not None:
        return MeanVarSpec(mean, var)
    if mean is not None:
        return MeanSpec(mean)
    if var is not None:
        return VarSpec(var)

    raise ValueError(
        "Must provide at least one moment or quantile specification"
    )

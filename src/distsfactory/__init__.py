"""distsfactory: Construct probability distributions from partial specifications."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("distsfactory")
except PackageNotFoundError:  # editable / source checkout without install
    __version__ = "0.0.0+unknown"

from ._api import make_dist, dist_exists, available_distributions
from ._partial import PartialDist, partial_dist

__all__ = [
    "make_dist", "dist_exists", "available_distributions",
    "PartialDist", "partial_dist",
]

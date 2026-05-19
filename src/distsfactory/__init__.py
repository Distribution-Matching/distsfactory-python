"""distsfactory: Construct probability distributions from partial specifications."""

__version__ = "0.1.0"

from ._api import make_dist, dist_exists, available_distributions
from ._partial import PartialDist, partial_dist

__all__ = [
    "make_dist", "dist_exists", "available_distributions",
    "PartialDist", "partial_dist",
]

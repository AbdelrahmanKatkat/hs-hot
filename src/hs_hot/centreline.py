"""Roads as centrelines: thin a road mask to a one-pixel line, and widen a line again (boolean masks; `centreline` extra).

The library's own output is a band of polygons (`decode.py`). When roads are judged as lines, the band is thinned to its middle
line, and a line is widened by a distance when it is compared as an area. Both work on `(H, W)` masks in model or chip pixels and know
nothing of CRS or ground units: the caller turns metres into pixels. `scikit-image` and `scipy` are imported when a function runs,
so `import hs_hot` stays light; install them with the `centreline` extra. Not exported from `hs_hot` (import `hs_hot.centreline`).
"""

import numpy as np


def _as_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {mask.shape}")
    return mask


def thin_to_centreline(mask: np.ndarray) -> np.ndarray:
    """One-pixel-wide centreline of a boolean road mask (skeletonisation); an empty mask gives an empty line."""
    from skimage.morphology import skeletonize

    return skeletonize(_as_mask(mask))


def widen_centreline(line: np.ndarray, radius_px: float) -> np.ndarray:
    """The pixels within `radius_px` of a centreline's pixels (boolean); an empty line gives an empty mask."""
    from scipy.ndimage import distance_transform_edt

    line = _as_mask(line)
    if radius_px < 0:
        raise ValueError(f"radius_px must be >= 0, got {radius_px}")
    if not line.any():
        return np.zeros_like(line)
    return distance_transform_edt(~line) <= radius_px

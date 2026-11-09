"""Model pixels -> chip pixels -> EPSG:4326 polygons (no ONNX layout knowledge)."""

from typing import Any

import numpy as np
from pyproj import Transformer

from .preprocess import PreprocessMeta


def ring_to_chip_pixels(ring: np.ndarray, meta: PreprocessMeta) -> np.ndarray:
    """Undo the model resize -> source chip pixels, for an `(N, 2)` ring of `(col, row)` vertices."""
    return ring / np.array([meta.scale_x, meta.scale_y])


def polygons_to_chip_pixels(polygons: list[dict[str, Any]], meta: PreprocessMeta) -> list[dict[str, Any]]:
    """Polygons in model-input pixels -> the same polygons in source-chip pixels."""
    return [
        {**polygon, "rings": [ring_to_chip_pixels(ring, meta) for ring in polygon["rings"]]}
        for polygon in polygons
    ]


def ring_to_epsg4326(ring: np.ndarray, transform: Any, crs: Any) -> list[list[float]]:
    """One chip-pixel ring -> closed EPSG:4326 ring `[[lon, lat], ...]`.

    A chip without a CRS (PNG, JPEG) has no map position: its ring is returned in the chip's own coordinates.
    """
    # Explicit affine maths: works across `affine` versions (`*` is deprecated, `@` needs affine>=3).
    cols, rows = ring[:, 0], ring[:, 1]
    x = transform.a * cols + transform.b * rows + transform.c
    y = transform.d * cols + transform.e * rows + transform.f
    if crs is not None and str(crs) != "EPSG:4326":
        x, y = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(x, y)
    return np.stack([x, y], axis=1).tolist()

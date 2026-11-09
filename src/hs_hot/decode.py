"""ONNX probability map -> polygons in model-input pixels (no CRS / affine dependence).

The exported graph already did the network and the sigmoid. What is left, and stays tunable, is an optional smoothing
of the probability, the threshold, an optional opening / closing of the mask, the tracing of the pixel edges, the minimum
area, the filling of small holes and the simplification (`PostprocessParams`).

A polygon is a dict `{"rings", "confidence", "class_id"}`: `rings` is a list of `(N, 2)` arrays of `(col, row)` vertices,
the exterior first and then its holes, each with its first vertex repeated at the end; `confidence` is the mean tree
probability over the polygon's pixels. In order of use: `smooth_probability`, `threshold_mask`, `open_mask`,
`close_mask`, `trace_polygons`, `simplify_ring`, `polygon_confidence`, `decode_polygons`.
"""

from typing import Any

import numpy as np
from rasterio.features import rasterize, shapes

from .params import PostprocessParams

CLASS_ID = 1  # highway


def threshold_mask(probability: np.ndarray, threshold: float) -> np.ndarray:
    """`(H, W)` probability -> `(H, W)` `uint8` mask, 1 where the probability reaches `threshold`."""
    return (probability >= threshold).astype(np.uint8)


def smooth_probability(probability: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing of a `(H, W)` probability (separable, edges repeated); `sigma` 0 returns it unchanged."""
    if sigma <= 0:
        return probability
    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    smoothed = probability.astype(np.float32)
    for axis in (0, 1):
        padded = np.pad(smoothed, [(radius, radius) if a == axis else (0, 0) for a in (0, 1)], mode="edge")
        size = smoothed.shape[axis]
        smoothed = sum(
            weight * np.take(padded, np.arange(shift, shift + size), axis=axis)
            for weight, shift in zip(kernel, range(2 * radius + 1))
        ).astype(np.float32)
    return smoothed


def _slide(mask: np.ndarray, radius: int, take_max: bool) -> np.ndarray:
    """Dilation (`take_max`) or erosion of a boolean mask with a square of side `2 * radius + 1`; outside = background."""
    for axis in (0, 1):
        pad = [(radius, radius) if a == axis else (0, 0) for a in (0, 1)]
        padded = np.pad(mask, pad, mode="constant", constant_values=False)
        size = mask.shape[axis]
        windows = [
            np.take(padded, np.arange(shift, shift + size), axis=axis) for shift in range(2 * radius + 1)
        ]
        mask = np.logical_or.reduce(windows) if take_max else np.logical_and.reduce(windows)
    return mask


def open_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Erosion then dilation: removes specks and thin bridges narrower than `2 * radius + 1` pixels."""
    if radius <= 0:
        return mask
    return _slide(_slide(mask > 0, radius, take_max=False), radius, take_max=True).astype(np.uint8)


def close_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilation then erosion: fills gaps and holes narrower than `2 * radius + 1` pixels (the chip edge is not eroded)."""
    if radius <= 0:
        return mask
    padded = np.pad(mask > 0, radius, mode="edge")  # a region touching the chip edge keeps its edge
    closed = _slide(_slide(padded, radius, take_max=True), radius, take_max=False)
    return closed[radius:-radius, radius:-radius].astype(np.uint8)


def trace_polygons(mask: np.ndarray) -> list[list[np.ndarray]]:
    """The pixel edges of a binary mask as lists of rings. A region with a hole is one polygon with an interior ring.

    Pixels that only touch diagonally stay separate polygons (4-connectivity).
    """
    foreground = mask > 0
    if not foreground.any():
        return []
    traced = shapes(foreground.astype(np.uint8), mask=foreground, connectivity=4)
    return [[np.asarray(ring, dtype=float) for ring in geometry["coordinates"]] for geometry, _ in traced]


def _ring_area(ring: np.ndarray) -> float:
    """Unsigned area of a closed ring (shoelace)."""
    cols, rows = ring[:, 0], ring[:, 1]
    return abs(float(np.dot(cols[:-1], rows[1:]) - np.dot(cols[1:], rows[:-1]))) / 2


def _polygon_area(rings: list[np.ndarray]) -> float:
    exterior, *holes = rings
    return _ring_area(exterior) - sum(_ring_area(hole) for hole in holes)


def _distance_to_line(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Distance of each point to the line through `start` and `end` (to `start` if they coincide)."""
    direction = end - start
    length = float(np.hypot(*direction))
    if length == 0.0:
        return np.hypot(*(points - start).T)
    return (
        np.abs(direction[0] * (points[:, 1] - start[1]) - direction[1] * (points[:, 0] - start[0])) / length
    )


def _simplify_path(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Douglas-Peucker on an open path: the vertices that keep every dropped vertex within `tolerance` of the line."""
    keep = np.zeros(len(points), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end - start < 2:
            continue
        distances = _distance_to_line(points[start + 1 : end], points[start], points[end])
        farthest = int(np.argmax(distances))
        if distances[farthest] > tolerance:
            middle = start + 1 + farthest
            keep[middle] = True
            stack += [(start, middle), (middle, end)]
    return points[keep]


def simplify_ring(ring: np.ndarray, tolerance: float) -> np.ndarray | None:
    """A closed ring with fewer vertices, none of the dropped ones further than `tolerance` pixels from the result.

    Returns `None` if fewer than three vertices remain, which is a ring that has collapsed. A tolerance of 0 returns
    the ring as it is.
    """
    if tolerance <= 0:
        return ring
    vertices = ring[:-1]
    if len(vertices) <= 3:
        return ring
    split = int(
        np.argmax(np.hypot(*(vertices - vertices[0]).T))
    )  # the farthest vertex splits the ring in two paths
    first = _simplify_path(vertices[: split + 1], tolerance)
    second = _simplify_path(np.vstack([vertices[split:], vertices[:1]]), tolerance)
    simplified = np.vstack([first[:-1], second])
    return simplified if len(simplified) >= 4 else None


def polygon_confidence(probability: np.ndarray, rings: list[np.ndarray]) -> float:
    """Mean road probability over the pixels inside the polygon (its holes excluded)."""
    exterior = rings[0]
    top, left = max(0, int(np.floor(exterior[:, 1].min()))), max(0, int(np.floor(exterior[:, 0].min())))
    bottom = min(probability.shape[0], int(np.ceil(exterior[:, 1].max())))
    right = min(probability.shape[1], int(np.ceil(exterior[:, 0].max())))
    window = probability[top:bottom, left:right]
    if window.size == 0:
        return 0.0
    shifted = [{"type": "Polygon", "coordinates": [(ring - [left, top]).tolist() for ring in rings]}]
    inside = rasterize(shifted, out_shape=window.shape, fill=0, default_value=1, dtype="uint8").astype(bool)
    return float(window[inside].mean()) if inside.any() else float(window.max())


def decode_polygons(probability: Any, cfg: PostprocessParams | None = None) -> list[dict[str, Any]]:
    """Mirror of the tail of the serving path for one chip.

    Args:
        probability: `(H, W)` road probability in model-input pixels (graph output `probability`).
    Returns:
        Polygons as dicts `rings, confidence, class_id`, in model-input pixels, those under `min_area_px` dropped.
    """
    cfg = cfg or PostprocessParams()
    probability = smooth_probability(np.asarray(probability, dtype=np.float32), cfg.smooth_sigma_px)
    mask = close_mask(open_mask(threshold_mask(probability, cfg.threshold), cfg.open_px), cfg.close_px)
    polygons = []
    for rings in trace_polygons(mask):
        if _polygon_area(rings) < cfg.min_area_px:
            continue
        rings = [rings[0]] + [hole for hole in rings[1:] if _ring_area(hole) >= cfg.fill_holes_px]
        exterior, *holes = (simplify_ring(ring, cfg.simplify_px) for ring in rings)
        if exterior is None:
            continue
        kept = [exterior] + [hole for hole in holes if hole is not None]
        polygons.append(
            {"rings": kept, "confidence": polygon_confidence(probability, kept), "class_id": CLASS_ID}
        )
    return polygons

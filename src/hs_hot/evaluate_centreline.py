"""Scoring roads as centrelines with a distance tolerance (numpy, Pillow and scipy; `centreline` extra).

A predicted line `P` (a one-pixel mask, for example `centreline.thin_to_centreline` of a thresholded probability) is compared with the
reference line `R` (the OSM centreline, `draw_lines`). With `d_R(x)` the distance in metres from pixel `x` to the nearest pixel of `R`
(`d_P` likewise) and a tolerance `r` in metres, counts pooled over chips:

    precision(r) = |{p in P : d_R(p) <= r}| / |P|        recall(r) = |{q in R : d_P(q) <= r}| / |R|
    F1           = 2 * precision * recall / (precision + recall)
    quality(r)   = TPr / (TPr + FP + FN)   TPr = matched reference pixels, FP = |P| - matched predicted, FN = |R| - TPr
    IoU(r)       = |A and B| / |A or B|    A = {x : d_P(x) <= r}, B = {x : d_R(x) <= r}   (both lines widened by r on each side)

`r = 1 m` means a point matches if the other line passes within 1 m of it on either side (a corridor 2 m wide). Pixel counts stand in for
lengths (a diagonal step is about 1.4 pixels; the same bias is on both lines), and matching is not one to one. Pooling over chips means a
chip without roads adds its predicted pixels as false positives and cannot make a score undefined. `metres_per_pixel` is per chip.
Not exported from `hs_hot` (import `hs_hot.evaluate_centreline`).
"""

from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw

RADII_M = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
HIST_BIN_M, HIST_MAX_M = 0.1, 5.0  # distances of predicted pixels to the reference, for the median offset


def draw_lines(
    lines: Sequence[np.ndarray],
    height: int,
    width: int,
    offset: tuple[float, float] = (0.0, 0.0),
    line_width: int = 1,
) -> np.ndarray:
    """Boolean `(height, width)` raster of polylines given as `(N, 2)` arrays of `(col, row)`; `offset` is a (col, row) shift in pixels."""
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for line in lines:
        draw.line([(float(c + offset[0]), float(r + offset[1])) for c, r in line], fill=1, width=line_width)
    return np.asarray(canvas, dtype=bool)


def distance_to(mask: np.ndarray) -> np.ndarray:
    """Distance in pixels from every pixel to the nearest True pixel of `mask` (infinite everywhere if it has none)."""
    from scipy.ndimage import distance_transform_edt

    if not mask.any():
        return np.full(mask.shape, np.inf)
    return distance_transform_edt(~mask)


class Tally:
    """Pooled counts of one predicted-line source against the reference lines; `add` per chip, `summary` at the end."""

    def __init__(self, radii: Sequence[float] = RADII_M) -> None:
        self.radii = tuple(radii)
        self.n_pred = 0
        self.n_ref = 0
        self.pred_within = np.zeros(len(self.radii))
        self.ref_within = np.zeros(len(self.radii))
        self.inter = np.zeros(len(self.radii))
        self.union = np.zeros(len(self.radii))
        self.hist = np.zeros(
            int(HIST_MAX_M / HIST_BIN_M) + 1
        )  # the last bin holds everything farther than HIST_MAX_M

    def add(self, pred: np.ndarray, ref: np.ndarray, metres_per_pixel: float) -> None:
        """Add one chip: `pred` and `ref` are boolean line rasters of the same shape."""
        pred, ref = np.asarray(pred, dtype=bool), np.asarray(ref, dtype=bool)
        if pred.shape != ref.shape:
            raise ValueError(f"prediction {pred.shape} and reference {ref.shape} differ in shape")
        if metres_per_pixel <= 0:
            raise ValueError(f"metres_per_pixel must be > 0, got {metres_per_pixel}")
        d_ref = distance_to(ref) * metres_per_pixel  # metres from a pixel to the nearest reference pixel
        d_pred = distance_to(pred) * metres_per_pixel
        pred_to_ref, ref_to_pred = d_ref[pred], d_pred[ref]
        self.n_pred += pred_to_ref.size
        self.n_ref += ref_to_pred.size
        for i, radius in enumerate(self.radii):
            self.pred_within[i] += (pred_to_ref <= radius).sum()
            self.ref_within[i] += (ref_to_pred <= radius).sum()
            near_pred, near_ref = d_pred <= radius, d_ref <= radius
            self.inter[i] += (near_pred & near_ref).sum()
            self.union[i] += (near_pred | near_ref).sum()
        bins = np.minimum(pred_to_ref / HIST_BIN_M, len(self.hist) - 1).astype(int)
        self.hist += np.bincount(bins, minlength=len(self.hist))

    def summary(self) -> dict:
        """Scores at every radius (`r0.5`, `r1`, ...), plus the median offset of the predicted pixels within `HIST_MAX_M` of a reference line."""
        result: dict = {"pred_pixels": int(self.n_pred), "ref_pixels": int(self.n_ref)}
        for i, radius in enumerate(self.radii):
            precision = self.pred_within[i] / self.n_pred if self.n_pred else 0.0
            recall = self.ref_within[i] / self.n_ref if self.n_ref else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            false_pos, false_neg = self.n_pred - self.pred_within[i], self.n_ref - self.ref_within[i]
            denominator = self.ref_within[i] + false_pos + false_neg
            result[f"r{radius:g}"] = {
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "quality": round(float(self.ref_within[i] / denominator) if denominator else 0.0, 4),
                "buffered_iou": round(float(self.inter[i] / self.union[i]) if self.union[i] else 0.0, 4),
            }
        matched = self.hist[:-1]
        if matched.sum():
            cumulative = np.cumsum(matched)
            median_bin = int(np.searchsorted(cumulative, matched.sum() / 2))
            result["median_offset_m"] = round((median_bin + 0.5) * HIST_BIN_M, 2)
            result["share_within_5m"] = round(float(matched.sum() / self.n_pred), 4)
        return result

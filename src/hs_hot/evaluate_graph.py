"""Scoring road networks over many tiles with APLS and TOPO (numpy and scipy; `centreline` extra).

`GraphTally.add(reference, prediction, metres_per_pixel)` takes two one-pixel skeleton masks of one tile and `summary()` returns the pooled scores:

- **APLS** per tile (`hs_hot.apls`), averaged over the scored tiles, with its two parts (missing roads / false roads) and how many tiles were skipped;
- **TOPO** counts (`hs_hot.topo`) summed over tiles, then precision, recall and F1 of the totals.

**These are the HOTOSM road-network metrics, not the official CosmiQ scores.** They follow the CosmiQ code step by step (matched against it on real tiles, see
`hs_hot.apls` and `hs_hot.topo`) but differ from it in the ways listed in `DIFFERENCES_FROM_OFFICIAL`; every summary carries that list and the variant name
so a number is never quoted as an official APLS or TOPO score. `summary()["diagnostics"]` counts what the preprocessing did.

Before scoring, each tile goes through the steps in `GraphScoring`, in this order:

1. **Border margin** (`border_margin_m`, default 0 = off). A strip of this width is cut off all four sides of BOTH masks. Why: the OSM labels include
   roads that lie partly or wholly outside a tile, and a model that predicts a wide band along them paints into the tile where the reference (clipped
   to the tile) has no line; after thinning that is a false road. A perfect model scored this way loses about 33 APLS points at a 6 m band, about 7 at
   2 m, about 2 at 0.9 m (`research/analysis/thinning_check.py`, 400 validation tiles), so without a margin the score depends on the target width and not
   only on the model. The strip is `ceil(border_margin_m / metres_per_pixel)` pixels. Cropping the same region for every model keeps scores comparable;
   it also hides the roads in that strip for everyone (an interior score, not the whole tile).
2. **Small fragments** (`ref_min_fragment_m`, `pred_min_fragment_m`, default 5 m each; 0 = off): whole disconnected pieces shorter than this are dropped
   from that mask, as the official SpaceNet scoring does when it loads a map (it uses 5 m for the reference and 10 pixels for the prediction).
3. **Graphs**: `road_graph.skeleton_to_graph` for both masks.

The default lengths are for tiles about 76 m wide (see `hs_hot.apls` and `hs_hot.topo`); the official SpaceNet values are in those docstrings.
`n_measurement_nodes=None` measures TOPO from every reference node once; an integer draws that many with replacement, as the official code does.
`crop_border` and `margin_pixels` live in `hs_hot.evaluate_centreline` (one rule for the line score and the graph scores) and are re-exported here.
Not exported from `hs_hot` (import `hs_hot.evaluate_graph`).
"""

from dataclasses import dataclass

import numpy as np

from . import topo
from .apls import MIDPOINT_M, MIN_PATH_M, SNAP_M, apls_graphs
from .evaluate_centreline import crop_border, margin_pixels
from .road_graph import drop_small_fragments, skeleton_to_graph

METRIC_VARIANT = "HOTOSM road-network metrics (APLS and TOPO after the CosmiQ code, with the differences listed in DIFFERENCES_FROM_OFFICIAL)"
DIFFERENCES_FROM_OFFICIAL = (
    "TOPO: a reference with an empty prediction counts every hole as missed (official: no counts, so predicting nothing is never punished).",
    "TOPO: every reference node is measured once (official: a random draw with replacement).",
    "APLS: a tile whose reference has no usable pair of control points is left out of the mean when skip_unscorable (official: scored 0).",
    "APLS: control points that land on the same spot of the other map all count as snapped (official loses all but the last).",
    "Fragments: a closed ring is kept if it is long enough (official deletes every one-node ring).",
    "Lengths (midpoints, TOPO radius and spacing, hole size, fragment size) and the border margin are chosen for ~76 m tiles, not 400 m SpaceNet chips.",
)


@dataclass(frozen=True)
class GraphScoring:
    snap_m: float = SNAP_M
    midpoint_m: float = MIDPOINT_M
    min_path_m: float = MIN_PATH_M
    curved_eps: float = -1.0
    skip_unscorable: bool = True
    topo_radius_m: float = topo.RADIUS_M
    topo_interval_m: float = topo.INTERVAL_M
    topo_hole_m: float = topo.HOLE_M
    n_measurement_nodes: int | None = None
    border_margin_m: float = 0.0
    ref_min_fragment_m: float = 5.0
    pred_min_fragment_m: float = 5.0


__all__ = [
    "DIFFERENCES_FROM_OFFICIAL",
    "METRIC_VARIANT",
    "GraphScoring",
    "GraphTally",
    "crop_border",
    "margin_pixels",
]


class GraphTally:
    """Pools APLS and TOPO over tiles. `summary()["diagnostics"]` counts what the preprocessing did, so a surprising score can be traced:
    fragments removed per side, tiles skipped by APLS, skeleton pixels no walk reached, and walks that ended on no node (all 0 on clean skeletons)."""

    def __init__(self, scoring: GraphScoring | None = None) -> None:
        self.scoring = scoring or GraphScoring()
        self.apls: list[float] = []
        self.part1: list[float] = []
        self.part2: list[float] = []
        self.skipped = 0
        self.topo_counts = np.zeros(3)
        self.fragments_removed = {"reference": 0, "prediction": 0}
        self.untraced_pixels = {"reference": 0, "prediction": 0}
        self.dead_ends = {"reference": 0, "prediction": 0}

    def add(
        self,
        reference: np.ndarray,
        prediction: np.ndarray,
        metres_per_pixel: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Score one tile: two boolean one-pixel skeleton masks of the same shape."""
        s = self.scoring
        if reference.shape != prediction.shape:
            raise ValueError(f"prediction {prediction.shape} and reference {reference.shape} differ in shape")
        margin = margin_pixels(
            s.border_margin_m, metres_per_pixel
        )  # raises on a bad pixel size or margin; crop_border raises if nothing is left
        ref, n_ref_removed = drop_small_fragments(
            crop_border(reference, margin), metres_per_pixel, s.ref_min_fragment_m
        )
        pred, n_pred_removed = drop_small_fragments(
            crop_border(prediction, margin), metres_per_pixel, s.pred_min_fragment_m
        )
        ref_graph, pred_graph = (
            skeleton_to_graph(ref, metres_per_pixel),
            skeleton_to_graph(pred, metres_per_pixel),
        )
        for side, graph, removed in (
            ("reference", ref_graph, n_ref_removed),
            ("prediction", pred_graph, n_pred_removed),
        ):
            self.fragments_removed[side] += removed
            self.untraced_pixels[side] += graph.untraced
            self.dead_ends[side] += graph.dead_ends
        result = apls_graphs(
            ref_graph, pred_graph, s.snap_m, s.midpoint_m, s.min_path_m, s.curved_eps, s.skip_unscorable
        )
        if result is None:
            self.skipped += 1
        else:
            for store, value in zip((self.apls, self.part1, self.part2), result, strict=True):
                store.append(value)
        seeds = topo.draw_seeds(ref_graph, s.n_measurement_nodes, rng)
        self.topo_counts += topo.topo_counts(
            ref_graph, pred_graph, seeds, s.topo_radius_m, s.topo_interval_m, s.topo_hole_m
        )

    def summary(self) -> dict:
        def mean(values: list[float]) -> float | None:
            return round(float(np.mean(values)), 4) if values else None

        return {
            "apls": mean(self.apls),
            "apls_part1_reference_onto_prediction": mean(self.part1),
            "apls_part2_prediction_onto_reference": mean(self.part2),
            "chips_scored": len(self.apls),
            "chips_skipped": self.skipped,
            "topo": topo.summarise(self.topo_counts),
            "diagnostics": {
                "fragments_removed": dict(self.fragments_removed),
                "apls_tiles_skipped": self.skipped,
                "skeleton_pixels_untraced": dict(self.untraced_pixels),
                "dead_end_walks": dict(self.dead_ends),
            },
            "metric_variant": METRIC_VARIANT,
        }

"""TOPO, the road-network "marbles and holes" score, written to follow the official CosmiQ implementation (numpy and scipy; `centreline` extra).

TOPO asks whether the predicted network holds the same *places* as the reference network, locally around many starting points. For one
measurement node (a node of the reference graph):

1. **Reference sub-graph.** The nodes within `radius_m` (straight-line distance; at most the 500 nearest, as in the official kd-tree query) of the
   node, the edges between them, and then only the connected piece that contains the node. An edge whose far end lies outside the radius is
   dropped whole, so the radius must be longer than the typical edge or the sub-graph is only the node itself.
2. **Holes.** Every node of that sub-graph, plus a point every `interval_m` along each edge (an edge shorter than `interval_m` gets none),
   counted from the end of the edge that lies closer to the measurement node.
3. **Prediction origin.** The prediction node nearest to the measurement node, if it is within `hole_m`. Its sub-graph and its points (the
   "marbles") are built the same way.
4. **Matching, one marble per hole.** Each marble in turn (nodes in index order, then edge points in edge order) takes the nearest hole within
   `hole_m` that no earlier marble has taken. A marble that finds none is a false positive; a hole nobody took is a false negative.

Counts are summed over the measurement nodes and over tiles. `precision = TP / (TP + FP)`, `recall = TP / (TP + FN)`, F1 from those (0 when a
denominator is 0; see `summarise`).

**Values.** The official function defaults are radius 150 m, interval 30 m, hole 5 m and 20 measurement nodes (its command line: hole 4 m and as many
nodes as there are). They were set for 400 m SpaceNet chips. The defaults here are for tiles about 76 m wide whose reference graphs have a median edge of
31 m (75th percentile 54 m, measured on 400 validation tiles, `research/analysis/tile_scale_stats.py`): radius 60 m (longer than most edges), interval
10 m (about 2.5 hole radii, 2-3 holes on a median edge), hole 4 m (the APLS snap distance), and every node measured once. They are choices, not a standard:
every one is a parameter and the official values reproduce the official counts (checked on real tiles, `research/analysis/official_check_compare.py`).

**Corner cases** (each has a test in `tests/test_topo.py`):

- no reference nodes: nothing is measured and the tile adds no counts (official);
- a reference but an empty prediction, or no prediction node within `hole_m` of the measurement node: every hole is a false negative and nothing
  is a false positive (the official code does this for the second case; for an empty prediction its `compute_topo` returns zeros, which would let a
  model that predicts nothing escape the recall penalty, so it is counted as missed here: a deliberate difference);
- a prediction on a tile with no reference roads adds nothing, because TOPO cannot see false roads where the reference has no node (official
  behaviour and a limitation: pair TOPO with APLS or a line score for that);
- one marble takes one hole (the official `allow_multi_hole=False`; its multi-hole option can give a recall above 1 and is not offered);
- a measurement node drawn twice is measured twice (official; with `n_measurement_nodes=None` every node is measured exactly once instead).

Not copied: the order in which the official code visits marbles with equal claims depends on networkx's set ordering; here it is fixed.
This is the HOTOSM variant of the metric: where it differs from the official code, `evaluate_graph.DIFFERENCES_FROM_OFFICIAL` says how and why.
Not exported from `hs_hot` (import `hs_hot.topo`).
"""

import numpy as np
from scipy.spatial import cKDTree

from .road_graph import RoadGraph, points_along

RADIUS_M = 60.0
INTERVAL_M = 10.0
HOLE_M = 4.0
KD_NEIGHBOURS = 500  # the official kd-tree ball query asks for at most 500 nodes


def _component(graph: RoadGraph, origin: int, radius_m: float) -> tuple[list[int], list[int]]:
    """Node ids and edge ids of the connected piece around `origin`, made of the nodes within `radius_m` and the edges between them."""
    k = min(KD_NEIGHBOURS, len(graph))
    dist, idx = graph.tree.query(graph.xy[origin], k=k, distance_upper_bound=radius_m)
    dist, idx = np.atleast_1d(dist), np.atleast_1d(idx)
    inside = {int(i) for d, i in zip(dist, idx, strict=True) if np.isfinite(d) and d <= radius_m}
    seen, stack = {origin}, [origin]
    while stack:
        u = stack.pop()
        for _, other in graph.adjacent[u]:
            if other in inside and other not in seen:
                seen.add(other)
                stack.append(other)
    edges = [i for i, (u, v, _, _) in enumerate(graph.edges) if u in seen and v in seen]
    return sorted(seen), edges


def points_around(
    graph: RoadGraph, origin: int, radius_m: float = RADIUS_M, interval_m: float = INTERVAL_M
) -> np.ndarray:
    """The holes (or marbles) of the sub-graph around node `origin`: its nodes, then a point every `interval_m` along each of its edges."""
    nodes, edge_ids = _component(graph, origin, radius_m)
    out = [graph.xy[nodes]]
    here = graph.xy[origin]
    for i in edge_ids:
        _, _, coords, length = graph.edges[i]
        if length < interval_m:
            continue
        if np.hypot(*(coords[-1] - here)) < np.hypot(
            *(coords[0] - here)
        ):  # points run away from the measurement node
            coords = coords[::-1]
        steps = np.arange(0, length, interval_m)[1:]
        if len(steps):
            out.append(points_along(coords, steps))
    return np.vstack(out)


def match(holes: np.ndarray, marbles: np.ndarray, hole_m: float) -> tuple[int, int, int]:
    """`(true positives, false positives, false negatives)` with one marble per hole."""
    tree, taken, tp = cKDTree(holes), np.zeros(len(holes), dtype=bool), 0
    for marble in marbles:
        free = [i for i in tree.query_ball_point(marble, r=hole_m) if not taken[i]]
        if free:
            taken[min(free, key=lambda i: float(np.hypot(*(holes[i] - marble))))] = True
            tp += 1
    return tp, len(marbles) - tp, len(holes) - tp


def draw_seeds(
    ref: RoadGraph, n_measurement_nodes: int | None = None, rng: np.random.Generator | None = None
) -> np.ndarray:
    """The reference nodes to measure from.

    `None`: every node once (deterministic). An integer: `min(n, node count)` nodes drawn with replacement, as the official `np.random.choice` does.
    """
    if len(ref) == 0:
        return np.zeros(0, dtype=int)
    if n_measurement_nodes is None:
        return np.arange(len(ref))
    rng = rng if rng is not None else np.random.default_rng(0)
    return rng.choice(len(ref), min(n_measurement_nodes, len(ref)))


def topo_counts(
    ref: RoadGraph,
    pred: RoadGraph,
    seeds,
    radius_m: float = RADIUS_M,
    interval_m: float = INTERVAL_M,
    hole_m: float = HOLE_M,
) -> np.ndarray:
    """`[TP, FP, FN]` summed over the reference nodes `seeds` (indices into `ref`)."""
    total = np.zeros(3)
    if len(ref) == 0:
        return total
    for seed in seeds:
        holes = points_around(ref, int(seed), radius_m, interval_m)
        origin = -1
        if len(pred):
            gap, nearest = pred.tree.query(ref.xy[int(seed)])
            origin = int(nearest) if gap <= hole_m else -1
        if origin < 0:  # no prediction at the measurement node: every hole is missed
            total += (0, 0, len(holes))
            continue
        total += match(holes, points_around(pred, origin, radius_m, interval_m), hole_m)
    return total


def summarise(counts) -> dict:
    """Precision, recall and F1 from pooled `[TP, FP, FN]` (0 where a denominator is 0), with the counts."""
    tp, fp, fn = (float(c) for c in counts)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }

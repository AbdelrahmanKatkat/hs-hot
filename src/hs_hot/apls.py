"""APLS, the Average Path Length Similarity of road networks, following the official CosmiQ code step by step (numpy and scipy; `centreline` extra).

APLS asks whether travelling along the predicted network takes you the same distance as along the reference. It works on two road graphs
(`road_graph.skeleton_to_graph`):

1. **Control points** of a graph: all its nodes, plus midpoints on its edges. An edge shorter than `0.75 * midpoint_m` gets none, an edge up
   to `midpoint_m` one in the middle, a longer edge is split into `len(arange(0, L, midpoint_m))` equal parts. With `curved_eps >= 0` only edges
   whose bounding-box diagonal differs from their length by at least that fraction get midpoints (official function default 0.012; the official
   command line uses -1, every edge, as here).
2. **Snapping.** The control points of the graph being checked are snapped onto the other graph: each goes to the nearest point of any edge of
   the other graph if that is within `snap_m`, otherwise it is missing.
3. **Pairs.** For every *ordered* pair (start, end) of control points that are connected in the checked graph:
   start missing in the other graph: penalty 1 whatever the length (an official quirk: there is no minimum length on this branch);
   path shorter than `min_path_m`: skipped; end missing, or no path in the other graph: penalty 1;
   otherwise `min(1, |L - L'| / L)`, with `L` the path length in the checked graph and `L'` in the other one.
   `C = 1 - mean(penalty)`; with no pair at all `C = 0`.
4. **Both directions.** `C1` checks the reference against the prediction and `C2` the prediction against the reference. `APLS` is their
   harmonic mean, 0 if either is 0. The score of a split is the mean of the tile scores.

Part 1 (`C1`) is the missing-road side: a reference road the prediction does not have, or has broken, or has detoured, costs here. Part 2 (`C2`)
is the false-road side: a predicted road with no reference nearby costs here.

**Values.** The official function defaults are snap 4 m, midpoints every 50 m and a minimum path of 10 m (the command line uses 0.001). They were set
for 400 m SpaceNet chips. The defaults here are for tiles about 76 m wide: snap 4 m (a ground distance for label and imagery misalignment, not a
tile size), midpoints every 10 m (edges there have a median length of 31 m, so 50 m would put no midpoint on more than half of them) and a minimum
path of 10 m (a pair closer than about 2.5 snap distances is mostly snapping noise). The official values reproduce the official scores: on 50 real
tiles the mean difference was 0.000 to 0.005 (`research/analysis/official_check_compare.py`). Every value is a parameter.

**Corner cases** (each has a test in `tests/test_apls.py`):

- an empty prediction against a reference scores 0, and so does an empty reference against a prediction (false roads cost everything);
- a control point farther than `snap_m` from the other graph is a full penalty;
- both graphs empty, or no pair of control points long enough: the official code scores 0. With `skip_unscorable=True` (the default here) such a tile
  returns None and is left out of the average instead. In the validation tiles 29% have no road at all, and a correct empty prediction should not
  score 0 there. `skip_unscorable=False` gives the official behaviour;
- a prediction with no usable pair while the reference has one scores 0 on that side;
- two control points that land on the very same spot of the other graph (for example both snap onto the end of a short stub) both count as snapped.
  The official code renames the node the first one created when the second arrives, so the first one vanishes and is scored as missing although it is
  within a metre or two of the road. That is an artefact of its node renaming (it also depends on the order of the node ids), it is not copied, and it is
  the cause of the largest differences from the official score on real tiles (up to 0.12 on a tile with 6 to 8 control points);
- travel-time APLS is not implemented: it needs a speed for every road, and these labels have none.

This is the HOTOSM variant of the metric: where it differs from the official code, `evaluate_graph.DIFFERENCES_FROM_OFFICIAL` says how and why.
Not exported from `hs_hot` (import `hs_hot.apls`).
"""

import itertools

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .road_graph import RoadGraph, points_along, skeleton_to_graph

SNAP_M = 4.0
MIDPOINT_M = 10.0
MIN_PATH_M = 10.0


def _edge_midpoints(length: float, midpoint_m: float) -> np.ndarray:
    if length < 0.75 * midpoint_m:
        return np.zeros(0)
    if length <= midpoint_m:
        return np.array([0.5 * length])
    return np.linspace(0, length, len(np.arange(0, length, midpoint_m)) + 1)[1:-1]


def control_points(
    graph: RoadGraph, midpoint_m: float = MIDPOINT_M, curved_eps: float = -1.0
) -> tuple[list[int], list[tuple[int, float]]]:
    """`(node indices, midpoints)`; a midpoint is `(edge index, arc length from the start of the edge)`."""
    edge_points: list[tuple[int, float]] = []
    for i, (_, _, coords, length) in enumerate(graph.edges):
        if length <= 0:
            continue
        if curved_eps >= 0:
            span = coords.max(axis=0) - coords.min(axis=0)
            if abs(float(np.hypot(*span)) - length) / length < curved_eps:
                continue
        edge_points += [(i, float(s)) for s in _edge_midpoints(length, midpoint_m)]
    return list(range(len(graph))), edge_points


def point_xy(graph: RoadGraph, edge: int, s: float) -> np.ndarray:
    """The `(x, y)` position of the point at arc length `s` along edge `edge`."""
    return points_along(graph.edges[edge][2], np.array([s]))[0]


class _EdgeIndex:
    """Nearest point on the edges of a graph, over all their segments at once."""

    def __init__(self, graph: RoadGraph) -> None:
        a, b, edge, start = [], [], [], []
        for i, (_, _, coords, _) in enumerate(graph.edges):
            seg = np.hypot(*np.diff(coords, axis=0).T)
            a.append(coords[:-1])
            b.append(coords[1:])
            edge.append(np.full(len(seg), i))
            start.append(np.concatenate([[0.0], np.cumsum(seg)[:-1]]))
        self.empty = not a
        if not self.empty:
            self.a, self.b = np.vstack(a), np.vstack(b)
            self.edge, self.start = np.concatenate(edge), np.concatenate(start)
            self.d = self.b - self.a
            self.len2 = (self.d**2).sum(axis=1)

    def nearest(self, xy: np.ndarray) -> tuple[float, int, float]:
        """`(distance, edge, arc length along the edge)` of the nearest point on any edge; ties go to the first edge."""
        if self.empty:
            return np.inf, -1, 0.0
        safe = np.where(self.len2 > 0, self.len2, 1.0)
        t = np.where(self.len2 > 0, ((xy - self.a) * self.d).sum(axis=1) / safe, 0.0).clip(0, 1)
        dist = np.hypot(*((self.a + t[:, None] * self.d) - xy).T)
        k = int(np.argmin(dist))
        return float(dist[k]), int(self.edge[k]), float(self.start[k] + t[k] * np.sqrt(self.len2[k]))


def _augmented(graph: RoadGraph, edge_points: list[tuple[int, float]]):
    """Sparse graph of `graph` with a node inserted at every `(edge, s)`; the inserted nodes get ids `len(graph) + k`.

    Two roads between the same pair of nodes (parallel edges), and a closed ring cut at a single inserted point (two pieces between the same two nodes),
    give several pieces for one pair of nodes. A sparse matrix would add their lengths up, so only the shortest piece of each pair is kept; a piece that
    starts and ends at the same node (a ring with nothing inserted) is dropped, it can never shorten a path.
    """
    n = len(graph)
    per_edge: dict[int, list[tuple[float, int]]] = {}
    for k, (e, s) in enumerate(edge_points):
        per_edge.setdefault(e, []).append((s, n + k))
    shortest: dict[tuple[int, int], float] = {}
    for i, (u, v, _, length) in enumerate(graph.edges):
        chain = [(0.0, u), *sorted(per_edge.get(i, [])), (length, v)]
        for (s0, a), (s1, b) in itertools.pairwise(chain):
            if a != b:
                key = (min(a, b), max(a, b))
                shortest[key] = min(
                    shortest.get(key, np.inf), max(s1 - s0, 0.0) + 1e-9
                )  # an explicit zero would not count as an edge
    size = n + len(edge_points)
    src, dst = (
        (np.array(side, dtype=int) for side in zip(*shortest, strict=True))
        if shortest
        else (np.zeros(0, int), np.zeros(0, int))
    )
    half = coo_matrix(
        (np.fromiter(shortest.values(), dtype=float, count=len(shortest)), (src, dst)), shape=(size, size)
    )
    return (half + half.T).tocsr()


def _side(
    checked: RoadGraph,
    other: RoadGraph,
    snap_m: float,
    midpoint_m: float,
    min_path_m: float,
    curved_eps: float,
) -> float | None:
    """`C` of the official `path_sim_metric` for the control points of `checked` snapped onto `other`; None when there is no pair at all."""
    nodes, edge_points = control_points(checked, midpoint_m, curved_eps)
    xy = [checked.xy[v] for v in nodes] + [point_xy(checked, e, s) for e, s in edge_points]
    m = len(xy)
    if m < 2:
        return None
    control = np.array(nodes + [len(checked) + k for k in range(len(edge_points))])
    d_checked = dijkstra(_augmented(checked, edge_points), directed=False, indices=control)[:, control]
    index = _EdgeIndex(other)
    snapped_at: list[tuple[int, float] | None] = []
    for point in xy:
        dist, e, s = index.nearest(np.asarray(point))
        snapped_at.append((e, s) if dist <= snap_m and e >= 0 else None)
    snapped = np.array([p is not None for p in snapped_at])
    d_other = np.full((m, m), np.inf)
    ok = np.flatnonzero(snapped)
    if len(ok):
        ids = np.array([len(other) + k for k in range(len(ok))])
        matrix = _augmented(other, [snapped_at[i] for i in ok])
        d_other[np.ix_(ok, ok)] = dijkstra(matrix, directed=False, indices=ids)[:, ids]
    penalties: list[float] = []
    for s in range(m):
        connected = np.isfinite(d_checked[s]) & (np.arange(m) != s)
        if not snapped[s]:
            penalties += [1.0] * int(connected.sum())  # official: counted whatever the length
            continue
        for e in np.flatnonzero(connected):
            length = d_checked[s, e]
            if length < min_path_m or length <= 0:
                continue
            other_length = d_other[s, e]
            missing = not (snapped[e] and np.isfinite(other_length))
            penalties.append(1.0 if missing else min(1.0, abs(length - other_length) / length))
    return float(1.0 - np.mean(penalties)) if penalties else None


def apls_graphs(
    ref: RoadGraph,
    pred: RoadGraph,
    snap_m: float = SNAP_M,
    midpoint_m: float = MIDPOINT_M,
    min_path_m: float = MIN_PATH_M,
    curved_eps: float = -1.0,
    skip_unscorable: bool = True,
) -> tuple[float, float, float] | None:
    """`(apls, part1, part2)` for one tile from two road graphs, or None for an unscorable tile when `skip_unscorable` (see the module docstring)."""
    unscorable = None if skip_unscorable else (0.0, 0.0, 0.0)
    if len(ref) == 0 and len(pred) == 0:
        return unscorable
    if len(ref) == 0 or len(pred) == 0:
        return 0.0, 0.0, 0.0
    part1 = _side(ref, pred, snap_m, midpoint_m, min_path_m, curved_eps)
    if part1 is None:  # the reference has no pair of control points long enough: nothing to compare
        return unscorable
    part2 = (
        _side(pred, ref, snap_m, midpoint_m, min_path_m, curved_eps) or 0.0
    )  # None: the prediction has no usable pair
    if part1 <= 0 or part2 <= 0:
        return 0.0, part1, part2
    return 2 * part1 * part2 / (part1 + part2), part1, part2


def apls(
    ref_mask: np.ndarray,
    pred_mask: np.ndarray,
    metres_per_pixel: float,
    snap_m: float = SNAP_M,
    midpoint_m: float = MIDPOINT_M,
    min_path_m: float = MIN_PATH_M,
    curved_eps: float = -1.0,
    skip_unscorable: bool = True,
) -> tuple[float, float, float] | None:
    """`apls_graphs` for two one-pixel skeleton masks (reference, prediction)."""
    ref = skeleton_to_graph(ref_mask, metres_per_pixel)
    pred = skeleton_to_graph(pred_mask, metres_per_pixel)
    return apls_graphs(ref, pred, snap_m, midpoint_m, min_path_m, curved_eps, skip_unscorable)

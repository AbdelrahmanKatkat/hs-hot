"""A road map as a graph: nodes at road ends and junctions, one edge per road piece between them (numpy and scipy; `centreline` extra).

APLS and TOPO compare road *networks*, so a thinned prediction (a one-pixel mask, for example `centreline.thin_to_centreline`) first has to become
a graph. `skeleton_to_graph` does that:

- **Nodes**: every end pixel (one 8-neighbour), every isolated pixel, and one node per cluster of junction pixels (three or more branches),
  placed at the cluster centroid.
- **Edges**: the run of line pixels between two nodes, kept as a polyline through the pixel centres. The edge length is the sum of the
  segment lengths (1 or sqrt 2 pixels times `metres_per_pixel`), so a diagonal road is measured like the pixel path it is drawn as.
- **Coordinates** are metres: `x = column * metres_per_pixel`, `y = row * metres_per_pixel`.

Corner cases handled (each has a test in `tests/test_road_graph.py`):

- the branches that meet at a junction touch each other diagonally: they are walked one by one and never merged into one road;
- a staircase or square corner with a redundant pixel is walked once, the extra pixel is not a second road;
- two nodes next to each other (a two-pixel road, an end pixel beside a junction) get a direct edge;
- a closed ring with no end or junction gets one node on it and a loop edge;
- an isolated pixel is a node with no edge;
- a walk that reaches no node (should not happen on a clean skeleton) ends in a new end node and is counted in `dead_ends`;
- skeleton pixels that no walk reached are counted in `untraced`, never dropped silently.

`remove_small_fragments` drops whole disconnected pieces that are too short to be a road, as the official SpaceNet scoring does before it
scores a map. Not exported from `hs_hot` (import `hs_hot.road_graph`).
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
EIGHT = np.ones((3, 3), dtype=int)


def crossings(mask: np.ndarray) -> np.ndarray:
    """Per pixel, the number of separate runs of set pixels in its 8-ring: 1 = end, 2 = along a line, 3 or more = junction."""
    p = np.pad(mask, 1)
    ring = [
        p[:-2, :-2],
        p[:-2, 1:-1],
        p[:-2, 2:],
        p[1:-1, 2:],
        p[2:, 2:],
        p[2:, 1:-1],
        p[2:, :-2],
        p[1:-1, :-2],
    ]
    return sum((~ring[i]) & ring[(i + 1) % 8] for i in range(8))


def points_along(coords: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Points at the given distances along a polyline of `(x, y)` rows (what shapely's `interpolate` does); distances beyond the end clamp to it."""
    cumulative = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(coords, axis=0).T))])
    return np.column_stack(
        [np.interp(distances, cumulative, coords[:, 0]), np.interp(distances, cumulative, coords[:, 1])]
    )


class RoadGraph:
    """`xy`: `(n, 2)` node positions in metres. `edges`: list of `(u, v, coords, length)`, `coords` an `(k, 2)` polyline from node u to node v.

    `anchors` is a skeleton pixel `(row, col)` that belongs to each node. `adjacent[u]` lists `(edge index, other node)`; `tree` is a kd-tree of the
    nodes (None for an empty graph). `untraced` and `dead_ends` count the skeleton pixels and walks that did not fit the graph (0 on a clean skeleton).
    """

    def __init__(self, xy, edges, untraced: int = 0, dead_ends: int = 0, anchors=None) -> None:
        self.xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        self.anchors = np.asarray(anchors if anchors is not None else [], dtype=int).reshape(-1, 2)
        self.edges = [
            (int(u), int(v), np.asarray(c, dtype=float), float(np.hypot(*np.diff(c, axis=0).T).sum()))
            for u, v, c in edges
        ]
        self.untraced, self.dead_ends = untraced, dead_ends
        self.adjacent: list[list[tuple[int, int]]] = [[] for _ in range(len(self.xy))]
        for i, (u, v, _, _) in enumerate(self.edges):
            self.adjacent[u].append((i, v))
            if v != u:
                self.adjacent[v].append((i, u))
        self.tree = cKDTree(self.xy) if len(self.xy) else None

    def __len__(self) -> int:
        return len(self.xy)


class _Walker:
    """Walks the pixels of a skeleton from node to node. Kept as a class so `skeleton_to_graph` reads as the steps it takes."""

    def __init__(self, mask: np.ndarray, metres_per_pixel: float) -> None:
        self.mask, self.mpp = mask, float(metres_per_pixel)
        self.height, self.width = mask.shape
        cross = crossings(mask)
        self.node_id = np.full(mask.shape, -1, dtype=np.int64)
        self.xy: list[list[float]] = []
        self.anchors: list[tuple[int, int]] = []
        labels, count = ndimage.label(
            mask & (cross >= 3), structure=EIGHT
        )  # one node per cluster of junction pixels
        for k in range(1, count + 1):
            pixels = np.argwhere(labels == k)
            centre = pixels.mean(axis=0)
            self.node_id[labels == k] = len(self.xy)
            self.anchors.append(
                tuple(int(v) for v in pixels[np.argmin(((pixels - centre) ** 2).sum(axis=1))])
            )
            self.xy.append([centre[1] * self.mpp, centre[0] * self.mpp])
        for r, c in np.argwhere(mask & (cross <= 1)):  # ends and isolated pixels
            self.node_id[r, c] = len(self.xy)
            self.anchors.append((int(r), int(c)))
            self.xy.append([c * self.mpp, r * self.mpp])
        self.path = mask & (self.node_id < 0)
        self.visited = np.zeros(mask.shape, dtype=bool)
        self.edges: list[tuple[int, int, np.ndarray]] = []
        self.direct: set[tuple[int, int]] = set()
        self.dead_ends = 0

    def pixel_xy(self, p: tuple[int, int]) -> list[float]:
        return [p[1] * self.mpp, p[0] * self.mpp]

    def neighbours(self, r: int, c: int):
        for dr, dc in OFFSETS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.height and 0 <= cc < self.width and self.mask[rr, cc]:
                yield rr, cc

    def trace(self, start: tuple[int, int], first: tuple[int, int]) -> None:
        """Walk from node pixel `start` through `first` along the road until another node (or `start` again for a ring)."""
        u = int(self.node_id[start])
        chain, prev, cur = [first], start, first
        self.visited[first] = True
        while True:
            ends, forward = [], []
            for q in self.neighbours(*cur):
                if q == prev:
                    continue
                nid = int(self.node_id[q])
                if nid >= 0:
                    if nid != u or len(chain) >= 3:  # a ring needs 3 pixels to come back to its own node
                        ends.append(q)
                elif not self.visited[q]:
                    forward.append(q)
            if ends:
                nearest = min(ends, key=lambda q: (q[0] - cur[0]) ** 2 + (q[1] - cur[1]) ** 2)
                v = int(self.node_id[nearest])
                break
            if not forward:  # no node ahead: end the road here with a new end node
                self.dead_ends += 1
                v = len(self.xy)
                self.xy.append(self.pixel_xy(cur))
                self.anchors.append(cur)
                break
            # The pixel farthest from the one we came from: at a square corner this takes the diagonal shortcut and
            # leaves the corner pixel as the redundant one, which is marked visited so it is not read as another road.
            nxt = max(forward, key=lambda q: (q[0] - prev[0]) ** 2 + (q[1] - prev[1]) ** 2)
            for q in forward:
                if q != nxt and max(abs(q[0] - nxt[0]), abs(q[1] - nxt[1])) <= 1:
                    self.visited[q] = True
            self.visited[nxt] = True
            chain.append(nxt)
            prev, cur = cur, nxt
        coords = [self.xy[u]] + [self.pixel_xy(p) for p in chain] + [self.xy[v]]
        self.edges.append((u, v, np.array(coords)))

    def walk_from(self, p: tuple[int, int]) -> None:
        for q in self.neighbours(*p):
            if self.node_id[q] >= 0:  # two node pixels side by side: a direct edge, once per pair of nodes
                a, b = int(self.node_id[p]), int(self.node_id[q])
                if a != b and (min(a, b), max(a, b)) not in self.direct:
                    self.direct.add((min(a, b), max(a, b)))
                    self.edges.append((a, b, np.array([self.xy[a], self.xy[b]])))
            elif not self.visited[q]:
                self.trace(p, q)

    def close_rings(self) -> int:
        """Handle what the walks left: closed rings (no end, no junction: one node and a loop edge) and redundant pixels. Returns the untraced count."""
        rest = self.path & ~self.visited
        near_done = ndimage.binary_dilation((self.node_id >= 0) | self.visited, structure=EIGHT)
        pieces, count = ndimage.label(rest, structure=EIGHT)
        untraced = 0
        for k in range(1, count + 1):
            piece = pieces == k
            pixels = np.argwhere(piece)
            if near_done[piece].any():  # beside a road that was walked: redundant pixels, not a separate road
                untraced += len(pixels)
                continue
            start = (int(pixels[0, 0]), int(pixels[0, 1]))
            self.node_id[start] = len(self.xy)
            self.xy.append(self.pixel_xy(start))
            self.anchors.append(start)
            self.visited[start] = True
            self.walk_from(start)
            untraced += int(piece.sum() - self.visited[piece].sum())
        return untraced


def skeleton_to_graph(mask: np.ndarray, metres_per_pixel: float) -> RoadGraph:
    """The road graph of a one-pixel skeleton mask (see the module docstring)."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return RoadGraph([], [])
    walker = _Walker(mask, metres_per_pixel)
    for p in np.argwhere(walker.node_id >= 0):
        walker.walk_from((int(p[0]), int(p[1])))
    untraced = walker.close_rings()
    return RoadGraph(
        walker.xy, walker.edges, untraced=untraced, dead_ends=walker.dead_ends, anchors=walker.anchors
    )


def drop_small_fragments(
    mask: np.ndarray, metres_per_pixel: float, min_length_m: float, max_nodes_to_skip: int = 100
) -> tuple[np.ndarray, int]:
    """`(mask without the small pieces, number of pieces removed)`; see `remove_small_fragments`."""
    mask = np.asarray(mask, dtype=bool)
    if min_length_m <= 0 or not mask.any():
        return mask, 0
    graph = skeleton_to_graph(mask, metres_per_pixel)
    parent = list(range(len(graph)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v, _, _ in graph.edges:
        parent[find(u)] = find(v)
    groups: dict[int, list[int]] = {}
    for node in range(len(graph)):
        groups.setdefault(find(node), []).append(node)
    labels = ndimage.label(mask, structure=EIGHT)[0]
    short: dict[int, list[bool]] = {}  # per connected piece of the mask: is each graph component in it short?
    for nodes in groups.values():
        r, c = graph.anchors[nodes[0]]
        if len(nodes) > max_nodes_to_skip:
            short.setdefault(int(labels[r, c]), []).append(False)
            continue
        index = {node: i for i, node in enumerate(nodes)}
        dist = np.full((len(nodes), len(nodes)), np.inf)
        np.fill_diagonal(dist, 0.0)
        longest_edge = 0.0
        for u, v, _, length in graph.edges:
            if u in index and v in index:
                a, b = index[u], index[v]
                if a != b:
                    dist[a, b] = dist[b, a] = min(dist[a, b], length)
                longest_edge = max(
                    longest_edge, length
                )  # a ring has one node and no pair of nodes: its own length counts
        for k in range(len(nodes)):  # Floyd-Warshall, at most `max_nodes_to_skip` nodes
            dist = np.minimum(dist, dist[:, k : k + 1] + dist[k : k + 1, :])
        longest = max(float(dist[np.isfinite(dist)].max()), longest_edge)
        short.setdefault(int(labels[r, c]), []).append(longest < min_length_m)
    out, removed = mask.copy(), 0
    for label, flags in short.items():
        if all(flags):  # remove a piece of the mask only when everything the graph found in it is short
            out[labels == label] = False
            removed += 1
    return out, removed


def remove_small_fragments(
    mask: np.ndarray, metres_per_pixel: float, min_length_m: float, max_nodes_to_skip: int = 100
) -> np.ndarray:
    """Drop whole disconnected pieces that are shorter than `min_length_m` (after the official `_clean_sub_graphs`).

    A piece's length is the longest path between two of its nodes, or the length of its longest road if that is more: the official code uses only the
    first, which is 0 for a closed ring (one node, no pair), so it would delete every roundabout whatever its size; that is not copied. A piece with more
    than `max_nodes_to_skip` nodes is kept without measuring, as in the official code; an isolated pixel has length 0 and goes. The official SpaceNet
    scoring applies this when it loads the maps: 5 m on the reference, 10 pixels on the prediction. `min_length_m <= 0` switches it off.
    """
    return drop_small_fragments(mask, metres_per_pixel, min_length_m, max_nodes_to_skip)[0]

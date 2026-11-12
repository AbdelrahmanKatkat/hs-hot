"""APLS on hand-built roads with answers worked out by hand; the official-code values were checked in research/analysis/apls_official_check.py."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from hs_hot.apls import apls, control_points, point_xy
from hs_hot.road_graph import skeleton_to_graph

H, W, ROW = 160, 320, 20
OFFICIAL = {
    "midpoint_m": 50.0,
    "min_path_m": 0.0,
    "skip_unscorable": False,
}  # the official command-line values


def _line(length: int = 300, row: int = ROW) -> np.ndarray:
    mask = np.zeros((H, W), bool)
    mask[row, 0 : length + 1] = True
    return mask


def _with_branch(mask: np.ndarray, rows: int = 100) -> np.ndarray:
    out = mask.copy()
    out[ROW + 1 : ROW + 1 + rows, 150] = True
    return out


def test_identical_maps_score_one_and_a_small_shift_too():
    assert apls(_line(), _line(), 1.0, **OFFICIAL)[0] == pytest.approx(1.0)
    assert apls(_line(), _line(row=ROW + 2), 1.0, **OFFICIAL)[0] == pytest.approx(1.0)  # inside the 4 m snap


def test_a_shift_beyond_the_snap_distance_scores_zero():
    assert apls(_line(), _line(row=ROW + 6), 1.0, **OFFICIAL)[0] == pytest.approx(0.0)


def test_a_twelve_metre_gap():
    """Reference control points at 0, 50, ..., 300 (7). The prediction lacks columns 145..155. The point at 150 is 6 m from a road end: missing.
    Three points lie each side of the gap: 6 + 9 = 15 of 21 pairs penalised, part 1 = 1 - 15/21; every prediction pair is exact, part 2 = 1."""
    pred = _line()
    pred[ROW, 145:156] = False
    score, part1, part2 = apls(_line(), pred, 1.0, **OFFICIAL)
    assert part1 == pytest.approx(6 / 21) and part2 == pytest.approx(1.0)
    assert score == pytest.approx(2 * (6 / 21) / (6 / 21 + 1))  # official: 0.444


def test_a_missing_branch_costs_part_one_and_an_extra_branch_part_two():
    """Reference with a branch: 9 control points, the two on the branch missing in a prediction without it -> 15 of 36 pairs penalised.
    The official code gives APLS 0.737 for both cases."""
    score, part1, part2 = apls(_with_branch(_line()), _line(), 1.0, **OFFICIAL)
    assert (
        part1 == pytest.approx(1 - 15 / 36)
        and part2 == pytest.approx(1.0)
        and score == pytest.approx(0.737, abs=0.001)
    )
    score, part1, part2 = apls(_line(), _with_branch(_line()), 1.0, **OFFICIAL)
    assert part1 == pytest.approx(1.0) and part2 == pytest.approx(1 - 15 / 36)


def test_the_minimum_path_length_skips_short_pairs():
    """An extra 8 m stub: the official code gives 0.857 without a minimum and 0.866 with its function default of 10 m."""
    pred = _with_branch(_line(), rows=8)
    assert apls(_line(), pred, 1.0, **OFFICIAL)[0] == pytest.approx(0.857, abs=0.002)
    assert apls(_line(), pred, 1.0, midpoint_m=50.0, min_path_m=10.0, skip_unscorable=False)[
        0
    ] == pytest.approx(0.866, abs=0.002)


def test_one_empty_map_scores_zero_either_way_round():
    empty = np.zeros((H, W), bool)
    assert apls(_line(), empty, 1.0) == (0.0, 0.0, 0.0)
    assert apls(empty, _line(), 1.0) == (0.0, 0.0, 0.0)


def test_both_empty_and_no_usable_pair_are_skipped_or_zero():
    empty = np.zeros((H, W), bool)
    short = np.zeros((H, W), bool)
    short[ROW, 0:6] = True  # a 5 m road: no pair is 10 m apart
    assert apls(empty, empty, 1.0) is None  # default: left out of the average
    assert apls(empty, empty, 1.0, skip_unscorable=False) == (0.0, 0.0, 0.0)  # official
    assert apls(short, short, 1.0, min_path_m=10.0) is None
    assert apls(short, short, 1.0, min_path_m=10.0, skip_unscorable=False) == (0.0, 0.0, 0.0)


def _xs(mask, **kw):
    g = skeleton_to_graph(mask, 1.0)
    nodes, edge_points = control_points(g, **kw)
    return sorted([float(g.xy[v][0]) for v in nodes] + [float(point_xy(g, e, s)[0]) for e, s in edge_points])


def test_midpoints_follow_the_official_rule():
    assert _xs(_line(30), midpoint_m=50.0) == [0, 30]  # shorter than 37.5 m: none
    assert _xs(_line(45), midpoint_m=50.0) == [0, 22.5, 45]  # up to 50 m: one in the middle
    assert _xs(_line(120), midpoint_m=50.0) == [0, 40, 80, 120]  # equal parts, not 50 and 100
    assert _xs(_line(300), midpoint_m=50.0) == [0, 50, 100, 150, 200, 250, 300]


def test_curved_only_option_skips_straight_roads():
    assert _xs(_line(120), midpoint_m=50.0, curved_eps=0.012) == [0, 120]
    bend = np.zeros((H, W), bool)
    bend[ROW, 0:61] = True
    bend[ROW : ROW + 61, 60] = True  # 60 m + 60 m: bounding box diagonal 85 m for 120 m of road
    assert control_points(skeleton_to_graph(bend, 1.0), midpoint_m=50.0, curved_eps=0.012)[1]


def test_control_points_landing_on_the_same_spot_all_count_as_snapped():
    """The reference ends in a small fork at column 100 (a junction, a 3 m stub and a road end 2 m on); its three control points near the end all land on the
    prediction's end node. They are all snapped, so no pair is fully penalised. (The official code renames the node the first one made, loses it, and scores it missing.)"""
    ref = _line(102)
    ref[ROW - 3 : ROW + 1, 100] = True
    pred = _line(100)
    _score, part1, part2 = apls(ref, pred, 1.0, midpoint_m=50.0, min_path_m=10.0, skip_unscorable=False)
    assert part1 > 0.95 and part2 > 0.95


def _rect(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> np.ndarray:
    mask[r0, c0 : c1 + 1] = mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = mask[r0 : r1 + 1, c1] = True
    return mask


def _shapes() -> dict[str, np.ndarray]:
    """Networks with closed rings, parallel edges (two roads between the same two junctions) and cycles."""
    canvas = lambda: np.zeros((H, W), bool)
    ring = _rect(canvas(), 20, 20, 60, 100)
    lollipop = _rect(canvas(), 20, 20, 60, 100)
    lollipop[60:100, 60] = True
    lens = _rect(canvas(), 20, 40, 60, 100)  # two junctions joined by two arcs, with a stub each
    lens[40, 20:41] = lens[40, 100:141] = True
    square_stubs = _rect(canvas(), 20, 40, 60, 100)
    square_stubs[10:21, 40] = square_stubs[60:71, 40] = square_stubs[10:21, 100] = square_stubs[
        60:71, 100
    ] = True
    eight = _rect(_rect(canvas(), 20, 20, 60, 70), 20, 70, 60, 120)
    mesh = _rect(_rect(canvas(), 20, 20, 60, 70), 20, 70, 60, 120)
    mesh[40, 20:71] = True
    return {
        "ring": ring,
        "lollipop": lollipop,
        "lens": lens,
        "square with stubs": square_stubs,
        "figure eight": eight,
        "mesh": mesh,
    }


@pytest.mark.parametrize("midpoint_m", [10.0, 50.0, 100.0])
def test_a_network_scored_against_itself_is_one_with_rings_parallel_edges_and_cycles(midpoint_m):
    for name, mask in _shapes().items():
        score = apls(mask, mask, 1.0, midpoint_m=midpoint_m, min_path_m=10.0, skip_unscorable=False)[0]
        assert score == pytest.approx(1.0), name


def test_distances_on_parallel_edges_and_a_ring_cut_at_one_point_are_not_added_up():
    """A ring of 158 m cut at one inserted point must keep the two halves as alternatives (79 m), and two direct edges between the same two nodes the
    shorter one. A sparse matrix adds duplicates up, which made these distances twice as long."""
    from hs_hot.apls import _augmented

    ring = _rect(np.zeros((H, W), bool), 20, 20, 40, 80)  # 160 m round
    g = skeleton_to_graph(ring, 1.0)
    _nodes, edge_points = control_points(g, midpoint_m=100.0)  # 160 m: split in two, so exactly one midpoint
    assert len(g) == 1 and len(edge_points) == 1
    from scipy.sparse.csgraph import dijkstra

    d = dijkstra(_augmented(g, edge_points), directed=False, indices=[0])[0, 1]
    assert d == pytest.approx(g.edges[0][3] / 2, abs=1.0)
    lens = np.zeros((H, W), bool)
    _rect(lens, 20, 40, 26, 60)
    lens[23, 20:41] = lens[23, 60:81] = True  # arcs of 26 m: too short for a midpoint, so two direct edges
    g = skeleton_to_graph(lens, 1.0)
    d = dijkstra(_augmented(g, []), directed=False, indices=np.arange(len(g)))
    junctions = [i for i in range(len(g)) if len(g.adjacent[i]) == 3]  # the two nodes that the two arcs join
    assert len(junctions) == 2
    arcs = [e[3] for e in g.edges if {e[0], e[1]} == set(junctions)]
    assert len(arcs) == 2
    assert d[junctions[0], junctions[1]] == pytest.approx(min(arcs))


@pytest.mark.parametrize(
    ("what", "midpoint_m", "official"),
    [
        ("ring vs the ring shifted 2 m", 100.0, 0.973),
        ("ring vs the ring opened by a 10 m gap", 100.0, 0.815),
        ("small lens vs the lens with one arc cut", 50.0, 0.958),
        ("small lens vs the lens shifted 2 m", 50.0, 0.925),
    ],
)
def test_loops_and_parallel_edges_match_the_official_code(what, midpoint_m, official):
    """Official CosmiQ scores on the same networks (research/analysis/official_check_loops.py). A ring with one midpoint and a lens with no midpoints were
    the cases where adding up duplicate sparse entries gave 0.49 and 0.70 instead."""
    ring = _rect(np.zeros((H, W), bool), 20, 20, 40, 80)
    ring_open = ring.copy()
    ring_open[20, 45:55] = False
    small = _rect(np.zeros((H, W), bool), 20, 40, 26, 60)
    small[23, 20:41] = small[23, 60:81] = True
    small_cut = small.copy()
    small_cut[20, 41:60] = False
    pairs = {
        "ring vs the ring shifted 2 m": (ring, _rect(np.zeros((H, W), bool), 22, 22, 42, 82)),
        "ring vs the ring opened by a 10 m gap": (ring, ring_open),
        "small lens vs the lens with one arc cut": (small, small_cut),
        "small lens vs the lens shifted 2 m": (small, np.roll(small, 2, axis=0)),
    }
    ref, pred = pairs[what]
    score = apls(ref, pred, 1.0, midpoint_m=midpoint_m, min_path_m=10.0, skip_unscorable=False)[0]
    assert score == pytest.approx(official, abs=0.002)

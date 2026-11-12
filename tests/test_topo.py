"""TOPO on hand-built roads: counts worked out by hand; the official counts were matched tile for tile in research/analysis/official_check_compare.py."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from hs_hot.road_graph import skeleton_to_graph
from hs_hot.topo import draw_seeds, points_around, summarise, topo_counts

H, W, ROW = 160, 320, 20
OFFICIAL = {"radius_m": 150.0, "interval_m": 30.0, "hole_m": 5.0}  # the official function defaults


def _road(length: int) -> np.ndarray:
    mask = np.zeros((H, W), bool)
    mask[ROW, 0 : length + 1] = True
    return mask


def _counts(ref, pred, **kw):
    ref_g, pred_g = skeleton_to_graph(ref, 1.0), skeleton_to_graph(pred, 1.0)
    left_end = [int(np.argmin(ref_g.xy[:, 0]))] if len(ref_g) else []
    return tuple(int(v) for v in topo_counts(ref_g, pred_g, left_end, **(OFFICIAL | kw)))


def test_a_road_against_itself_fills_every_hole():
    assert _counts(_road(120), _road(120)) == (5, 0, 0)  # holes: two ends and points at 30, 60, 90


def test_a_gap_gives_one_false_positive_and_three_misses():
    """Prediction 0..49 and 71..120. From the left end: marbles at 0, 30 and 49. Holes 0, 30, 60, 90, 120. 49 is 11 m from the nearest hole."""
    pred = _road(120)
    pred[ROW, 50:71] = False
    assert _counts(_road(120), pred) == (2, 1, 3)


def test_no_prediction_at_the_measurement_node_misses_every_hole():
    far = np.zeros((H, W), bool)
    far[ROW + 20, 0:121] = True
    assert _counts(_road(120), far) == (0, 0, 5)
    assert _counts(_road(120), np.zeros((H, W), bool)) == (0, 0, 5)  # empty prediction: counted as missed


def test_no_reference_adds_nothing():
    empty = np.zeros((H, W), bool)
    assert _counts(empty, _road(120)) == (0, 0, 0)
    assert len(draw_seeds(skeleton_to_graph(empty, 1.0))) == 0


def test_short_edges_get_no_points_and_edges_beyond_the_radius_are_dropped():
    assert _counts(_road(20), _road(20)) == (2, 0, 0)  # 20 m < 30 m interval: only the two nodes
    assert _counts(_road(300), _road(300)) == (1, 0, 0)  # the far end is outside 150 m: the edge is dropped


def test_points_run_away_from_the_measurement_node():
    g = skeleton_to_graph(_road(100), 1.0)
    right = int(np.argmax(g.xy[:, 0]))
    from_right = sorted(np.round(points_around(g, right, 150.0, 30.0)[:, 0]).astype(int))
    from_left = sorted(np.round(points_around(g, 1 - right, 150.0, 30.0)[:, 0]).astype(int))
    assert from_right == [0, 10, 40, 70, 100] and from_left == [0, 30, 60, 90, 100]


def test_one_marble_takes_one_hole():
    """The road drawn twice, joined by a link: 10 marbles for 5 holes -> 5 true and 5 false positives."""
    pred = _road(120)
    pred[ROW + 2, 0:121] = True
    pred[ROW : ROW + 3, 60] = True
    assert _counts(_road(120), pred) == (5, 5, 0)


def test_seeds_are_every_node_once_or_a_capped_draw_with_replacement():
    g = skeleton_to_graph(_road(120), 1.0)
    assert draw_seeds(g).tolist() == [0, 1]
    assert len(draw_seeds(g, 20, np.random.default_rng(0))) == 2  # at most the node count
    drawn = np.concatenate([draw_seeds(g, 2, np.random.default_rng(i)) for i in range(40)])
    assert set(drawn) <= {0, 1}


def test_summary_handles_zero_denominators():
    assert summarise((0, 0, 0))["f1"] == 0.0
    s = summarise((5, 0, 5))
    assert s["precision"] == 1.0 and s["recall"] == 0.5

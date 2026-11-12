"""Pooled APLS and TOPO over tiles, the border margin and the fragment filter (tiny generated masks, offline)."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from hs_hot.evaluate_graph import GraphScoring, GraphTally, crop_border

H, W, ROW = 120, 200, 40


def _road(length: int = 120, row: int = ROW) -> np.ndarray:
    mask = np.zeros((H, W), bool)
    mask[row, 0 : length + 1] = True
    return mask


def test_a_perfect_prediction_scores_one_everywhere():
    tally = GraphTally()
    tally.add(_road(), _road(), 1.0)
    out = tally.summary()
    assert out["apls"] == pytest.approx(1.0) and out["chips_scored"] == 1 and out["chips_skipped"] == 0
    assert out["topo"]["precision"] == 1.0 and out["topo"]["recall"] == 1.0


def test_an_empty_tile_with_an_empty_prediction_is_skipped_not_scored_zero():
    tally = GraphTally()
    tally.add(np.zeros((H, W), bool), np.zeros((H, W), bool), 1.0)
    out = tally.summary()
    assert out["apls"] is None and out["chips_skipped"] == 1
    tally = GraphTally(GraphScoring(skip_unscorable=False))
    tally.add(np.zeros((H, W), bool), np.zeros((H, W), bool), 1.0)
    assert tally.summary()["apls"] == 0.0  # the official rule


def test_false_roads_on_an_empty_tile_score_zero():
    tally = GraphTally()
    tally.add(np.zeros((H, W), bool), _road(), 1.0)
    assert tally.summary()["apls"] == 0.0


def test_the_border_margin_removes_a_false_road_that_lies_in_the_strip():
    """The prediction has the reference road plus a road along the tile edge (inside the strip): a false road without a margin, ignored with one."""
    pred = _road()
    pred[2, 0:150] = True  # 2 px from the top edge
    plain = GraphTally(GraphScoring(border_margin_m=0.0))
    cropped = GraphTally(GraphScoring(border_margin_m=5.0))
    for tally in (plain, cropped):
        tally.add(_road(), pred, 1.0)
    assert plain.summary()["apls"] < 0.99
    assert cropped.summary()["apls"] == pytest.approx(1.0)


def test_crop_border_and_zero_margin():
    mask = np.ones((6, 6), bool)
    assert crop_border(mask, 2).shape == (2, 2)
    assert crop_border(mask, 0) is mask


def test_the_fragment_filter_drops_short_noise_before_scoring():
    pred = _road()
    pred[100, 10:14] = True  # a 3 m fragment far from any road
    with_filter = GraphTally()
    without = GraphTally(GraphScoring(pred_min_fragment_m=0.0))
    for tally in (with_filter, without):
        tally.add(_road(), pred, 1.0)
    assert with_filter.summary()["apls"] == pytest.approx(1.0)
    assert without.summary()["apls"] < 1.0


def test_the_summary_carries_diagnostics_and_names_the_metric_variant():
    from hs_hot.evaluate_graph import DIFFERENCES_FROM_OFFICIAL, METRIC_VARIANT

    pred = _road()
    pred[100, 10:14] = True  # a 3 m fragment, removed before scoring
    tally = GraphTally()
    tally.add(_road(), pred, 1.0)
    tally.add(np.zeros((H, W), bool), np.zeros((H, W), bool), 1.0)  # skipped by APLS
    out = tally.summary()
    assert out["diagnostics"]["fragments_removed"] == {"reference": 0, "prediction": 1}
    assert out["diagnostics"]["apls_tiles_skipped"] == 1 and out["chips_skipped"] == 1
    assert out["diagnostics"]["skeleton_pixels_untraced"] == {"reference": 0, "prediction": 0}
    assert out["diagnostics"]["dead_end_walks"] == {"reference": 0, "prediction": 0}
    assert out["metric_variant"] == METRIC_VARIANT and "HOTOSM" in METRIC_VARIANT
    assert any(
        "empty prediction" in line for line in DIFFERENCES_FROM_OFFICIAL
    )  # the deliberate TOPO difference is listed

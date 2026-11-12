"""The line scorer's border margin, its median offset, a gap, and the input checks shared with the graph scorer (tiny generated masks, offline)."""

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from hs_hot.evaluate_centreline import Tally, crop_border, margin_pixels
from hs_hot.evaluate_graph import GraphScoring, GraphTally


def _row(row: int, start: int = 0, stop: int = 100, height: int = 40, width: int = 120) -> np.ndarray:
    mask = np.zeros((height, width), bool)
    mask[row, start:stop] = True
    return mask


def test_a_gap_in_the_prediction_costs_recall_and_not_precision():
    """The reference is 100 m long; the prediction lacks columns 40..49. At 0.5 m tolerance those 10 reference pixels have no prediction within reach."""
    pred = _row(20)
    pred[20, 40:50] = False
    tally = Tally()
    tally.add(pred, _row(20), 1.0)
    out = tally.summary()["r0.5"]
    assert out["precision"] == 1.0 and out["recall"] == pytest.approx(0.9)
    assert out["f1"] == pytest.approx(2 * 0.9 / 1.9, abs=1e-4)


def test_identical_lines_have_a_median_offset_of_exactly_zero_and_a_shift_is_reported_as_it_is():
    same = Tally()
    same.add(_row(20), _row(20), 0.3)
    assert same.summary()["median_offset_m"] == 0.0
    shifted = Tally()
    shifted.add(_row(24), _row(20), 0.25)  # 4 pixels = exactly 1.0 m
    assert shifted.summary()["median_offset_m"] == pytest.approx(1.0, abs=0.01)


def test_the_median_offset_ignores_pixels_beyond_five_metres_and_says_how_many():
    pred = _row(20)
    pred[38, 0:50] = True  # 18 px from the reference line: in the last bin, not in the median
    tally = Tally()
    tally.add(pred, _row(20), 1.0)
    out = tally.summary()
    assert out["median_offset_m"] == 0.0
    assert out["share_within_5m"] == pytest.approx(100 / 150, abs=1e-3)


def test_a_false_line_in_the_border_strip_is_ignored_with_a_margin_and_counted_without():
    pred = _row(20)
    pred[1, 0:100] = True  # one pixel from the top edge
    plain, cropped = Tally(), Tally()
    plain.add(pred, _row(20), 1.0)
    cropped.add(pred, _row(20), 1.0, border_margin_m=3.0)
    assert plain.summary()["r1"]["precision"] == pytest.approx(0.5)
    assert cropped.summary()["r1"]["precision"] == 1.0 and cropped.summary()["r1"]["recall"] == 1.0
    assert (
        cropped.n_pred == 100 - 3
    )  # the line starts at the left edge, so the margin also takes 3 columns off its start


def test_the_margin_is_in_metres_so_it_scales_with_the_pixel_size():
    assert margin_pixels(3.6, 0.3) == 12 and margin_pixels(3.6, 0.15) == 24 and margin_pixels(0.0, 0.3) == 0
    assert margin_pixels(3.0, 1.0) == 3 and margin_pixels(3.1, 1.0) == 4  # rounded up


def test_a_margin_that_leaves_nothing_raises_instead_of_scoring_an_empty_tile():
    tile = np.ones((20, 20), bool)
    assert crop_border(tile, 9).shape == (2, 2)
    with pytest.raises(ValueError, match="leaves nothing"):
        crop_border(tile, 10)
    with pytest.raises(ValueError, match="leaves nothing"):
        Tally().add(tile, tile, 1.0, border_margin_m=15.0)
    with pytest.raises(ValueError, match="leaves nothing"):
        GraphTally(GraphScoring(border_margin_m=15.0)).add(tile, tile, 1.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_the_pixel_size_must_be_finite_and_positive_in_both_scorers(bad):
    line = _row(20)
    with pytest.raises(ValueError, match="metres_per_pixel"):
        Tally().add(line, line, bad)
    with pytest.raises(ValueError, match="metres_per_pixel"):
        GraphTally().add(line, line, bad)


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_the_margin_must_be_finite_and_not_negative(bad):
    line = _row(20)
    with pytest.raises(ValueError, match="border_margin_m"):
        Tally().add(line, line, 1.0, border_margin_m=bad)


def test_the_graph_scorer_refuses_masks_of_different_shapes():
    with pytest.raises(ValueError, match="differ in shape"):
        GraphTally().add(_row(20), _row(20, width=130), 1.0)

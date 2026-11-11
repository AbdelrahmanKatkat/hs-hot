"""Centreline helpers: thinning a mask, widening a line, and the tolerance scores (tiny generated masks, offline)."""

import numpy as np
import pytest

pytest.importorskip("skimage")
pytest.importorskip("scipy")

from hs_hot.centreline import thin_to_centreline, widen_centreline
from hs_hot.evaluate_centreline import Tally, draw_lines


def _line(row: int, size: int = 40) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[row, :] = True
    return mask


def test_a_band_thins_to_a_line_inside_it_along_its_middle():
    band = np.zeros((40, 40), dtype=bool)
    band[17:24, :] = True  # 7 pixels wide, middle row 20
    line = thin_to_centreline(band)
    assert not (line & ~band).any()
    assert line.sum() <= 40 * 1.2  # about one pixel per column
    rows = np.flatnonzero(line.any(axis=1))
    assert rows.min() >= 19 and rows.max() <= 21
    assert (thin_to_centreline(line) == line).all()


def test_an_empty_mask_thins_to_nothing_and_a_wrong_shape_is_refused():
    assert not thin_to_centreline(np.zeros((8, 8), dtype=bool)).any()
    with pytest.raises(ValueError, match="2-D"):
        thin_to_centreline(np.zeros((2, 8, 8), dtype=bool))


def test_widening_a_line_reaches_exactly_the_radius():
    line = np.zeros((21, 21), dtype=bool)
    line[10, 10] = True
    wide = widen_centreline(line, 3.0)
    assert wide[10, 13] and wide[7, 10] and not wide[10, 14] and not wide[6, 10]
    assert not widen_centreline(np.zeros((5, 5), dtype=bool), 2.0).any()
    with pytest.raises(ValueError, match="radius_px"):
        widen_centreline(line, -1.0)


def test_draw_lines_gives_a_one_pixel_line():
    drawn = draw_lines([np.array([[0.0, 5.0], [39.0, 5.0]])], 10, 40)
    assert drawn.sum() == 40 and drawn[5].all()


def test_a_line_scored_against_itself_is_perfect():
    tally = Tally()
    tally.add(_line(10), _line(10), 0.3)
    for radius in ("r0.5", "r1", "r4"):
        scores = tally.summary()[radius]
        assert scores["precision"] == scores["recall"] == scores["f1"] == scores["quality"] == 1.0
        assert scores["buffered_iou"] == 1.0


def test_an_offset_inside_the_tolerance_matches_and_outside_it_does_not():
    tally = Tally()
    tally.add(_line(14), _line(10), 0.25)  # 4 pixels = 1.0 m apart
    summary = tally.summary()
    assert summary["r1"]["f1"] == 1.0 and summary["r0.5"]["f1"] == 0.0
    assert summary["r1"]["buffered_iou"] == pytest.approx(
        5 / 13, abs=1e-3
    )  # (scores are rounded to 4 decimals) strips of 9 rows, 5 of them shared, 13 in the union
    assert 1.0 <= summary["median_offset_m"] <= 1.1


def test_the_distance_is_in_metres_so_the_pixel_size_matters():
    close, far = Tally(), Tally()
    close.add(_line(14), _line(10), 0.25)  # 1.0 m
    far.add(_line(14), _line(10), 0.5)  # 2.0 m
    assert close.summary()["r1"]["f1"] == 1.0 and far.summary()["r1"]["f1"] == 0.0


def test_empty_prediction_empty_reference_and_both_empty_do_not_divide_by_zero():
    nothing = np.zeros((20, 20), dtype=bool)
    missed, invented, neither = Tally(), Tally(), Tally()
    missed.add(nothing, _line(5, 20), 0.3)
    invented.add(_line(5, 20), nothing, 0.3)
    neither.add(nothing, nothing, 0.3)
    assert missed.summary()["r1"]["recall"] == 0.0 and missed.summary()["r1"]["precision"] == 0.0
    assert invented.summary()["r1"]["precision"] == 0.0 and "median_offset_m" not in invented.summary()
    assert neither.summary()["r1"] == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "quality": 0.0,
        "buffered_iou": 0.0,
    }


def test_counts_pool_over_chips_and_bad_input_is_refused():
    tally = Tally()
    tally.add(_line(10), _line(10), 0.3)
    tally.add(np.zeros((40, 40), dtype=bool), _line(10), 0.3)  # a chip where the whole road was missed
    assert tally.summary()["r1"]["recall"] == pytest.approx(0.5)
    assert tally.summary()["r1"]["precision"] == 1.0
    with pytest.raises(ValueError, match="differ in shape"):
        tally.add(_line(1, 10), _line(1, 20), 0.3)
    with pytest.raises(ValueError, match="metres_per_pixel"):
        tally.add(_line(1), _line(1), 0.0)

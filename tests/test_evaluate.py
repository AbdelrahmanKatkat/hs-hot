"""Pixel scoring: counts, the metrics derived from them, and scoring through an ONNX-like session."""

from pathlib import Path

import numpy as np
import pytest

from hs_hot.dataset import BACKGROUND, IGNORE, ROAD
from hs_hot.evaluate import _prf, pick_chips, pixel_counts, score_session


def test_counts_at_several_thresholds_leave_out_the_ignored_pixels():
    truth = np.array(
        [[ROAD, ROAD, BACKGROUND, BACKGROUND], [ROAD, IGNORE, IGNORE, BACKGROUND]], dtype=np.uint8
    )
    probability = np.array([[0.9, 0.4, 0.6, 0.1], [0.2, 0.99, 0.99, 0.55]])
    counts = pixel_counts(probability, truth, (0.5, 0.3))
    assert counts.tolist() == [
        [1, 2, 2],  # at 0.5: tp (0,0); fp (0,2) and (1,3); fn (0,1) and (1,0)
        [2, 2, 1],  # at 0.3: the 0.4 pixel is now found
    ]


def test_metrics_from_counts_and_the_empty_cases():
    metrics = _prf(tp=8, fp=2, fn=2)
    assert metrics["precision"] == 0.8 and metrics["recall"] == 0.8 and metrics["f1"] == pytest.approx(0.8)
    assert metrics["iou"] == pytest.approx(8 / 12)
    assert _prf(0, 0, 0) == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "iou": 0.0,
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
    }


def test_pick_chips_spans_the_list_evenly():
    paths = [Path(f"{index:02d}.tif") for index in range(20)]
    assert pick_chips(paths, None) == paths and pick_chips(paths, 50) == paths
    assert [path.name for path in pick_chips(paths, 4)] == ["00.tif", "05.tif", "10.tif", "15.tif"]


class FakeSession:
    """Stands in for onnxruntime: the road is the top half of the image, or nothing."""

    class _In:
        name = "image"

    def __init__(self, finds_the_road: bool):
        self.finds_the_road = finds_the_road

    def get_inputs(self):
        return [self._In()]

    def run(self, _outputs, feeds):
        _, height, width = feeds["image"].shape
        probability = np.zeros((height, width), dtype=np.float32)
        if self.finds_the_road:
            probability[: height // 2, :] = 0.95
        return [probability]


def _half_labels(names):
    """The top half of each 256 px chip is road: a band 128 px wide along row 64, long enough to leave the chip."""
    band = {"kind": "road", "line": np.array([[-300.0, 64.0], [556.0, 64.0]]), "width_px": 128.0}
    return {name: [band] for name in names}


def test_a_session_that_finds_the_road_scores_high_and_one_that_does_not_scores_zero(labelled_chips):
    chips, _, names = labelled_chips
    labels, params = _half_labels(names), {"model_gsd_m": 0.5}
    good = score_session(
        FakeSession(True), images_dir=chips, labels=labels, params=params, thresholds=(0.5, 0.99)
    )
    assert good[0.5]["iou"] > 0.9 and good[0.5]["fp"] < 0.05 * good[0.5]["tp"]
    assert good[0.99]["tp"] == 0  # nothing reaches 0.99
    bad = score_session(FakeSession(False), images_dir=chips, labels=labels, names=names[:2], params=params)[
        0.5
    ]
    assert bad["tp"] == 0 and bad["recall"] == 0.0 and bad["fn"] > 0


def test_the_labels_may_be_a_geojson_path_and_a_chip_without_labels_counts_as_no_roads(labelled_chips):
    chips, geojson, names = labelled_chips
    scores = score_session(FakeSession(True), images_dir=chips, labels=geojson, params={"model_gsd_m": 0.5})[
        0.5
    ]
    assert (
        scores["tp"] > 0 and 0.3 < scores["recall"] < 0.9
    )  # the road across is found; the diagonal one is not
    nothing = score_session(FakeSession(True), images_dir=chips, labels={}, params={"model_gsd_m": 0.5})[0.5]
    assert nothing["tp"] == 0 and nothing["fp"] > 0 and nothing["fn"] == 0

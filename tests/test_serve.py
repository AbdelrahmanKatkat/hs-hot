"""Inference with a fake ONNX session: `predict_session` gives probabilities; decoding and georeferencing are the caller's."""

import numpy as np
import pytest
from PIL import Image

from conftest import CHIP_NAME
from hs_hot.params import PostprocessParams
from hs_hot.postprocess import postprocess_outputs
from hs_hot.serve import list_images, predict_session


def at_fine_scale(params):
    """The tests plant a block in model pixels of a 756 px chip: the zoom-20 chip scaled to 0.05 m, not the default 0.15 m."""
    return {"model_gsd_m": 0.05, **(params or {})}


def collect(session, folder, params):
    """What a pack's `predict` does: `predict_session`, then `postprocess_outputs`."""
    params = at_fine_scale(params)
    return postprocess_outputs(predict_session(session, folder, params), params)


class FakeSession:
    """Stands in for onnxruntime: returns no road, or one planted block, as the `probability` output."""

    class _In:
        name = "image"

    def __init__(self, plant: bool = False):
        self.plant = plant
        self.calls: list[tuple[int, ...]] = []

    def get_inputs(self):
        return [self._In()]

    def run(self, _outputs, feeds):
        image = feeds["image"]
        assert image.ndim == 3 and image.dtype == np.float32 and 0.0 <= image.min() <= image.max() <= 1.0
        self.calls.append(image.shape)
        probability = np.zeros(image.shape[1:], dtype=np.float32)
        if self.plant:
            probability[300:600, 150:450] = 0.9  # model-input pixels: rows 300-600, columns 150-450
        return [probability]


def test_list_images_is_sorted_and_an_empty_folder_is_an_error(tmp_path):
    (tmp_path / "b.png").write_bytes(b"")
    (tmp_path / "a.tif").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    assert [path.name for path in list_images(tmp_path)] == ["a.tif", "b.png"]
    with pytest.raises(FileNotFoundError, match="No input images"):
        list_images(tmp_path / "missing")


def test_a_polygon_comes_back_georeferenced(write_chip, tmp_path):
    write_chip(name=CHIP_NAME)
    (feature,) = collect(FakeSession(plant=True), tmp_path, None)["features"]
    assert feature["properties"] == {
        "confidence": 0.9,
        "class_id": 1,
        "class_name": "highway",
        "source": CHIP_NAME,
    }
    ring = np.array(feature["geometry"]["coordinates"][0])
    assert (np.abs(ring[:, 0] - 8.98) < 0.01).all() and (np.abs(ring[:, 1] - 17.68) < 0.05).all()  # lon, lat


def test_the_model_sees_the_scaled_chip(write_chip, tmp_path):
    write_chip(name=CHIP_NAME)
    session = FakeSession(plant=True)
    ((_, probability, meta),) = predict_session(session, tmp_path, at_fine_scale(None))
    assert session.calls == [(3, meta.resized_height, meta.resized_width)] and meta.resized_width > 700  # ~3x
    assert probability.shape == (meta.resized_height, meta.resized_width)


def test_predict_session_collects_every_chip_and_empty_or_missing_params_are_safe(write_chip, tmp_path):
    write_chip(name="OAM-524288-524288-20.tif")
    write_chip(name="OAM-524289-524288-20.tif")
    session = FakeSession(plant=True)
    collection = collect(session, tmp_path, {})
    assert len(session.calls) == 2 and len(collection["features"]) == 2
    assert {f["properties"]["source"] for f in collection["features"]} == {
        "OAM-524288-524288-20.tif",
        "OAM-524289-524288-20.tif",
    }
    assert collect(FakeSession(plant=False), tmp_path, None) == {
        "type": "FeatureCollection",
        "features": [],
    }


def test_params_change_the_decoding_not_the_model_call(write_chip, tmp_path):
    write_chip(name=CHIP_NAME)
    assert len(collect(FakeSession(plant=True), tmp_path, {"threshold": 0.95})["features"]) == 0
    assert len(collect(FakeSession(plant=True), tmp_path, {"confidence_threshold": 0.5})["features"]) == 1
    big = {"min_area_px": int(PostprocessParams().min_area_px) * 1000}
    assert len(collect(FakeSession(plant=True), tmp_path, big)["features"]) == 0


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_a_png_chip_keeps_chip_pixel_coordinates(tmp_path):
    Image.new("RGB", (256, 256)).save(tmp_path / CHIP_NAME.replace(".tif", ".png"))
    (feature,) = collect(FakeSession(plant=True), tmp_path, None)["features"]
    ring = np.array(feature["geometry"]["coordinates"][0])
    lower, upper = (
        ring.min(axis=0),
        ring.max(axis=0),
    )  # the scale is 2.98: columns 150-450 -> 50.3-151, rows 300-600 -> 100.7-201.3
    assert lower == pytest.approx([50.3, 100.7], abs=1.0) and upper == pytest.approx([151.0, 201.3], abs=1.0)


def test_predict_session_returns_probabilities_and_decodes_nothing(write_chip, tmp_path):
    write_chip(name="OAM-524288-524288-20.tif")
    write_chip(name="OAM-524289-524288-20.tif")
    session = FakeSession(plant=True)
    outputs = predict_session(
        session, tmp_path, at_fine_scale({"threshold": 0.99})
    )  # a decoding key has no effect here
    assert [path.name for path, _, _ in outputs] == ["OAM-524288-524288-20.tif", "OAM-524289-524288-20.tif"]
    assert len(session.calls) == 2  # one model pass per chip
    for _, probability, meta in outputs:
        assert (
            probability.shape == (meta.resized_height, meta.resized_width) and probability.dtype == np.float32
        )
        assert probability.max() == pytest.approx(0.9)

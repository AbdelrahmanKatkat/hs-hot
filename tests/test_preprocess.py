import numpy as np
import pytest

from conftest import CHIP_NAME
from hs_hot.params import PreprocessParams
from hs_hot.preprocess import preprocess_rgb, preprocess_chip
from hs_hot.tiles import chip_scale


def test_chip_is_resized_once_to_chw_float_in_unit_range():
    rgb = np.full((256, 256, 3), 128, dtype=np.uint8)
    image, meta = preprocess_rgb(rgb, 3.125)
    assert image.shape == (3, 800, 800) and image.dtype == np.float32
    assert (meta.resized_width, meta.resized_height) == (800, 800)
    assert (meta.scale_x, meta.scale_y) == (3.125, 3.125)
    np.testing.assert_allclose(image, 128 / 255, atol=1e-6)  # not normalised: the graph does that


def test_channels_stay_in_rgb_order():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 255, 128, 0
    image, _ = preprocess_rgb(rgb, 2.0)
    np.testing.assert_allclose(image[:, 3, 3], [1.0, 128 / 255, 0.0], atol=1e-6)


def test_scale_below_one_downsizes_and_rounds_the_size():
    image, meta = preprocess_rgb(np.zeros((256, 300, 3), dtype=np.uint8), 0.5)
    assert image.shape == (3, 128, 150) and (meta.width, meta.height) == (300, 256)
    image, meta = preprocess_rgb(np.zeros((100, 100, 3), dtype=np.uint8), 2.99)
    assert image.shape == (3, 299, 299) and meta.scale_x == pytest.approx(2.99)


def test_no_long_side_cap():
    image, _ = preprocess_rgb(np.zeros((200, 1000, 3), dtype=np.uint8), 4.0)
    assert image.shape == (3, 800, 4000)


def test_rejects_non_rgb_arrays():
    with pytest.raises(ValueError):
        preprocess_rgb(np.zeros((10, 10, 4), dtype=np.uint8), 1.0)


def test_preprocess_chip_takes_the_scale_from_the_oam_zoom(write_chip):
    image, meta = preprocess_chip(write_chip(name=CHIP_NAME))
    expected = chip_scale(CHIP_NAME, 256, 0.30)
    assert image.shape == (3, round(256 * expected), round(256 * expected))
    assert (meta.width, meta.height) == (256, 256)
    assert str(meta.crs) == "EPSG:3857" and meta.transform.a == 1.0


def test_a_coarse_zoom_is_capped_at_max_input_px(write_chip):
    path = write_chip(
        name="OAM-193343-110117-18.tif"
    )  # zoom 18 at latitude 27.6: at 0.05 m uncapped x10.6 = 2,700 px
    capped, meta = preprocess_chip(path, PreprocessParams(model_gsd_m=0.05))
    assert capped.shape == (3, 1024, 1024) and (meta.resized_width, meta.resized_height) == (1024, 1024)
    loose, _ = preprocess_chip(path, PreprocessParams(model_gsd_m=0.05, max_input_px=2048))
    assert loose.shape == (3, 2048, 2048)


def test_model_gsd_changes_the_scale(write_chip):
    path = write_chip(name=CHIP_NAME)
    coarse, _ = preprocess_chip(path, PreprocessParams(model_gsd_m=0.10))
    fine, _ = preprocess_chip(path, PreprocessParams(model_gsd_m=0.05))
    assert fine.shape[1] == pytest.approx(2 * coarse.shape[1], abs=1)


def test_preprocess_chip_rejects_non_oam_names(write_chip):
    with pytest.raises(ValueError, match="OAM"):
        preprocess_chip(write_chip(name="chip.tif"))


def test_preprocess_chip_rejects_fewer_than_three_bands(write_chip):
    with pytest.raises(ValueError, match="3 bands"):
        preprocess_chip(write_chip(name=CHIP_NAME, count=1))

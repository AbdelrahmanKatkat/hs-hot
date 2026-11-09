"""The augmentation steps and the optional post-processing steps: off by default, and each does what it says."""

import numpy as np
import pytest

from hs_hot.augment import augment_chip_and_mask
from hs_hot.dataset import BACKGROUND, IGNORE, ROAD
from hs_hot.decode import close_mask, decode_polygons, open_mask, smooth_probability
from hs_hot.params import AugmentParams, PostprocessParams, parse_postprocess_params, parse_train_params


def _pair(size=40):
    rng = np.random.default_rng(0)
    image = rng.random((3, size, size)).astype(np.float32)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[10:20, 5:30] = ROAD
    mask[30:34, 30:34] = IGNORE
    return image, mask


def test_defaults_keep_the_old_behaviour_and_the_new_steps_are_off():
    assert parse_train_params(None).orientation == "none"
    train = parse_train_params({})
    assert (train.photometric, train.scale_jitter, train.degrade_prob) == (0.0, 0.0, 0.0)
    post = PostprocessParams()
    assert (post.smooth_sigma_px, post.open_px, post.close_px, post.fill_holes_px) == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "bad", [{"orientation": "spin"}, {"photometric": 1.0}, {"scale_jitter": -0.1}, {"degrade_prob": 1.5}]
)
def test_bad_training_values_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_train_params(bad)


@pytest.mark.parametrize("bad", [{"open_px": -1}, {"smooth_sigma_px": -1.0}, {"fill_holes_px": -5}])
def test_bad_postprocess_values_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_postprocess_params(bad)


def test_the_default_changes_nothing():
    image, mask = _pair()
    same_image, same_mask = augment_chip_and_mask(image, mask, np.random.default_rng(1))
    assert np.array_equal(same_image, image) and np.array_equal(same_mask, mask)


def test_orientation_modes_move_pixels_only():
    image, mask = _pair()
    for orientation in ("dihedral", "mirror", "none"):
        moved_image, moved_mask = augment_chip_and_mask(
            image, mask, np.random.default_rng(1), AugmentParams(orientation=orientation)
        )
        assert moved_image.shape == image.shape and moved_mask.shape == mask.shape
        assert sorted(moved_mask.ravel().tolist()) == sorted(mask.ravel().tolist())
    same_image, same_mask = augment_chip_and_mask(
        image, mask, np.random.default_rng(1), AugmentParams(orientation="none")
    )
    assert np.array_equal(same_image, image) and np.array_equal(same_mask, mask)


def test_mirror_never_turns_the_chip():
    image, mask = _pair()
    for seed in range(8):
        _, moved = augment_chip_and_mask(
            image, mask, np.random.default_rng(seed), AugmentParams(orientation="mirror")
        )
        assert np.array_equal(moved, mask) or np.array_equal(moved, mask[:, ::-1])


def test_photometric_and_degrade_change_the_image_but_not_the_mask():
    image, mask = _pair()
    params = AugmentParams(orientation="none", photometric=0.3, degrade_prob=1.0)
    changed_image, changed_mask = augment_chip_and_mask(image, mask, np.random.default_rng(2), params)
    assert np.array_equal(changed_mask, mask)
    assert not np.allclose(changed_image, image)
    assert changed_image.min() >= 0.0 and changed_image.max() <= 1.0 and changed_image.dtype == np.float32


@pytest.mark.parametrize("seed", range(6))
def test_scale_jitter_keeps_the_size_and_ignores_the_padding(seed):
    image, mask = _pair()
    params = AugmentParams(orientation="none", scale_jitter=0.3)
    jittered_image, jittered_mask = augment_chip_and_mask(image, mask, np.random.default_rng(seed), params)
    assert jittered_image.shape == image.shape and jittered_mask.shape == mask.shape
    assert set(np.unique(jittered_mask)) <= {BACKGROUND, ROAD, IGNORE}


def test_smoothing_keeps_the_sum_of_a_flat_field_and_softens_a_spike():
    flat = np.full((20, 20), 0.4, dtype=np.float32)
    assert np.allclose(smooth_probability(flat, 1.5), 0.4, atol=1e-5)
    spike = np.zeros((21, 21), dtype=np.float32)
    spike[10, 10] = 1.0
    smoothed = smooth_probability(spike, 1.0)
    assert smoothed[10, 10] < 1.0 and smoothed[10, 11] > 0 and smoothed.shape == spike.shape
    assert smooth_probability(spike, 0.0) is spike


def test_opening_removes_a_speck_and_closing_fills_a_pin_hole():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:20, 5:20] = 1
    mask[25, 25] = 1  # a speck
    assert open_mask(mask, 1)[25, 25] == 0 and open_mask(mask, 1)[10, 10] == 1
    holed = mask.copy()
    holed[12, 12] = 0
    assert close_mask(holed, 1)[12, 12] == 1
    edge = np.zeros((10, 10), dtype=np.uint8)
    edge[:, :5] = 1
    assert close_mask(edge, 2)[:, 0].all()  # the chip edge is not eroded
    assert open_mask(mask, 0) is mask and close_mask(mask, 0) is mask


def test_decode_fills_small_holes_and_merges_with_closing_but_not_by_default():
    probability = np.zeros((60, 60), dtype=np.float32)
    probability[10:50, 10:50] = 0.9
    probability[28:31, 28:31] = 0.0  # a 3 x 3 = 9 px hole
    plain = decode_polygons(probability, PostprocessParams(min_area_px=0, simplify_px=0))
    assert len(plain) == 1 and len(plain[0]["rings"]) == 2
    filled = decode_polygons(probability, PostprocessParams(min_area_px=0, simplify_px=0, fill_holes_px=10))
    assert len(filled[0]["rings"]) == 1
    two = np.zeros((40, 40), dtype=np.float32)
    two[5:35, 5:18] = 0.9
    two[5:35, 20:33] = 0.9  # two blocks 2 px apart
    assert len(decode_polygons(two, PostprocessParams(min_area_px=0, simplify_px=0))) == 2
    assert len(decode_polygons(two, PostprocessParams(min_area_px=0, simplify_px=0, close_px=1))) == 1


def test_the_public_api_exports_the_augmentation_and_the_inference_halves_without_torch():
    import subprocess
    import sys

    import hs_hot

    for name in (
        "AugmentParams",
        "augment_chip_and_mask",
        "postprocess_outputs",
        "predict_chip",
    ):
        assert name in hs_hot.__all__ and hasattr(hs_hot, name)
    code = "import sys, hs_hot; assert 'torch' not in sys.modules"
    assert (
        subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
    )  # the CPU core stays torch-free

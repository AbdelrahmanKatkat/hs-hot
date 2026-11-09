"""Chips + labels as training pairs: scaled together, augmented together, and checked at construction."""

import numpy as np
import pytest

pytest.importorskip("torch")

from hs_hot.dataset import build_chip_labels  # noqa: E402
from hs_hot.params import PreprocessParams  # noqa: E402
from hs_hot.dataset import IGNORE  # noqa: E402
from hs_hot.segmentation_dataset import SegmentationDataset, pad_to_largest  # noqa: E402

PRE = PreprocessParams(
    model_gsd_m=0.1
)  # zoom 20 near the equator is 0.149 m/px: a 256 px chip scales to about 380 px


def test_a_chip_and_its_labels_come_back_at_the_same_scaled_size(labelled_chips):
    chips, labels, _ = labelled_chips
    dataset = SegmentationDataset(chips, labels, PRE)  # the GeoJSON path: the dataset builds the label shapes
    image, mask = dataset[0]
    assert (
        len(dataset) == 4 and image.shape[0] == 3 and image.shape[1:] == mask.shape and image.shape[1] > 256
    )
    assert str(image.dtype) == "torch.float32" and str(mask.dtype) == "torch.uint8"
    assert set(np.unique(mask.numpy())) == {0, 1}  # background and the road bands


def test_a_mapping_gives_the_same_pairs_as_the_path_and_a_missing_chip_has_no_labels(labelled_chips):
    chips, labels, names = labelled_chips
    from_path = SegmentationDataset(chips, labels, PRE)[0][1]
    from_mapping = SegmentationDataset(chips, build_chip_labels(chips, labels), PRE)[0][1]
    assert np.array_equal(from_path.numpy(), from_mapping.numpy())
    assert not SegmentationDataset(chips, {}, PRE)[0][1].numpy().any()


def test_augmentation_keeps_the_size_and_valid_mask_values(labelled_chips):
    chips, labels, _ = labelled_chips
    plain, augmented = (
        SegmentationDataset(chips, labels, PRE),
        SegmentationDataset(chips, labels, PRE, augment=True, seed=1),
    )
    for index in range(4):
        (image, mask), (turned_image, turned_mask) = plain[index], augmented[index]
        assert turned_image.shape == image.shape and turned_mask.shape == mask.shape
        assert sorted(turned_mask.flatten().tolist()) == sorted(
            mask.flatten().tolist()
        )  # the same pixels, moved


def test_names_select_chips_and_a_missing_name_is_an_error(labelled_chips):
    chips, labels, names = labelled_chips
    assert [path.name for path in SegmentationDataset(chips, labels, PRE, names=names[:2]).paths] == names[:2]
    with pytest.raises(ValueError, match="not found"):
        SegmentationDataset(chips, labels, PRE, names=["OAM-1-1-20.tif"])
    with pytest.raises(ValueError, match="No chips found"):
        SegmentationDataset(labels, labels, PRE)


def test_chips_of_different_sizes_are_stacked_with_ignored_padding():
    import torch

    small = (torch.ones(3, 4, 6), torch.ones(4, 6, dtype=torch.uint8))
    large = (torch.full((3, 8, 5), 0.5), torch.zeros(8, 5, dtype=torch.uint8))
    images, masks = pad_to_largest([small, large])
    assert images.shape == (2, 3, 8, 6) and masks.shape == (2, 8, 6) and masks.dtype == torch.uint8
    assert (
        (images[0, :, :4, :6] == 1).all()
        and (images[0, :, 4:, :] == 0).all()
        and (images[0, :, :, :].max() == 1)
    )
    assert (masks[0, :4, :6] == 1).all() and (masks[0, 4:, :] == IGNORE).all() and (masks[0, :, :] != 0).all()
    assert (masks[1, :, :5] == 0).all() and (
        masks[1, :, 5:] == IGNORE
    ).all()  # the labels stay on their pixels

"""Chips + labels as a torch dataset for U-Net fine-tuning (needs the `train` extra; import explicitly)."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .augment import augment_chip_and_mask
from .dataset import IGNORE, list_chip_paths, load_labels, scale_chip_and_labels
from .params import AugmentParams, PreprocessParams
from .preprocess import read_chip


def pad_to_largest(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack `(image, mask)` pairs of different sizes: pad each to the largest height and width of the batch.

    Chips of one dataset scale to different sizes when their ground resolutions differ (zoom 19 at 0.25 m and at 0.30 m).
    The image is padded with black on the bottom and right, and the mask with `IGNORE`, so the padding is left out of
    the loss and the scores and every label stays on its pixels.
    """
    height = max(image.shape[1] for image, _ in batch)
    width = max(image.shape[2] for image, _ in batch)
    images, masks = [], []
    for image, mask in batch:
        pad = (
            0,
            width - image.shape[2],
            0,
            height - image.shape[1],
        )  # left, right, top, bottom of the last two dims
        images.append(torch.nn.functional.pad(image, pad))
        masks.append(torch.nn.functional.pad(mask, pad, value=IGNORE))
    return torch.stack(images), torch.stack(masks)


class SegmentationDataset(torch.utils.data.Dataset):
    """Chips + their label shapes (`dataset.build_chip_labels`) as `(image, mask)` pairs.

    Each chip is scaled exactly as at serving time and its labels are drawn at that resolution
    (`scale_chip_and_labels`); with `augment` (True for the quarter turn and mirror only, or an `AugmentParams` that
    picks the steps), the pair is also changed by `augment_chip_and_mask`. The whole scaled chip is returned; chips of different
    projects scale to different sizes, so a batch is stacked with `pad_to_largest`.
    `labels` is the mapping from `build_chip_labels`, or the GeoJSON path to build it from; a chip missing from the
    mapping counts as having no labels.
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels: str | Path | Mapping[str, list[dict[str, Any]]],
        pre: PreprocessParams,
        *,
        names: Sequence[str] | None = None,
        augment: bool | AugmentParams = False,
        seed: int = 0,
    ) -> None:
        self.pre = pre
        self.augment = (
            AugmentParams(orientation="dihedral") if augment is True else (augment or None)
        )  # False: none
        self.rng = np.random.default_rng(seed)
        self.paths = list_chip_paths(images_dir)
        if names is not None:
            wanted = set(names)
            self.paths = [path for path in self.paths if path.name in wanted]
            missing = wanted - {path.name for path in self.paths}
            if missing:
                raise ValueError(
                    f"{len(missing)} named chips not found in {images_dir}, e.g. {sorted(missing)[:3]}"
                )
        if not self.paths:
            raise ValueError(f"No chips found in {images_dir}")
        self.labels = load_labels(images_dir, labels)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[idx]
        rgb, _, _ = read_chip(path)
        image, mask = scale_chip_and_labels(rgb, self.labels.get(path.name, []), self.pre, path.name)
        if self.augment is not None:
            image, mask = augment_chip_and_mask(image, mask, self.rng, self.augment)
        return torch.from_numpy(image), torch.from_numpy(mask)

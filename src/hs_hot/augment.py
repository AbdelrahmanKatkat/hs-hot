"""Training-time changes to a chip and its mask: orientation, scale jitter, brightness / contrast / gamma, blur or JPEG.

Used only on training chips (validation, test and serving never change a chip). Which steps run, and how strongly, is
set by `AugmentParams` (`params.py`); every step except the orientation is off until asked for, and the orientation
default is `none`. The image and the mask always change together, so a label stays on its pixels.

`augment_chip_and_mask` applies the steps in this order:

1. orientation: a random quarter turn and mirror (`dihedral`), a mirror only (`mirror`), or nothing (`none`);
2. scale jitter: resize by a random factor near 1 and crop or pad back to the same size;
3. photometric: brightness, contrast and gamma, on the image only;
4. degrade: a Gaussian blur or a JPEG round trip, on the image only.
"""

import io

import numpy as np
from PIL import Image, ImageFilter

from .dataset import IGNORE
from .params import AugmentParams


def _resize_and_fit(
    image: np.ndarray, mask: np.ndarray, factor: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Resize the pair by `factor` (image bilinear, mask nearest) and crop or pad back to the original size.

    A larger pair is cropped at a random place; a smaller one is padded with a reflection of the image and with `IGNORE`
    in the mask, so the padded pixels are neither rewarded nor punished.
    """
    height, width = mask.shape
    new_h, new_w = max(2, round(height * factor)), max(2, round(width * factor))
    resized = np.stack(
        [
            np.asarray(Image.fromarray(channel, mode="F").resize((new_w, new_h), Image.BILINEAR))
            for channel in image
        ]
    )
    resized_mask = np.asarray(Image.fromarray(mask).resize((new_w, new_h), Image.NEAREST))
    if factor >= 1.0:
        top, left = int(rng.integers(0, new_h - height + 1)), int(rng.integers(0, new_w - width + 1))
        return (
            resized[:, top : top + height, left : left + width],
            resized_mask[top : top + height, left : left + width],
        )
    pad_h, pad_w = height - new_h, width - new_w
    top, left = int(rng.integers(0, pad_h + 1)), int(rng.integers(0, pad_w + 1))
    spec = ((top, pad_h - top), (left, pad_w - left))
    return (
        np.pad(resized, ((0, 0), *spec), mode="reflect"),
        np.pad(resized_mask, spec, mode="constant", constant_values=IGNORE),
    )


def _photometric(image: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Random brightness, contrast (each x 1 +- `strength`) and gamma (exp(+- `strength`)) on an image in [0, 1]."""
    brightness, contrast = rng.uniform(1 - strength, 1 + strength, size=2)
    gamma = float(np.exp(rng.uniform(-strength, strength)))
    bright = image * brightness
    image = np.clip((bright - bright.mean()) * contrast + bright.mean(), 0.0, 1.0)
    return np.power(image, gamma)


def _degrade(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A Gaussian blur (radius 0.5 to 1.5 px) or a JPEG round trip (quality 40 to 90), chosen at random."""
    picture = Image.fromarray((image.transpose(1, 2, 0) * 255).round().astype(np.uint8))
    if rng.integers(0, 2):
        picture = picture.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 1.5))))
    else:
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=int(rng.integers(40, 91)))
        picture = Image.open(buffer).convert("RGB")
    return np.asarray(picture, dtype=np.float32).transpose(2, 0, 1) / 255.0


def augment_chip_and_mask(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, params: AugmentParams | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Change an image and its mask together (aerial imagery has no up); the default is a quarter turn and a mirror.

    Order: orientation (pixels only move), scale jitter (the mask is resized with nearest, so `IGNORE` is not blended),
    then the photometric and blur / JPEG steps, which touch the image only. The size never changes.
    """
    params = params or AugmentParams()
    if params.orientation != "none":
        turns = int(rng.integers(0, 4)) if params.orientation == "dihedral" else 0
        mirror = bool(rng.integers(0, 2))
        image, mask = np.rot90(image, turns, axes=(1, 2)), np.rot90(mask, turns, axes=(0, 1))
        if mirror:
            image, mask = image[:, :, ::-1], mask[:, ::-1]
    if params.scale_jitter > 0:
        factor = float(rng.uniform(1 - params.scale_jitter, 1 + params.scale_jitter))
        image, mask = _resize_and_fit(np.ascontiguousarray(image), np.ascontiguousarray(mask), factor, rng)
    if params.photometric > 0:
        image = _photometric(image, params.photometric, rng)
    if params.degrade_prob > 0 and rng.random() < params.degrade_prob:
        image = _degrade(image, rng)
    return np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(mask)

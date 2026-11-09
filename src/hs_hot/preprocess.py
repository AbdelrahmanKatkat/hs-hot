"""Chip -> model input: one scale step, nothing else.

Normalisation, padding to a multiple of 32 and everything after are left to the model (inside its ONNX graph),
so this module only decides the scale and resizes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .params import PreprocessParams
from .tiles import chip_scale


@dataclass(frozen=True)
class PreprocessMeta:
    """Everything needed to map a prediction back to chip pixels and then to the map."""

    width: int  # source chip, pixels
    height: int
    resized_width: int  # model input, pixels
    resized_height: int
    transform: Any
    crs: Any

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.width

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.height


def read_chip(image_path: str | Path) -> tuple[np.ndarray, Any, Any]:
    """Read bands 1-3 as uint8 HWC RGB plus the chip's affine transform and CRS.

    PNG/JPEG chips have no georeferencing: the transform is the identity and the CRS is `None`.
    """
    import rasterio

    with rasterio.open(image_path) as src:
        if src.count < 3:
            raise ValueError(f"{image_path}: expected >= 3 bands (RGB), found {src.count}")
        rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
        return np.clip(rgb, 0, 255).astype(np.uint8), src.transform, src.crs


def preprocess_rgb(
    rgb_hwc: np.ndarray, scale: float, *, transform: Any = None, crs: Any = None
) -> tuple[np.ndarray, PreprocessMeta]:
    """RGB uint8 chip -> `(3, H', W')` float32 in [0, 1] (CHW, no batch) plus `PreprocessMeta`.

    One bilinear resize in float, `H' = round(H * scale)`. The ONNX graph input has the same layout
    (`image`), and it applies the ImageNet normalisation itself.
    """
    if rgb_hwc.ndim != 3 or rgb_hwc.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB array, got shape {rgb_hwc.shape}")
    h, w, _ = rgb_hwc.shape
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    chw = rgb_hwc.astype(np.float32).transpose(2, 0, 1) / 255.0
    resized = np.empty((3, new_h, new_w), dtype=np.float32)
    for channel in range(3):
        resized[channel] = np.asarray(
            Image.fromarray(chw[channel]).resize((new_w, new_h), Image.Resampling.BILINEAR), dtype=np.float32
        )
    meta = PreprocessMeta(
        width=w, height=h, resized_width=new_w, resized_height=new_h, transform=transform, crs=crs
    )
    return resized, meta


def preprocess_chip(
    image_path: str | Path, cfg: PreprocessParams | None = None
) -> tuple[np.ndarray, PreprocessMeta]:
    """Read a chip named `OAM-<x>-<y>-<z>.<ext>`, derive the scale from its zoom, and preprocess it."""
    cfg = cfg or PreprocessParams()
    rgb, transform, crs = read_chip(image_path)
    scale = chip_scale(
        Path(image_path).name,
        rgb.shape[1],
        cfg.model_gsd_m,
        height_px=rgb.shape[0],
        max_input_px=cfg.max_input_px,
    )
    return preprocess_rgb(rgb, scale, transform=transform, crs=crs)

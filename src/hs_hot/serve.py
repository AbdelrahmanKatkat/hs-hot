"""Inference: directory of OAM chips -> scale -> ONNX -> road probability. Nothing is decoded or georeferenced here.

1. `preprocess_chip`: `model_gsd_m`, `max_input_px` (the scale).
2. the ONNX graph: no keys.

`predict_session` returns each chip's probability and metadata. The rest is the postprocess (`postprocess.py`,
`postprocess_outputs`): decode with the `threshold`, `smooth_sigma_px`, `open_px`, `close_px`, `min_area_px`,
`fill_holes_px` and `simplify_px` keys, then georeference to EPSG:4326. A fAIr pack's `predict` calls both.
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from .params import parse_preprocess_params
from .preprocess import PreprocessMeta, preprocess_chip

log = logging.getLogger(__name__)


def list_images(input_dir: str | Path) -> list[Path]:
    """List chip files in a directory (png/tif/jpg)."""
    root = Path(input_dir)
    patterns = ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg")
    files = sorted(path for pattern in patterns for path in root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No input images found in {root}")
    return files


def predict_chip(
    session: Any, image_path: str | Path, params: dict[str, Any] | None
) -> tuple[np.ndarray, PreprocessMeta]:
    """One chip, stages 1 and 2: scale it, run the ONNX session, and return its `(H', W')` road probability.

    The probability is in model-input pixels; the returned metadata maps it back to the chip. Nothing is decoded here.
    """
    image, meta = preprocess_chip(image_path, parse_preprocess_params(params))
    log.debug(
        "%s: scale %.3f -> input %dx%d",
        Path(image_path).name,
        meta.scale_x,
        meta.resized_width,
        meta.resized_height,
    )
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: image})[0], meta


def predict_session(
    session: Any, input_images: str | Path, params: dict[str, Any] | None
) -> list[tuple[Path, np.ndarray, PreprocessMeta]]:
    """Inference for every chip of a directory (`predict_chip` on each): `[(chip path, road probability, metadata), ...]`.

    Nothing is decoded and nothing is georeferenced here; see the module docstring for stages 3 and 4.
    """
    started = time.time()
    outputs = [(path, *predict_chip(session, path, params)) for path in list_images(input_images)]
    log.info("predict_session: %d chips in %.1fs", len(outputs), time.time() - started)
    return outputs

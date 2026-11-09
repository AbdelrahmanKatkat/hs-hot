"""Evaluation helpers for hs-hot: pixel precision / recall / F1 / IoU of the road class (torch only for `score_dataset`).

Scoring happens in model-input pixels, the resolution the network predicts at: the chip and its mask are scaled exactly
as in training (`scale_chip_and_labels`), so the numbers do not depend on how polygons are traced. `IGNORE` pixels count
for nothing. One prediction per chip gives the counts at every threshold at once.
"""

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import IGNORE, ROAD, list_chip_paths, load_labels, scale_chip_and_labels
from .params import PreprocessParams, parse_preprocess_params
from .preprocess import read_chip

log = logging.getLogger(__name__)


def pixel_counts(probability: np.ndarray, truth: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """`(len(thresholds), 3)` counts of true positive, false positive and false negative pixels at each threshold.

    `truth` holds `ROAD`, `IGNORE` and background; ignored pixels are left out of all three counts.
    """
    valid = truth != IGNORE
    is_road = (truth == ROAD) & valid
    is_background = ~is_road & valid
    counts = np.zeros((len(thresholds), 3), dtype=np.int64)
    for row, threshold in enumerate(thresholds):
        predicted = probability >= threshold
        counts[row] = (
            (predicted & is_road).sum(),
            (predicted & is_background).sum(),
            (~predicted & is_road).sum(),
        )
    return counts


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def pick_chips(paths: list[Path], max_chips: int | None) -> list[Path]:
    """At most `max_chips` chips, evenly spaced through the sorted list (so a pilot still spans the area)."""
    if not max_chips or max_chips >= len(paths):
        return paths
    step = len(paths) / max_chips
    return [paths[int(i * step)] for i in range(max_chips)]


def _chips(images_dir: str | Path, names: Sequence[str] | None, max_chips: int | None = None) -> list[Path]:
    paths = list_chip_paths(images_dir)
    if names is not None:
        wanted = set(names)
        paths = [path for path in paths if path.name in wanted]
    return pick_chips(paths, max_chips)


def _probability_torch(model: Any, image: np.ndarray) -> np.ndarray:
    """One preprocessed `(3, H, W)` image through the torch model: the same road probability as the ONNX gives."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install hs-hot with `[train]` extras to evaluate torch models.") from exc

    model.eval()
    device = next(model.parameters()).device  # runs wherever the model lives (CPU or GPU)
    with torch.no_grad():
        logits = model(torch.from_numpy(image).to(device)[None])
    return torch.sigmoid(logits)[0, 0].cpu().numpy()


def _score(
    paths: list[Path],
    labels: Mapping[str, list[dict[str, Any]]],
    predict: Callable[[np.ndarray], np.ndarray],
    pre: PreprocessParams,
    thresholds: Sequence[float],
) -> dict[float, dict[str, float]]:
    """P/R/F1/IoU over `paths` at each threshold, one `predict(image)` call per chip; logs one INFO line per threshold."""
    totals = np.zeros((len(thresholds), 3), dtype=np.int64)
    started = time.time()
    for path in paths:
        rgb, _, _ = read_chip(path)
        image, truth = scale_chip_and_labels(rgb, labels.get(path.name, []), pre, path.name)
        totals += pixel_counts(predict(image), truth, thresholds)
    results = {threshold: _prf(*row.tolist()) for threshold, row in zip(thresholds, totals)}
    for threshold, metrics in results.items():
        log.info(
            "evaluation on %d chips, threshold %.2f: precision=%.3f recall=%.3f f1=%.3f iou=%.3f in %.0fs",
            len(paths), threshold, metrics["precision"], metrics["recall"], metrics["f1"], metrics["iou"],
            time.time() - started,
        )  # fmt: skip
    return results


def score_dataset(
    model: Any,
    *,
    images_dir: str | Path,
    labels: str | Path | Mapping[str, list[dict[str, Any]]],
    params: dict[str, Any] | None = None,
    names: Sequence[str] | None = None,
    thresholds: tuple[float, ...] = (0.5,),
    max_chips: int | None = None,
) -> dict[float, dict[str, float]]:
    """Pixel metrics of a torch `model` on labelled chips, at each threshold (one pass per chip).

    `labels` is the mapping of `dataset.build_chip_labels` or the GeoJSON path to build it from; a chip missing from it
    counts as having no trees. `names` limits the scoring to those chips (for example a validation split of
    one chips folder); `params` carries `model_gsd_m` and `max_input_px`, as at serving time.
    """
    return _score(
        _chips(images_dir, names, max_chips),
        load_labels(images_dir, labels),
        lambda image: _probability_torch(model, image),
        parse_preprocess_params(params),
        thresholds,
    )


def score_session(
    session: Any,
    *,
    images_dir: str | Path,
    labels: str | Path | Mapping[str, list[dict[str, Any]]],
    names: Sequence[str] | None = None,
    params: dict[str, Any] | None = None,
    thresholds: tuple[float, ...] = (0.5,),
) -> dict[float, dict[str, float]]:
    """Pixel metrics through an ONNX session: the deployed path, with no torch involved.

    `names` limits the scoring to those chips (for example a validation split). This is what a pipeline step should
    report as the metrics of a model it just exported.
    """
    input_name = session.get_inputs()[0].name
    return _score(
        _chips(images_dir, names),
        load_labels(images_dir, labels),
        lambda image: session.run(None, {input_name: image})[0],
        parse_preprocess_params(params),
        thresholds,
    )


def sweep_threshold(
    model: Any,
    *,
    images_dir: str | Path,
    labels: str | Path | Mapping[str, list[dict[str, Any]]],
    names: Sequence[str] | None = None,
    thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    params: dict[str, Any] | None = None,
    max_chips: int | None = None,
) -> list[dict[str, float]]:
    """Precision / recall / F1 / IoU at each threshold, from one prediction per chip.

    Use it to choose the `threshold` of a freshly trained model: a fine-tuned model is not calibrated like another
    generation, so the default of one is not the best of the next.
    """
    scores = score_dataset(
        model, images_dir=images_dir, labels=labels, params=params, names=names,
        thresholds=thresholds, max_chips=max_chips,
    )  # fmt: skip
    rows = [{"threshold": threshold, **metrics} for threshold, metrics in scores.items()]
    best = max(rows, key=lambda row: row["f1"])
    log.info(
        "threshold sweep: best f1=%.3f (iou %.3f) at threshold %.2f",
        best["f1"],
        best["iou"],
        best["threshold"],
    )
    return rows

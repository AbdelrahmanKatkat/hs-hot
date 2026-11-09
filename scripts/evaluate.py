"""Entry point to `hs_hot.evaluate`: score a U-Net on labelled chips, with a sweep of the probability threshold.

    python scripts/evaluate.py --weights research/runs/B1/HIGHWAY_UNET.pt --chips research/data/highways/val/chips \
        --labels research/data/highways/val/labels --thresholds 0.3 0.4 0.5 0.6 0.7 --json metrics.json

Scores the torch weights (`--weights`) or the deployed ONNX (`--onnx`, no torch involved), in model-input pixels:
precision, recall, F1 and IoU of the road class, pixels marked ignore left out. Pick the threshold of a new model from the
sweep; do not reuse another model's. Needs the `train` extra for `--weights`.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from hs_hot.evaluate import score_session, sweep_threshold

log = logging.getLogger("evaluate")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--weights", type=Path, help="a U-Net state dict (.pt)")
    model.add_argument("--onnx", type=Path, help="an exported ONNX: scores the deployed path")
    parser.add_argument("--chips", required=True, type=Path)
    parser.add_argument(
        "--labels", required=True, type=Path, help="the labels GeoJSON of the chips, or its folder"
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    parser.add_argument("--model-gsd-m", type=float, help="metres per model pixel (must match training)")
    parser.add_argument("--max-chips", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json", type=Path, help="write all results here")
    return parser


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def _device(requested: str) -> str:
    import torch

    return "cpu" if requested == "cpu" or not torch.cuda.is_available() else "cuda"


def _score_weights(args: argparse.Namespace, params: dict[str, Any]) -> list[dict[str, float]]:
    from hs_hot.export import build_unet

    device = _device(args.device)
    model = build_unet(weights=args.weights).to(device)
    return sweep_threshold(
        model, images_dir=args.chips, labels=args.labels, thresholds=tuple(args.thresholds),
        params=params, max_chips=args.max_chips,
    )  # fmt: skip


def _score_onnx(args: argparse.Namespace, params: dict[str, Any]) -> list[dict[str, float]]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    scores = score_session(
        session, images_dir=args.chips, labels=args.labels, params=params, thresholds=tuple(args.thresholds)
    )
    return [{"threshold": threshold, **metrics} for threshold, metrics in scores.items()]


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    params = {"model_gsd_m": args.model_gsd_m} if args.model_gsd_m else {}
    rows = _score_weights(args, params) if args.weights else _score_onnx(args, params)
    best = max(rows, key=lambda row: row["f1"])
    log.info("best f1 %.3f (iou %.3f) at threshold %.2f", best["f1"], best["iou"], best["threshold"])
    return {"model": str(args.weights or args.onnx), "chips": str(args.chips), "sweep": rows, "best": best}


def main() -> None:
    args = _build_parser().parse_args()
    _configure_logging()
    results = _evaluate(args)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

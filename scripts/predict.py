"""Entry point to `hs_hot.predict_session` then `postprocess_outputs`: run an exported ONNX on a folder of chips and write GeoJSON polygons (CPU).

    python scripts/predict.py --onnx research/runs/B1/HIGHWAY_UNET.onnx --chips research/data/highways/test/chips \
        --out predictions.geojson --params '{"threshold": 0.5}'

Chips are named `OAM-<x>-<y>-<z>.<ext>`. Needs `onnxruntime` (the `infer` extra); no torch.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import onnxruntime as ort

from hs_hot import postprocess_outputs, predict_session


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--chips", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--params", default="{}", help='JSON, for example {"threshold": 0.5, "min_area_px": 400}'
    )
    return parser


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def _predict(args: argparse.Namespace) -> dict[str, Any]:
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    params = json.loads(args.params)
    return postprocess_outputs(predict_session(session, args.chips, params), params)


def _write_geojson(collection: dict[str, Any], out: Path) -> None:
    out.write_text(json.dumps(collection), encoding="utf-8")
    logging.getLogger("predict").info("wrote %d features to %s", len(collection["features"]), out)


def main() -> None:
    args = _build_parser().parse_args()
    _configure_logging()
    collection = _predict(args)
    _write_geojson(collection, args.out)


if __name__ == "__main__":
    main()

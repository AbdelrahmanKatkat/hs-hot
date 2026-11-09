"""Entry point to `hs_hot.export`: write a highway U-Net state dict as an ONNX file.

    python scripts/export_onnx.py research/runs/B1/HIGHWAY_UNET.pt HIGHWAY_UNET.onnx

Needs the `train` extra (torch and onnx); the output runs with `scripts/predict.py` or `hs_hot.predict_session`.
"""

import argparse
from pathlib import Path

from hs_hot.export import build_unet, export_onnx_bytes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("weights", type=Path, help="a U-Net state dict (.pt)")
    parser.add_argument("out", type=Path, help="the ONNX file to write")
    return parser


def _export(weights: Path, out: Path) -> None:
    out.write_bytes(export_onnx_bytes(build_unet(weights=weights)))


def main() -> None:
    args = _build_parser().parse_args()
    _export(args.weights, args.out)


if __name__ == "__main__":
    main()

"""Entry point to `hs_hot.train`: fine-tune the highway U-Net and write `<name>.pt` and `<name>.onnx`.

    python scripts/finetune.py --train-dir research/data/highways/train --val-dir research/data/highways/val \
        --encoder-weights weights/resnet50_imagenet1k_v2.pth --out research/runs/B1 --name HIGHWAY_UNET \
        --epochs 10 --batch-size 4

A dataset folder holds `chips/` (georeferenced GeoTIFFs named `OAM-<x>-<y>-<z>.tif`) and `labels/` (one GeoJSON of road centrelines).
Start a new U-Net from a torchvision ResNet-50 checkpoint with `--encoder-weights` (it initialises the encoder), or continue one with `--weights` (a
`<name>.pt` written by this script). Every training option that is left out keeps the default of
`hs_hot.params.TrainParams`. Needs the `train` extra; with a local CUDA torch use `--no-sync` / the venv python, not a
plain `uv run`.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

from hs_hot.export import export_onnx_bytes
from hs_hot.params import TrainParams, parse_preprocess_params, parse_train_params
from hs_hot.train import TrainResult, train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--val-dir", type=Path, help="enables validation loss and best-epoch selection")
    start = parser.add_mutually_exclusive_group()
    start.add_argument(
        "--encoder-weights",
        type=Path,
        help="a torchvision ResNet-50 state dict (resnet50-*.pth): a new U-Net",
    )
    start.add_argument("--weights", type=Path, help="a U-Net state dict (<name>.pt): continue training it")
    parser.add_argument("--out", required=True, type=Path, help="output folder; must be empty or new")
    parser.add_argument("--name", default="HIGHWAY_UNET", help="output file stem: <name>.pt and <name>.onnx")
    parser.add_argument("--model-gsd-m", type=float, help="metres per model pixel (must match serving)")
    parser.add_argument("--log-every", type=int, default=0, help="0 = about 50 step lines per epoch")
    parser.add_argument(
        "--max-val-chips", type=int, help="validate on this many evenly spaced chips (quick pilots)"
    )
    for flag, kind in (("epochs", int), ("batch-size", int), ("lr", float), ("weight-decay", float),
                       ("max-grad-norm", float), ("scheduler", str),
                       ("patience", int), ("device", str), ("seed", int), ("warmup-steps", int),
                       ("epoch-fraction", float), ("dice-weight", float), ("orientation", str),
                       ("photometric", float), ("scale-jitter", float), ("degrade-prob", float)):  # fmt: skip
        parser.add_argument(f"--{flag}", type=kind)
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--unfreeze-from",
        help="with a frozen encoder, still train ResNet stages from this one: layer4, layer3, ...",
    )
    parser.add_argument(
        "--encoder-lr-factor", type=float, help="learning-rate multiplier for the encoder weights that train"
    )
    return parser


def dataset_dirs(root: Path) -> tuple[Path, Path]:
    return root / "chips", root / "labels"


def _prepare_out_dir(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new --out")
    out.mkdir(parents=True, exist_ok=True)


def _configure_logging(out: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out / "train.log", encoding="utf-8"),
        ],
    )


def _train_params(args: argparse.Namespace) -> TrainParams:
    """Command-line options as the training hyperparameters; options left out keep the TrainParams default."""
    hyperparameters = {
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
        "weight_decay": args.weight_decay, "max_grad_norm": args.max_grad_norm, "scheduler": args.scheduler,
        "early_stop_patience": args.patience, "device": args.device,
        "seed": args.seed, "freeze_encoder": args.freeze_encoder,
        "warmup_steps": args.warmup_steps, "epoch_fraction": args.epoch_fraction,
        "unfreeze_from": args.unfreeze_from, "encoder_lr_factor": args.encoder_lr_factor,
        "dice_weight": args.dice_weight, "orientation": args.orientation, "photometric": args.photometric,
        "scale_jitter": args.scale_jitter, "degrade_prob": args.degrade_prob,
    }  # fmt: skip
    return parse_train_params(
        {key: value for key, value in hyperparameters.items() if value is not None}
    )  # None -> TrainParams default


def _run_training(args: argparse.Namespace) -> TrainResult:
    train_images, train_labels = dataset_dirs(args.train_dir)
    val_images, val_labels = dataset_dirs(args.val_dir) if args.val_dir else (None, None)
    return train(
        train_images_dir=train_images, train_labels=train_labels,
        val_images_dir=val_images, val_labels=val_labels,
        cfg=_train_params(args),
        pre=parse_preprocess_params({"model_gsd_m": args.model_gsd_m}),
        checkpoint=args.weights, encoder_checkpoint=args.encoder_weights,
        log_every=args.log_every, max_val_chips=args.max_val_chips,
        epoch_checkpoint_dir=args.out / "epochs",
    )  # fmt: skip


def _save_outputs(result: TrainResult, out: Path, name: str) -> None:
    """Weights (`.pt`), the ONNX export, the loss history, and one log line naming the files."""
    pt_path, onnx_path = out / f"{name}.pt", out / f"{name}.onnx"
    torch.save(result.model.state_dict(), pt_path)
    onnx_path.write_bytes(export_onnx_bytes(result.model))
    history: dict[str, Any] = {
        **result.history,
        "best_epoch": result.best_epoch + 1,
        "best_loss": result.best_loss,
    }
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    logging.getLogger("finetune").info(
        "wrote %s (%.1f MB) and %s (%.1f MB)",
        pt_path,
        pt_path.stat().st_size / 1e6,
        onnx_path,
        onnx_path.stat().st_size / 1e6,
    )


def main() -> None:
    args = _build_parser().parse_args()
    _prepare_out_dir(args.out)
    _configure_logging(args.out)
    result = _run_training(args)
    _save_outputs(result, args.out, args.name)


if __name__ == "__main__":
    main()

"""Constants, defaults and dict -> dataclass parsing for `hs-hot`.

Parsers tolerate `None` / missing keys. The encoder is a torchvision ResNet-50 pretrained on ImageNet, so the ImageNet
normalisation below is the one it was trained with. A DINOv3 encoder would change `IMAGE_MEAN` / `IMAGE_STD`, the
patch-size rule of the input and the encoder stages.
"""

from dataclasses import dataclass
from typing import Any, Mapping

# --- encoder constants (not stored in the checkpoint) ---
# ImageNet statistics (the torchvision transform default), which the pretrained encoder expects.
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

# --- scale policy ---
# The Guru highways chips are zoom 19: 0.30 m per pixel at Freetown and Nairobi (94% of the tiles), 0.25 m at Stellenbosch.
# The starting point is the native resolution, so a Freetown chip enters the model at 256 px (scale 1.0, a 6 m road band is
# 20 px wide) and a Stellenbosch chip is shrunk to 0.83. A finer scale (0.15 m, 2x) was tried and scored slightly lower
# (see Results in `docs/Theory.md`).
MODEL_GSD_M = 0.30
# Refuse inputs that would be upscaled beyond this (a 256 px chip would exceed 4096 px, i.e. a zoom
# that is far too coarse for this model). At the default 0.30 m, zoom 19 is left at about its native size and zoom 18 is upscaled
# about 2x; no zoom outside 19 has been measured.
MAX_SCALE = 16.0
# The longest side of a scaled chip never exceeds this many pixels (`PreprocessParams.max_input_px`). Without it
# the input size follows the zoom: a 256 px chip at zoom 18 (latitude 28) becomes 2,700 px and a CPU pod with a
# few GiB is OOM-killed. 1024 px leaves zoom 19 untouched (about 256 px) and bounds the memory of coarser zooms.
MAX_INPUT_PX = 1024

# How a training chip is turned before it reaches the model (see `AugmentParams`).
ORIENTATIONS = ("dihedral", "mirror", "none")

# ResNet-50 stages of the encoder, shallow to deep; `unfreeze_from` picks the first one that trains.
ENCODER_STAGES = ("layer1", "layer2", "layer3", "layer4")


@dataclass(frozen=True)
class SplitParams:
    val_ratio: float = 0.2
    split_seed: int = 42
    block_size: int = 4


@dataclass(frozen=True)
class PreprocessParams:
    """How a chip is scaled before the model. Training and serving must use the same value."""

    model_gsd_m: float = MODEL_GSD_M  # ground size of one model-input pixel, metres
    max_input_px: int = MAX_INPUT_PX  # longest side of the scaled chip; a coarser zoom is upscaled less


@dataclass(frozen=True)
class AugmentParams:
    """What changes a training chip (and its mask) before the model; validation and test chips are never changed.

    Every step is off until asked for, the orientation included.
    """

    orientation: str = (
        "none"  # "dihedral": random quarter turn + mirror; "mirror": mirror only; "none": unchanged
    )
    photometric: float = (
        0.0  # brightness, contrast (each x 1 +- this) and gamma (exp(+-this)) jitter; 0 = off
    )
    scale_jitter: float = 0.0  # random resize by 1 +- this, then crop or pad back to the same size; 0 = off
    degrade_prob: float = 0.0  # chance that a chip is blurred or JPEG-compressed (imitates coarser imagery)


@dataclass(frozen=True)
class PostprocessParams:
    """How the road probability becomes polygons (outside the ONNX graph, so all of it stays tunable)."""

    threshold: float = (
        0.3  # a pixel is road from this probability on (best on validation for the whole-encoder model, docs/Theory.md)
    )
    min_area_px: int = 400  # polygons smaller than this many model-input pixels are dropped (36 m2 at 0.30 m)
    simplify_px: float = 1.0  # Douglas-Peucker tolerance in model-input pixels (0 keeps every pixel edge)
    smooth_sigma_px: float = 0.0  # Gaussian smoothing of the probability before the threshold (0 = off)
    open_px: int = (
        0  # square opening radius on the mask before tracing: removes thin bridges and specks (0 = off)
    )
    close_px: int = 0  # square closing radius on the mask before tracing: fills thin gaps (0 = off)
    fill_holes_px: int = 0  # interior holes smaller than this many pixels are filled (0 = keep all)


@dataclass(frozen=True)
class TrainParams:
    """Fine-tuning settings. The defaults are the recipe that scored best on the validation tiles of the Guru highways.

    The whole ResNet encoder trains, at a tenth of the learning rate of the new decoder, so it adapts to roads without
    losing what it learned from ImageNet. The encoder's BatchNorm layers never train. Ten epochs that each use a random
    tenth of the chips add up to about one pass over the data. No augmentation is on by default.
    """

    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0  # 0 disables clipping
    scheduler: str = (
        "cosine"  # "cosine" | "none"; cosine decays to 0 over all steps of the run (stepped per batch)
    )
    warmup_steps: int = 0  # linear learning-rate ramp over the first batches (0 = none)
    freeze_encoder: bool = False  # the whole encoder trains (best on validation, docs/Theory.md); True keeps it frozen
    unfreeze_from: str = "none"  # with freeze_encoder: still train ResNet stages from here on ("layer4", "layer3", "none" = none)
    encoder_lr_factor: float = (
        0.1  # learning-rate multiplier for the encoder weights that do train (1.0 = same as the heads)
    )
    early_stop_patience: int = (
        5  # epochs without a better loss before stopping (>= epochs: never stops early)
    )
    epoch_fraction: float = (
        0.1  # each epoch trains on a fresh random fraction of the tiles (1.0 = all of them)
    )
    dice_weight: float = (
        1.0  # weight of the Dice loss next to the binary cross-entropy (0 = cross-entropy only)
    )
    orientation: str = AugmentParams.orientation  # see AugmentParams
    photometric: float = AugmentParams.photometric
    scale_jitter: float = AugmentParams.scale_jitter
    degrade_prob: float = AugmentParams.degrade_prob
    device: str = "auto"  # cuda if visible, else cpu (training is meant to run on a GPU)
    seed: int = 42  # shuffling and augmentation


def _get(mapping: Mapping[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    """First present non-None key wins (canonical key first, then aliases)."""
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return default


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("bool is not a valid int here")
    return int(value)


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("bool is not a valid float here")
    return float(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "y", "on"}:
            return True
        if raw in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Expected bool-like value, got {value!r}")


def _check_unit(name: str, value: float, *, low_open: bool = False) -> float:
    ok = (0.0 < value <= 1.0) if low_open else (0.0 <= value <= 1.0)
    if not ok:
        raise ValueError(f"{name} must be in {'(0, 1]' if low_open else '[0, 1]'}, got {value}")
    return value


def parse_split_params(hyperparameters: Mapping[str, Any] | None) -> SplitParams:
    given = hyperparameters or {}
    val_ratio = _check_unit(
        "val_ratio", _as_float(_get(given, ("val_ratio",), SplitParams.val_ratio)), low_open=True
    )
    block_size = _as_int(_get(given, ("block_size",), SplitParams.block_size))
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    return SplitParams(
        val_ratio=val_ratio,
        split_seed=_as_int(_get(given, ("split_seed",), SplitParams.split_seed)),
        block_size=block_size,
    )


def parse_preprocess_params(params: Mapping[str, Any] | None) -> PreprocessParams:
    """`model_gsd_m`: metres per model-input pixel (default 0.30, see `MODEL_GSD_M`).

    `max_input_px`: cap on the longest side of the scaled chip (default 1024), which bounds memory at coarse zooms.
    """
    given = params or {}
    model_gsd_m = _as_float(_get(given, ("model_gsd_m",), PreprocessParams.model_gsd_m))
    if not 0.005 <= model_gsd_m <= 1.0:
        raise ValueError(f"model_gsd_m must be in [0.005, 1.0] metres, got {model_gsd_m}")
    max_input_px = _as_int(_get(given, ("max_input_px",), PreprocessParams.max_input_px))
    if not 64 <= max_input_px <= 8192:
        raise ValueError(f"max_input_px must be in [64, 8192] pixels, got {max_input_px}")
    return PreprocessParams(model_gsd_m=model_gsd_m, max_input_px=max_input_px)


def parse_postprocess_params(params: Mapping[str, Any] | None) -> PostprocessParams:
    """`threshold` is canonical; `confidence_threshold` (the name of the other fAIr models) is an alias."""
    given = params or {}
    threshold = _check_unit(
        "threshold",
        _as_float(_get(given, ("threshold", "confidence_threshold"), PostprocessParams.threshold)),
    )
    min_area_px = _as_int(_get(given, ("min_area_px",), PostprocessParams.min_area_px))
    simplify_px = _as_float(_get(given, ("simplify_px",), PostprocessParams.simplify_px))
    smooth_sigma_px = _as_float(_get(given, ("smooth_sigma_px",), PostprocessParams.smooth_sigma_px))
    open_px = _as_int(_get(given, ("open_px",), PostprocessParams.open_px))
    close_px = _as_int(_get(given, ("close_px",), PostprocessParams.close_px))
    fill_holes_px = _as_int(_get(given, ("fill_holes_px",), PostprocessParams.fill_holes_px))
    for name, ok, value in (
        ("min_area_px", min_area_px >= 0, min_area_px),
        ("simplify_px", simplify_px >= 0, simplify_px),
        ("smooth_sigma_px", smooth_sigma_px >= 0, smooth_sigma_px),
        ("open_px", open_px >= 0, open_px),
        ("close_px", close_px >= 0, close_px),
        ("fill_holes_px", fill_holes_px >= 0, fill_holes_px),
    ):
        if not ok:
            raise ValueError(f"{name} out of range, got {value}")
    return PostprocessParams(
        threshold=threshold, min_area_px=min_area_px, simplify_px=simplify_px, smooth_sigma_px=smooth_sigma_px,
        open_px=open_px, close_px=close_px, fill_holes_px=fill_holes_px,
    )  # fmt: skip


def parse_train_params(params: Mapping[str, Any] | None) -> TrainParams:
    """Parse training knobs from a flat dict (fAIr strips the `training.` prefix)."""
    given = params or {}
    epochs = _as_int(_get(given, ("epochs",), TrainParams.epochs))
    batch_size = _as_int(_get(given, ("batch_size",), TrainParams.batch_size))
    learning_rate = _as_float(_get(given, ("learning_rate", "lr"), TrainParams.learning_rate))
    weight_decay = _as_float(_get(given, ("weight_decay",), TrainParams.weight_decay))
    patience = _as_int(_get(given, ("early_stop_patience",), TrainParams.early_stop_patience))
    epoch_fraction = _as_float(_get(given, ("epoch_fraction",), TrainParams.epoch_fraction))
    warmup_steps = _as_int(_get(given, ("warmup_steps",), TrainParams.warmup_steps))
    max_grad_norm = _as_float(_get(given, ("max_grad_norm",), TrainParams.max_grad_norm))
    scheduler = str(_get(given, ("scheduler",), TrainParams.scheduler)).strip().lower()
    # By default the whole encoder trains. Naming stages in `unfreeze_from` means a frozen encoder with only those
    # stages training (unless `freeze_encoder` says otherwise); `freeze_encoder=True` alone still trains `layer4`.
    # The encoder learns at `encoder_lr_factor` of the head's rate.
    unfreeze_given = _get(given, ("unfreeze_from",), None)
    partial = unfreeze_given is not None and str(unfreeze_given).strip().lower() != "none"
    freeze_encoder = _as_bool(_get(given, ("freeze_encoder", "freeze_backbone"), partial))
    unfreeze_default = "layer4" if freeze_encoder else "none"
    unfreeze_from = str(unfreeze_given if unfreeze_given is not None else unfreeze_default).strip().lower()
    encoder_lr_factor = _as_float(_get(given, ("encoder_lr_factor",), TrainParams.encoder_lr_factor))
    dice_weight = _as_float(_get(given, ("dice_weight",), TrainParams.dice_weight))
    orientation = str(_get(given, ("orientation",), TrainParams.orientation)).strip().lower()
    photometric = _as_float(_get(given, ("photometric",), TrainParams.photometric))
    scale_jitter = _as_float(_get(given, ("scale_jitter",), TrainParams.scale_jitter))
    degrade_prob = _as_float(_get(given, ("degrade_prob",), TrainParams.degrade_prob))
    for name, ok, value in (
        ("epochs", epochs >= 1, epochs),
        ("batch_size", batch_size >= 1, batch_size),
        ("learning_rate", learning_rate > 0, learning_rate),
        ("weight_decay", weight_decay >= 0, weight_decay),
        ("early_stop_patience", patience >= 1, patience),
        ("epoch_fraction", 0.0 < epoch_fraction <= 1.0, epoch_fraction),
        ("warmup_steps", warmup_steps >= 0, warmup_steps),
        ("max_grad_norm", max_grad_norm >= 0, max_grad_norm),
        ("scheduler", scheduler in ("cosine", "none"), scheduler),
        ("unfreeze_from", unfreeze_from in ENCODER_STAGES + ("none",), unfreeze_from),
        ("encoder_lr_factor", 0.0 < encoder_lr_factor <= 1.0, encoder_lr_factor),
        ("dice_weight", dice_weight >= 0, dice_weight),
        ("orientation", orientation in ORIENTATIONS, orientation),
        ("photometric", 0.0 <= photometric < 1.0, photometric),
        ("scale_jitter", 0.0 <= scale_jitter < 1.0, scale_jitter),
        ("degrade_prob", 0.0 <= degrade_prob <= 1.0, degrade_prob),
    ):
        if not ok:
            raise ValueError(f"{name} out of range, got {value}")
    if unfreeze_from != "none" and not freeze_encoder:
        raise ValueError(
            "unfreeze_from needs freeze_encoder=True (with freeze_encoder=False the whole encoder trains)"
        )
    return TrainParams(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        scheduler=scheduler,
        warmup_steps=warmup_steps,
        early_stop_patience=patience,
        freeze_encoder=freeze_encoder,
        unfreeze_from=unfreeze_from,
        encoder_lr_factor=encoder_lr_factor,
        epoch_fraction=epoch_fraction,
        dice_weight=dice_weight,
        orientation=orientation,
        photometric=photometric,
        scale_jitter=scale_jitter,
        degrade_prob=degrade_prob,
        device=str(_get(given, ("device",), TrainParams.device)),
        seed=_as_int(_get(given, ("seed",), TrainParams.seed)),
    )

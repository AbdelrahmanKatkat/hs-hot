"""`hs-hot`: HOT fAIr highway (road) semantic segmentation helpers (U-Net on an ImageNet ResNet-50 encoder)."""

from .augment import augment_chip_and_mask
from .dataset import build_chip_labels, spatial_split
from .decode import decode_polygons
from .params import (
    AugmentParams,
    PostprocessParams,
    PreprocessParams,
    SplitParams,
    TrainParams,
    parse_postprocess_params,
    parse_preprocess_params,
    parse_split_params,
    parse_train_params,
)
from .postprocess import polygons_to_feature_collection, postprocess_outputs
from .preprocess import PreprocessMeta, preprocess_chip, preprocess_rgb
from .serve import predict_chip, predict_session
from .tiles import chip_scale, tile_gsd_m

__all__ = [
    # parameters
    "PreprocessParams",
    "PostprocessParams",
    "SplitParams",
    "TrainParams",
    "AugmentParams",
    "parse_preprocess_params",
    "parse_postprocess_params",
    "parse_split_params",
    "parse_train_params",
    # tiles and dataset
    "tile_gsd_m",
    "chip_scale",
    "build_chip_labels",
    "spatial_split",
    # training-time augmentation (every step is off by default)
    "augment_chip_and_mask",
    # preprocess -> decode -> postprocess
    "PreprocessMeta",
    "preprocess_rgb",
    "preprocess_chip",
    "decode_polygons",
    "polygons_to_feature_collection",
    "postprocess_outputs",
    # inference: `predict_session` stops at the probability; the caller decodes and georeferences
    "predict_chip",
    "predict_session",
]
# Optional (train extra, import explicitly): hs_hot.export, hs_hot.unet, hs_hot.segmentation_dataset, hs_hot.train, hs_hot.evaluate

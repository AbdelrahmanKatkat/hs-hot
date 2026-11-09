# `hs-hot`

Python helper library for the HOT **highway (road) semantic segmentation** model on fAIr: a U-Net decoder on a ResNet-50 encoder
pretrained on ImageNet (torchvision weights). This library holds the model logic — preprocessing, decoding, georeferencing, training and
export — and a fAIr model pack (`fAIr-models/models/<pack>/`) stays a thin STAC/ZenML/Docker adapter on top of it. A DINOv3 encoder is the
planned next backbone; the migration is described in [`docs/Theory.md`](docs/Theory.md) section 10.

**Python:** >= 3.12. Design decisions, the contract and the experiments are in [`docs/Theory.md`](docs/Theory.md).

## Inference path

```text
chip dir (OAM-<x>-<y>-<z>.tif/.png/.jpg, RGB)
  -> preprocess_chip()           one resize; scale = chip ground resolution (from the zoom and tile row) / 0.30 m
  -> onnxruntime session.run()   ONNX: normalise, U-Net, sigmoid -> probability (H, W), any size
  -> decode_polygons()           threshold (0.3), trace the pixel edges, drop small polygons, simplify; optional smoothing,
                                 opening/closing and hole fill are off by default
  -> polygons_to_chip_pixels()   back to chip pixels
  -> polygons_to_feature_collection()   EPSG:4326 polygons (the road bands, holes kept)
```

Inference: `hs_hot.predict_session(session, input_images, params)` returns each chip's road probability and metadata (scale + ONNX, nothing
decoded); `hs_hot.postprocess_outputs(outputs, params)` turns that into a GeoJSON `FeatureCollection` (`confidence`, `class_id`,
`class_name` = `highway`, `source` per feature). A fAIr pack's `predict` calls the two in turn. All `params` are optional; see
`docs/Theory.md` section 6.

**Chip names must be `OAM-<x>-<y>-<z>.<ext>`.** The zoom and tile row give the true ground resolution, so the scale is right even for
PNG/JPEG chips with no georeferencing (their polygons stay in chip pixels). A zoom that would need an upscale above 16x is rejected, and the
scaled chip is capped at `max_input_px` (default 1024) on its longest side. The output is a **polygon band per road**, about 6 m wide as the
labels are, not a centreline; a centreline output is not built (Theory section 1). Polygons are traced per chip, so a road cut by a chip
border comes back as two polygons.

## Install, extras and who installs what

| Extra | Contents | Who installs it |
|---|---|---|
| *(none)* | numpy, Pillow, pyproj, rasterio. **No torch.** | the CPU inference image (plus `onnxruntime`, which the fAIr serving base provides) |
| `train` | torch, torchvision, onnx: fine-tuning, evaluation and ONNX export | the fAIr **GPU** training image |
| `infer` | onnxruntime | running an exported ONNX locally (`scripts/predict.py`) |
| `test` | onnxruntime, pytest | CI |
| `dataprep` | datasets (Hugging Face) | **this repository only** (`research/`); fAIr-models must never install it |

Training always runs on a GPU, but the CUDA build of torch is chosen by the image (for example
`--extra-index-url https://download.pytorch.org/whl/cu126`), not by this package's metadata. In this workspace the development environment
is the shared `HOTOSM/.venv` (Python 3.12, CUDA torch, every `*-hot` library installed editable); use its Python directly (`--no-sync`), not a
plain `uv run`, so the CUDA build is kept.

A fAIr model pack installs the library from a pinned release tag, for example
`uv pip install "hs-hot[train] @ git+https://github.com/<owner>/hs-hot@<tag>"` in the training stage and
`"hs-hot @ git+https://github.com/<owner>/hs-hot@<tag>"` in the inference stage. The repository has to be reachable without credentials,
and the tag must exist. No repository or tag exists yet.

ONNX I/O: input `image (3, H, W)` float32 in [0, 1] with dynamic `H`, `W`; output `probability (H, W)` float32 in [0, 1]. Normalisation, the
network and the sigmoid are inside the graph; the threshold and the polygon steps are in `decode_polygons`.

## Layout

| Module | Role |
|---|---|
| `params.py` | defaults + `parse_*` (tolerate `None`, aliases `confidence_threshold`, `lr`) |
| `preprocess.py` | the one scale step, `PreprocessMeta` |
| `tiles.py` | OAM chip name -> ground resolution -> scale (zoom and latitude, no CRS) |
| `decode.py` | probability -> polygons in model pixels: threshold, trace, minimum area, simplify (no CRS) |
| `georef.py`, `postprocess.py` | model -> chip pixels -> EPSG:4326, GeoJSON; `postprocess_outputs` (probabilities -> decode -> GeoJSON) is the ready-to-use postprocess |
| `serve.py` | `predict_session`: chips in, road probability + metadata out (scale + ONNX only; no postprocessing); `predict_chip` is one chip |
| `dataset.py` | `spatial_split`; `build_chip_labels` (a labels GeoJSON of centrelines -> road shapes in each chip's pixels); `scale_chip_and_labels` (draw the bands at the model's scale) |
| `augment.py` | the training-only changes to a chip and its mask: orientation, scale jitter, brightness / contrast / gamma, blur or JPEG; each is a training parameter and off by default (`AugmentParams`) |
| `unet.py` | `HighwayUNet`: ResNet-50 encoder, U-Net decoder, normalisation inside (`train`) |
| `export.py` | `build_unet` (owns the checkpoints), `probability_network`, the ONNX export (`train`) |
| `segmentation_dataset.py`, `train.py` | chips + labels as a torch dataset; fine-tuning (`train`) |
| `evaluate.py` | pixel precision / recall / F1 / IoU, threshold sweep (torch only for `score_dataset`) |

`unet`, `export`, `segmentation_dataset` and `train` need the `train` extra and are imported explicitly (`from hs_hot.train import ...`);
`import hs_hot`, `serve`, `evaluate`, `decode` and `export` never load torch until a function that needs it is called.

## Scripts

`scripts/` holds thin command-line entry points; each one only parses arguments and calls `hs_hot`.

| Script | Calls | Extra |
|---|---|---|
| `scripts/finetune.py` | `hs_hot.train` + `export`: fine-tunes, writes `<name>.pt` and `<name>.onnx`, `train.log`, `history.json`, one checkpoint per epoch | `train` |
| `scripts/evaluate.py` | `hs_hot.evaluate`: pixel P/R/F1/IoU and a threshold sweep of torch weights or of the deployed ONNX | `train` (`--weights`) or `infer` (`--onnx`) |
| `scripts/predict.py` | `hs_hot.predict_session` then `postprocess_outputs`: an ONNX over a folder of chips to GeoJSON, on CPU | `infer` |
| `scripts/export_onnx.py` | `hs_hot.export`: a `<name>.pt` to an ONNX file | `train` |

A dataset folder for `finetune.py` holds `chips/` (georeferenced GeoTIFFs named `OAM-<x>-<y>-<z>.tif`) and `labels/` (one GeoJSON of road
centrelines): the layout a fAIr pack receives. `research/analysis/export_fair_layout.py` builds it for the Guru highways dataset from the
parquet files (local, not committed), and `research/analysis/check_label_masks.py` compares the masks the library draws with the dataset's own.

## Labels

Masks are not files: `build_chip_labels` turns the GeoJSON into road shapes in each chip's pixels, and the training and scoring code draws
them at the model's resolution. A `LineString` (or each part of a `MultiLineString`) is a centreline buffered by 3 m on each side, so a road is
a band 6 m wide, the way the Guru highways masks are drawn; the band width is converted to pixels at the chip's true ground resolution, and
bends and ends are round. A road just outside a chip still draws its band into it. Points and polygons are skipped. Chips must be
georeferenced and named `OAM-<x>-<y>-<z>`. On 1,296 evenly spaced chips of the three splits, the masks drawn this way match the dataset's own
at pixel IoU 0.952, drawing 2.4% fewer road pixels (Theory section 2).

## Defaults and results (the Guru highways)

The dataset is `hotosm/fAIr-guru-td-highways`: 12,880 tiles of 256 px at zoom 19 (0.25 to 0.30 m per pixel), split by region into train
10,253, validation 1,717 and test 910, from three Tasking Manager projects (Freetown, Nairobi, Stellenbosch). The defaults below are the
recipe that won on validation (Theory section 13):

| Setting | Default | Note |
|---|---|---|
| Encoder weights | torchvision ResNet-50, ImageNet V2 (`weights/resnet50_imagenet1k_v2.pth`) | V1 weights scored the same (0.437 against 0.438) |
| Scale | `model_gsd_m=0.30` | the native resolution: a Freetown chip enters at 256 px; a 2x finer scale was slightly worse and slower |
| Encoder training | the whole encoder trains at a tenth of the head's rate (`freeze_encoder=false`, `encoder_lr_factor=0.1`) | BatchNorm of the encoder stays in eval mode (not measured); `freeze_encoder=true` freezes all of it; `unfreeze_from=layer3` (or `layer4`) freezes the encoder except those stages and the deeper ones, and gave the same score within noise (`layer3`) |
| Optimiser | AdamW 1e-3, weight decay 1e-4, cosine, clip 1.0, Dice weight 1.0 | carried over as a starting point, not swept on this dataset |
| Length | `epochs=10`, `epoch_fraction=0.1`, `batch_size=4` | ten epochs of a random tenth of the chips are one pass over the data; 20 epochs did not help |
| Augmentation | none (`orientation=none`, `photometric=0`, `scale_jitter=0`, `degrade_prob=0`) | rotation and mirror did not help on a frozen encoder |
| Inference threshold | 0.3 | best on validation for this model (0.2 to 0.4 are within 0.003 IoU) |

Scores of the final model, pixel level, road class, threshold 0.3 (the labels are 6 m bands drawn from OSM centrelines, many minor roads
are unmapped, so these are lower bounds on how well roads are found):

| Split | Chips | Precision | Recall | F1 | IoU |
|---|---|---|---|---|---|
| train | 10,253 | 0.668 | 0.693 | 0.680 | 0.515 |
| validation | 1,717 | 0.613 | 0.659 | 0.635 | 0.465 |
| test | 910 | 0.637 | 0.700 | 0.667 | 0.500 |

## From a pipeline step (what a fAIr-models pack calls)

```python
from hs_hot import build_chip_labels, parse_split_params, parse_preprocess_params, spatial_split
from hs_hot.params import parse_train_params
from hs_hot.train import train                    # `train` extra, imported inside the step
from hs_hot.export import export_onnx_bytes
from hs_hot.evaluate import score_session         # no torch: scores the exported ONNX

split = parse_split_params(hyperparameters)       # val_ratio, split_seed, block_size
labels = build_chip_labels(chips_dir, labels_geojson)  # chip name -> road shapes in chip pixels
train_names, val_names = spatial_split(sorted(labels), split.val_ratio, split.split_seed, block_size=split.block_size)
result = train(
    train_images_dir=chips_dir, train_labels=labels, train_names=train_names,
    val_images_dir=chips_dir, val_labels=labels, val_names=val_names,
    cfg=parse_train_params(hyperparameters), pre=parse_preprocess_params(hyperparameters),
    checkpoint=base_weights, on_epoch_end=lambda epoch, train_loss, val_loss: ...)  # -> MLflow
onnx_bytes = export_onnx_bytes(result.model)
metrics = score_session(onnx_session, images_dir=chips_dir, labels=labels, names=val_names)[0.5]
```

`checkpoint` is a path or an open binary stream (for example `io.BytesIO(weights_bytes)`) of a U-Net state dict; to start a new U-Net from a
torchvision ResNet-50 checkpoint (`resnet50-*.pth`) give `encoder_checkpoint` instead. At serving time the pack's `predict` calls `hs_hot.predict_session(session, input_images, params)` (inference only) and then
its `postprocess`, which is `hs_hot.postprocess_outputs(outputs, params)` (decode, then georeference) with the properties renamed.

Each step rebuilds the labels from the chips and the GeoJSON it was given, so nothing is written or shared between isolated steps.

## Tests

```bash
ruff check src scripts tests
python -m pytest -q                   # offline: toy chips, random weights, a fake ONNX session
```

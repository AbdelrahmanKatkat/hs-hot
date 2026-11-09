# Theory: highway (road) semantic segmentation with a ResNet-50 U-Net

The design contract of the `hs-hot` library: what the model takes in, what it returns, and why each choice was made.

## 0. Summary

| Item | Value |
|---|---|
| Task / geometry | Binary semantic segmentation (`semantic-segmentation`) of roads; the output is **polygons**, one per connected road region (a band about 6 m wide); holes are kept, so a block enclosed by roads is not filled in |
| Model | A U-Net: a torchvision ResNet-50 pretrained on ImageNet (V2 weights) as encoder, a new decoder (`HighwayUNet`, section 4), 32.5 million parameters |
| Input | 3-band RGB, scaled by the zoom rule (section 3) to `model_gsd_m = 0.30` m per pixel, the native scale of 94% of the tiles |
| Output | A per-pixel road probability at the input size; threshold, tracing and cleanup are outside the graph |
| Labels | OSM `highway=*` centrelines in one GeoJSON, drawn per chip as a band 6 m wide at the model's resolution (section 2) |
| Dataset | `hotosm/fAIr-guru-td-highways`: 12,880 tiles, zoom 19, already split by region (section 10) |
| Runtime | Training on a GPU with torch; inference on a CPU with `onnxruntime`, no torch |
| Result (2026-11-09) | Whole encoder fine-tuned at a tenth of the head's rate, threshold 0.3: pixel **IoU 0.465 / F1 0.635 on validation, IoU 0.500 / F1 0.667 on test** (test scored once, one seed; section 12) |
| Status | Library, tests (139 passing), dataset export, mask check, tuning trials and the final scores done. Not done: a pack, a release tag, object-level scores, a DINOv3 backbone |

## 1. Task and output contract

- **Task:** every pixel of an RGB chip is road or background. Roads that touch merge into one polygon.
- **Feature schema** (`postprocess_outputs`): a `FeatureCollection` of EPSG:4326 `Polygon` features, holes kept, with the properties
  `confidence` (the mean road probability over the polygon's pixels), `class_id` (int, always 1), `class_name` (`highway`) and `source` (the
  chip file name). A fAIr pack renames them to its own schema.
- **Polygons, not lines.** The labels are centrelines but the dataset's masks (and so the training target) are 6 m bands, so the model
  returns the bands. fAIr accepts `polygon`, `line` or `point` outputs; a centreline output would need a skeleton step after decoding, which is
  not built. Polygon-per-chip tracing also cuts a road at every chip border.
- **One class.** The dataset has 16 `highway` values (`residential` 66.8%, `service` 7.6%, `path` 7.3%, ...); they are merged into one road
  class here. A per-class model is possible with the same code and a multi-channel output; it is not built.
- **Empty prediction:** an empty `FeatureCollection`. A chip without georeferencing (PNG, JPEG) returns polygons in chip pixels.

## 2. Labels and masks

A fAIr pack gets chips (`OAM-<x>-<y>-<z>.tif`, georeferenced) and one GeoJSON of labels; it does not get masks. `build_chip_labels` turns the
GeoJSON into road shapes in each chip's pixels, nothing is copied or written, and the mask is drawn on the fly at the model's resolution
(`scale_chip_and_labels`), so its edges are as sharp as the image.

| Step | Where | Result |
|---|---|---|
| 1 | `read_chip` (`preprocess.py`) | the RGB chip, `uint8`, height x width x 3 |
| 2 | `build_chip_labels` (`dataset.py`) | per chip, the centrelines whose band touches it, in chip pixels, each with its band width in pixels |
| 3 | `scale_chip_and_labels` (`dataset.py`) | image `(3, H', W')` float32 in [0, 1] at the model's scale and a mask `(H', W')` `uint8` with 0 background, 1 road, 255 ignore (padding only) |
| 4 | `augment_chip_and_mask` (`augment.py`), training chips only | the same pair, changed together by the steps that are switched on (section 9) |
| 5 | `SegmentationDataset` and a torch `DataLoader` | batches |

- **The band.** A `LineString` (a `MultiLineString` gives one shape per part) is drawn `width_px = 2 * 3 m / chip ground size` wide with
  Pillow (`ImageDraw.line` plus a disc on every vertex, so bends and ends are round). The dataset's own masks buffer each line by 3 m on
  each side, so the library does the same. No geometry library is used.
- **Roads outside the chip.** A centreline within 10 m of the chip's extent is kept and drawn if its band reaches in, so the road just
  beyond a border still shows its edge. The dataset does the same, so a few tiles with `num_features` 0 still have road pixels in
  their mask.
- **Check (run on 2026-11-05).** `research/analysis/check_label_masks.py` compares the mask drawn from `labels.geojson` with the dataset's own
  mask PNG on about 430 evenly spaced chips of each split: pixel IoU **0.952 (train), 0.952 (val), 0.952 (test)**, median per chip 0.952, 10th
  percentile 0.93; the library draws 2.4% fewer road pixels (2.33 M against 2.39 M on train). The gap is the width of the band (the dataset
  buffers in a projected CRS, the library converts metres to pixels with the tile's ground size) and the pixel rounding of thin lines; it
  is small against the effect of the scale and the loss, and no correction is applied for it.
- **Weak labels.** OSM is incomplete: projects 40695 and 50793 asked mappers to connect the existing network and add missing minor roads, so
  tracks and footpaths visible in their imagery are unlabelled; project 60363 (Freetown, 94% of the tiles) asked for all visible roads. The
  6 m band is wider than a footpath and narrower than a trunk road. A correct road nobody mapped counts as an error, so
  scores are bounded by this and must be reported with it.
- **Ignore.** No label draws the ignore value (255); only scale-jitter padding does. The loss and the scores already skip it, so it stays
  available for a later rule (for example ignoring unlabelled projects).

## 3. Preprocessing: scale from the zoom level

Every OAM chip is named `OAM-<x>-<y>-<z>.<ext>`. The tile row `y` and the zoom `z` give the chip's true ground resolution without reading
the file itself, so PNG/JPEG chips with no georeferencing scale the same way as GeoTIFFs. The ResNet-50 encoder has no ground scale of its
own (section 4), so there is no "native" input size for it: chips are resized once, by a fixed rule, to whatever resolution the recipe is
tuned at, and the same rule runs at training and at serving time (`preprocess.py`, `tiles.py`):

```
chip_gsd_m = 40075016.68558 * cos(lat_of_tile_centre) / 2^z / chip_width_px     # true ground size of a chip pixel
scale      = chip_gsd_m / model_gsd_m                                              # model_gsd_m = 0.30 (default)
scale      = min(scale, max_input_px / longest chip side)                         # max_input_px = 1024
input size = round(chip size * scale)                                              # one bilinear resize, in float
```

**Why 0.30 m.** The Guru highways chips are zoom 19: 0.30 m per pixel at Freetown and Nairobi (94% of the tiles with Freetown alone) and
0.25 m at Stellenbosch. The encoder is an ImageNet ResNet-50, which has no ground scale of its own, so there is no
reason to resample: the starting point is the native resolution, scale 1.0 for Freetown (a 256 px chip, a 6 m band 20 px wide) and 0.83
for Stellenbosch. A finer scale was tried and did not help (section 12, trial T5: 0.15 m, 2x upsampling, IoU 0.430 against 0.438 and about
1.8 times the training time). The first default of this library, 0.15 m, was carried over from the tree model, whose NEON encoder was
trained at 0.05 m; that reasoning does not apply here.

The same parameter, `model_gsd_m`, must be used for training, scoring and serving. At zoom 19 and 0.30 m the scaled chip stays far below
the 1024 px cap; zoom 18 would scale about 2 times and is not measured.

## 4. Model and ONNX contract

`HighwayUNet` pairs a torchvision ResNet-50 encoder with a five-stage U-Net decoder written for this library, and is exported to a single
ONNX graph that a CPU serving image can run with no torch installed:

- **Encoder:** torchvision ResNet-50 (stem, `layer1` to `layer4`, strides 2, 4, 8, 16, 32), loaded from the ImageNet V2 checkpoint
  `resnet50-*.pth` (the `fc` layer is dropped). The V1 checkpoint scores the same (section 12).
- **Decoder:** five up-blocks (256, 128, 64, 32, 16 channels), each upsampling to the size of the skip features, joining them and two 3x3
  convolutions with BatchNorm; a 1x1 convolution gives the logit.
- **ONNX (opset 17, legacy exporter):** input `image` `(3, H, W)` float32 in [0, 1] RGB with dynamic `H`, `W`; output `probability` `(H, W)`
  float32 in [0, 1]. Normalisation (ImageNet), the network and the sigmoid are inside the graph.
- The tunable steps (threshold, tracing, minimum area, simplification) are outside the graph.
- **Why ImageNet.** It is a public starting point with no tie to another task The licence of the ImageNet-trained weights has not been
  reviewed. An encoder pretrained on tree crowns (NEON forestry imagery) could be used instead; it has not been tried.

## 5. Inference and postprocessing

1. `preprocess_chip` reads the RGB chip and scales it by the zoom rule. Normalisation is inside the model.
2. The ONNX graph runs once: `image (3, H', W')` -> `probability (H', W')`.
3. `decode_polygons` turns the probability into polygons in model pixels, each with a `confidence`.
4. `polygons_to_chip_pixels` divides the vertices by the scale, giving chip pixels.
5. `polygons_to_feature_collection` applies the chip's affine transform, converts to EPSG:4326 and writes the properties of section 1.

`predict_session` is steps 1 and 2 (it returns each chip's probability and metadata); `postprocess_outputs` (`postprocess.py`) is steps 3 to
5 and returns the FeatureCollection. A fAIr pack's `predict` calls the two in turn, and `run_inference` calls `predict`. Decoding is in
model space (no CRS), georeferencing in map space (no network layout).

For roads the decoding defaults have not been tuned: `min_area_px = 400` is 36 m2 at 0.30 m (a 6 m band 6 m long), a small fragment
survives while a thin footpath band is only a few pixels wide and a threshold that suits a road may break it. `simplify_px = 1` keeps the
band outline; a long thin polygon has many vertices. A sweep on validation chips (section 12) would choose them.

## 6. Parameters

Keys are bare; fAIr strips the `training.` and `inference.` prefixes. `parse_*` tolerates `None` and missing keys. Defaults are in
`params.py`; they are the recipe that won on validation (section 12).

| Key | Default | Meaning |
|---|---|---|
| `model_gsd_m` | 0.30 | ground size of a model-input pixel, metres; range [0.005, 1] (section 3) |
| `max_input_px` | 1024 | cap on the longest side of the scaled chip |
| `threshold` (alias `confidence_threshold`) | 0.3 | a pixel is road from this probability on; best on validation for the final model |
| `min_area_px` | 400 | polygons smaller than this many model-input pixels are dropped |
| `simplify_px` | 1.0 | Douglas-Peucker tolerance in model-input pixels |
| `smooth_sigma_px`, `open_px`, `close_px`, `fill_holes_px` | 0 | optional cleanups, off |
| `epochs`, `batch_size`, `learning_rate` | 10, 4, 1e-3 | training; AdamW, cosine, clip 1.0, weight decay 1e-4 |
| `freeze_encoder`, `unfreeze_from`, `encoder_lr_factor` | false, `none`, 0.1 | the whole encoder trains at a tenth of the head's rate; `freeze_encoder=true` freezes it except `layer4` (or the stages `unfreeze_from` names); encoder BatchNorm stays in eval mode |
| `epoch_fraction` | 0.1 | each epoch uses a fresh random tenth of the chips |
| `dice_weight` | 1.0 | Dice term next to the binary cross-entropy |
| `orientation`, `photometric`, `scale_jitter`, `degrade_prob` | none, 0, 0, 0 | augmentation, all off (section 9) |
| `val_ratio`, `split_seed`, `block_size` | 0.2, 42, 4 | the pack's own spatial split (section 7) |

## 7. Spatial split

The Guru highways dataset is **already split by region** (section 10), so for research the three given splits are used as they are
(`research/data/highways/{train,val,test}`). The library's `spatial_split` (whole `(x // K, y // K)` blocks go to one side, `K = 4` tiles, about
300 m at zoom 19) is for a fAIr pack that is given one dataset and must split it itself; neighbouring chips share road continuations, so a
random chip split would leak. Tiles at the edge of a region border tiles of another split, and a road crossing an edge has parts in both; the
dataset card says so, and it makes the validation and test scores slightly optimistic.

## 8. Verification and open questions

**Checked:** the library compiles and its 139 offline tests pass in the shared `HOTOSM/.venv` (torch 2.14.1+cu126, GPU visible);
the dataset export reproduces the card's counts (12,880 tiles, 9,784 with roads; train 10,253 / validation 1,717 / test 910); the drawn masks
match the dataset's at pixel IoU 0.952 (section 2); the trials and final scores of section 12.

**Not checked yet:**

- the postprocess values on roads (`min_area_px`, `simplify_px`, closing of gaps), which were not swept;
- whether a trainable encoder BatchNorm helps (it is held in eval mode by default; no trial, the switch was removed);
- whether a tree-detector encoder or DINOv3 beats ImageNet initialisation;
- the padding test (a chip whose size is not a multiple of 32), a self-intersection check on the polygons, a hand-checked sample of test
  chips, an object-level (per road) score, a tolerance (buffered) pixel score, a ground-truth check of the band width, and zoom 18 behaviour;
- how much of the error is unlabelled roads (project 60363 vs the other two) rather than model error, and a second seed.

## 9. Augmentation

`augment.py` holds four independently switchable, training-only steps, run in this order on the image and its mask together so the pair
always stays aligned and the `IGNORE` value is never blended: orientation (`dihedral` quarter turn + mirror, `mirror` only, or `none`),
scale jitter (resize by a random factor, then crop or pad back), brightness / contrast / gamma, and blur or JPEG recompression. Every step
defaults to off; validation and test chips are never augmented regardless of the setting.

On roads, orientation (`dihedral`) was tried on the frozen-encoder recipe and scored 0.434 against the baseline's 0.438 (section 12, T4), so
it stays off; it was not tried together with a trainable encoder, and the other three steps were not tried on this dataset at all.


## 10. Dataset: the Guru highways (`hotosm/fAIr-guru-td-highways`)

Downloaded on 2026-11-05 to `research/data/fAIr-guru-td-highways` (revision `3fcd42e0`).

| | |
|---|---|
| Tiles | 12,880 of 256 x 256 px, zoom 19 (0.25 to 0.30 m per pixel), 72.67 km2 |
| Splits | train 10,253, validation 1,717, test 910: 10 contiguous regions per project, 1 to test, 1 to validation |
| With roads | 9,784 tiles (train 7,861, validation 1,189, test 734); 3,096 without |
| Roads | 23,777 labels: `residential` 66.8%, `service` 7.6%, `path` 7.3%, `unclassified` 5.0%, `secondary` 4.2%, ... 16 values |
| Projects | 60363 Freetown, Sierra Leone: 12,142 tiles, 21,343 roads, 0.30 m; 50793 Nairobi, Kenya: 415 tiles, 0.30 m; 40695 Stellenbosch, South Africa: 323 tiles, 0.25 m |
| Columns | `image`, `mask` (0 / 255), tile id and XYZ, bbox, project, `num_features`, `class_counts`, `label_geojson` |
| Licence | CC-BY 4.0 (imagery, OpenAerialMap) and ODbL 1.0 (labels, OpenStreetMap) |

Things to keep in mind: Freetown is 94% of the tiles, so the scores are a Freetown score; Nairobi and Stellenbosch are too small to score
separately with confidence (a split test has 910 tiles, most of them Freetown); the export script reproduces the card's counts
(`research/analysis/export_fair_layout.py`). The split by region leaves some leakage at region edges (section 7).

## 11. Training pipeline

`train()` (`train.py`) builds a `SegmentationDataset` per split, loads the ImageNet encoder (the whole encoder trains at a tenth of the head's rate; BatchNorm in eval mode
always), minimises binary cross-entropy plus Dice on the non-ignored pixels with AdamW and a cosine schedule stepped per batch, runs
`epoch_fraction` of the chips per epoch, keeps the best epoch by validation loss, and writes `<name>.pt`, `<name>.onnx`, `train.log`,
`history.json` and one checkpoint per epoch. Scoring (`evaluate.py`) is pixel precision, recall, F1 and IoU of the road class at a list of
thresholds, on the torch model or on the exported ONNX. Chips of different projects scale to different sizes (504 px at 0.30 m and
423 px at 0.25 m were seen at the old 0.15 m scale), so batches are padded with `IGNORE` (`pad_to_largest`).

## 12. Results

These runs were made between 2026-11-05 and 2026-11-09 on a GPU RTX 4060 8 GB. Every trial starts from the same baseline and changes one thing; each is
trained with `scripts/finetune.py` (10 epochs of a random tenth of the 10,253 train chips, batch 4, seed 42, no augmentation) and scored
with `scripts/evaluate.py` on **all 1,717 validation chips** over a threshold sweep (0.2 to 0.7). One seed; a change counts only if it moves
IoU by at least 0.01. The test split was not looked at until the final model was chosen.

| Run | Change from the baseline | Val IoU | Val F1 | Best threshold | Reading |
|---|---|---|---|---|---|
| B1_v2 | baseline: frozen encoder except `layer4` (a tenth of the rate), ImageNet V2, 0.30 m | 0.438 | 0.609 | 0.5 | |
| T1 | `layer3` and `layer4` train | 0.462 | 0.632 | 0.3 | +0.024: helps |
| **T2** | **the whole encoder trains (a tenth of the rate)** | **0.465** | **0.635** | **0.3** | **+0.027: helps; chosen** |
| T3 | 20 epochs instead of 10 | 0.439 | 0.610 | 0.5 | +0.001: no gain (best epoch 18 of 20) |
| T4 | orientation augmentation (`dihedral`) | 0.434 | 0.606 | 0.5 | -0.004: no gain |
| T5 | `model_gsd_m` 0.15 (2x upsampling) | 0.430 | 0.601 | 0.5 | -0.008: no gain, 18 min against about 10 |
| T6 | ImageNet V1 weights instead of V2 | 0.437 | 0.609 | 0.4 | -0.001: same |

What it says. With most of the encoder frozen the model was limited by its features, not by training length (T3): letting the encoder adapt
is the only change that moved the score. T1 and T2 differ by 0.003, which is noise, so T2 was taken as the simplest rule (everything trains).
Native scale, ImageNet V2 and no augmentation stay. T4 and T5 were measured on the frozen recipe; their interaction with a trainable encoder
was not measured.

**Final model: T2, threshold 0.3** (best F1 and IoU on validation; thresholds 0.2 to 0.4 are within 0.003 IoU). Pixel scores of the road class:

| Split | Chips | Precision | Recall | F1 | IoU |
|---|---|---|---|---|---|
| train | 10,253 | 0.668 | 0.693 | 0.680 | 0.515 |
| validation | 1,717 | 0.613 | 0.659 | 0.635 | 0.465 |
| test (scored once, at the validation threshold) | 910 | 0.637 | 0.700 | 0.667 | 0.500 |

The baseline B1_v2 scored 0.498 on train and 0.479 on test at 0.5 (IoU), so the baseline was under-fitted and T2 raises all three splits.
The gap between train (0.515) and validation (0.465) is small: the model is not over-fitted, and the score is bounded by the labels
(unmapped minor roads, the 6 m band, section 2) at least as much as by the model. Test is higher than validation; the regions differ and
the reason was not investigated. Overlay sheets (100 random chips per split, the prediction in green
over the labels in blue, a red frame on chips with no label) were written for the final model to `research/runs/T2_full/overlays/`
(`samples_<split>.png`, `samples_pred_<split>.png`); the scores of those 100 chips (train 0.525, validation 0.438, test 0.474 IoU) are from a
sample and are not the split scores above.

**Not measured:** the postprocess sweep on polygons, object-level (per road) scores, per-project scores (Freetown is 94% of the tiles),
a second seed, a trainable encoder BatchNorm, orientation together with a trainable encoder, and a tolerance score that does not count a
prediction a few pixels off the label as an error.


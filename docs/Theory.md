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
| Centreline scoring (2026-11-10) | Thinned to a line, the final model scores tolerance F1 0.42 at 1 m, 0.61 at 2 m and 0.70 at 3 m on validation; models trained on a 2 m or a 0.9 m strip are within noise of it, a 0.3 m line is worse (section 13). Thinning and scoring are in `hs_hot.centreline` / `hs_hot.evaluate_centreline` (`centreline` extra); the experiments are in `research/`, validation only |
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
for Stellenbosch. A finer scale was tried and did not help (section 12, trial T5: 2x upsampling, IoU 0.430 against 0.438 and about
1.8 times the training time).

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
  reviewed.

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
| `freeze_encoder`, `unfreeze_from`, `encoder_lr_factor` | false, `none`, 0.1 | the whole encoder trains at a tenth of the head's rate; `freeze_encoder=true` freezes all of it; `unfreeze_from` (`layer1` to `layer4`) freezes it except that stage and the deeper ones (naming it without `freeze_encoder` implies the freeze); encoder BatchNorm stays in eval mode |
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
  chips, an object-level (per road) score, a tolerance score on the polygon bands (measured on thinned centrelines only, section 13), a ground-truth check of the band width, and zoom 18 behaviour;
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
423 px at 0.25 m were seen at a 2x finer scale), so batches are padded with `IGNORE` (`pad_to_largest`).

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
| T5 | a 2x finer scale (upsampling) | 0.430 | 0.601 | 0.5 | -0.008: no gain, 18 min against about 10 |
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
a second seed, a trainable encoder BatchNorm, orientation together with a trainable encoder, and a tolerance score for the polygon bands
(a tolerance score for thinned centrelines is in section 13).

## 13. Centreline scoring (2026-11-10, validation only)

Why: a challenge may score road **centrelines** (the OSM line with a buffer) instead of 6 m bands. The library returns bands, so this section
measures what happens when the band is thinned to a line, how the scores move with the tolerance, and whether training on a narrower target
helps. The thinning and the scoring are in the library (`hs_hot.centreline`, `hs_hot.evaluate_centreline`, `centreline` extra, tests in
`tests/test_centreline.py`; not exported from `hs_hot`, no default changed); the experiments are in `research/`
(`analysis/centreline_lib.py`, `analysis/centreline_eval.py`, `train_T7.py`, which takes any label width with `--half-width-m`).
Validation split only (1,717 chips); the test split was not used. One seed.

**Models compared.** Same recipe as section 12 (whole encoder trains, 10 epochs of a tenth of the chips); only the training target differs.
The label is always drawn from the OSM centreline in `labels.geojson` (not from the dataset's mask rasters), as a band of the given width.

| Name here | Trial | Target the model is trained on | Weights |
|---|---|---|---|
| Band model | T2 | OSM centreline + 3 m each side (6 m band, about 20 px), the final model of section 12 | `research/runs/T2_full/` |
| Narrow-target model | T7 | centreline + 1 m each side (2 m strip, about 7 px) | `research/runs/T7_narrow/` (best epoch 7 of 10, 12 min 26 s) |
| 3-pixel model | T8 | centreline + 0.45 m each side (about 0.9 m, 3 px) | `research/runs/T8_3px/` (best epoch 10 of 10, 10 min 34 s) |
| 1-pixel model | T9 | centreline + 0.15 m each side (about 0.3 m, 1 px) | `research/runs/T9_1px/` (best epoch 10 of 10, 10 min 18 s) |

Each model's pixel score on its own masks is not comparable with the band model's 0.465 because the target differs (T7 IoU 0.270 and F1 0.425
at threshold 0.3; T8 IoU 0.150 and F1 0.262 at 0.2; T9 IoU 0.060 and F1 0.113 at 0.3); the models are compared with the line scores below.
T8 and T9 were still improving at epoch 10 (validation loss), so a longer run might change them; T3 (section 12) showed no gain from a longer run
for the band model only.

**Method.** The road probability is cut at a threshold (0.3 here), thinned to a one-pixel line with `skimage.morphology.skeletonize`
(set `P`), and compared with the OSM centreline drawn one pixel wide (set `R`), both on the model's pixel grid (about 0.30 m per pixel;
each chip uses its own metres per pixel). With `d_R(x)` the distance in metres from pixel `x` to the nearest pixel of `R` (a Euclidean
distance transform) and `d_P(x)` likewise for `P`, and a tolerance `r` in metres, counts pooled over all chips:

```
precision(r) = |{p in P : d_R(p) <= r}| / |P|
recall(r)    = |{q in R : d_P(q) <= r}| / |R|
F1           = 2 * precision * recall / (precision + recall)
quality(r)   = TPr / (TPr + FP + FN)    TPr = matched reference pixels, FP = |P| - matched predicted pixels, FN = |R| - TPr
IoU(r)       = |A and B| / |A or B|     A = {x : d_P(x) <= r}, B = {x : d_R(x) <= r}   (both lines widened by r on each side)
```

`r = 1 m` means a point counts as matched if the other line passes within 1 m of it, on either side (a corridor 2 m wide). Pixel counts
stand in for lengths (a diagonal step is about 1.4 pixels; the same bias is on both lines). Matching is not one to one: one reference pixel
can vouch for several predicted pixels. The IoU of widened lines is the overlap an IoU metric on "centreline plus buffer" polygons would see;
it is not the pixel IoU of section 12.

**Reference rows** (same chips): the OSM line against itself (must be 1.0; it is), the OSM line moved by `d` metres in a random direction
per chip, and the dataset's own 6 m band thinned to a line (what thinning alone loses). F1 / IoU of widened lines:

| Reference | r = 0.5 m | r = 1 m | r = 2 m | r = 3 m |
|---|---|---|---|---|
| Perfect 6 m band, thinned | 0.840 / 0.620 | 0.900 / 0.753 | 0.954 / 0.855 | 0.975 / 0.890 |
| OSM line moved 0.5 m | 0.840 / 0.532 | 0.999 / 0.741 | 0.999 / 0.856 | 1.000 / 0.903 |
| OSM line moved 1 m | 0.359 / 0.248 | 0.930 / 0.542 | 0.997 / 0.733 | 0.998 / 0.815 |
| OSM line moved 2 m | 0.167 / 0.093 | 0.362 / 0.252 | 0.921 / 0.525 | 0.994 / 0.659 |
| OSM line moved 3 m | 0.121 / 0.063 | 0.243 / 0.142 | 0.490 / 0.370 | 0.936 / 0.535 |

**Results**, validation, threshold 0.3 (P precision, R recall, Qual quality, IoU of the widened lines):

| Tolerance r (m) | Band P | Band R | Band F1 | Band Qual | Band IoU | Narrow P | Narrow R | Narrow F1 | Narrow Qual | Narrow IoU |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.225 | 0.232 | 0.228 | 0.129 | 0.124 | 0.257 | 0.231 | 0.243 | 0.139 | 0.133 |
| 1 | 0.412 | 0.428 | 0.420 | 0.265 | 0.235 | 0.461 | 0.418 | 0.439 | 0.282 | 0.245 |
| 1.5 | 0.503 | 0.522 | 0.512 | 0.344 | 0.295 | 0.558 | 0.507 | 0.531 | 0.363 | 0.305 |
| 2 | 0.600 | 0.620 | 0.610 | 0.438 | 0.363 | 0.655 | 0.599 | 0.626 | 0.458 | 0.370 |
| 3 | 0.695 | 0.709 | 0.702 | 0.538 | 0.440 | 0.741 | 0.684 | 0.711 | 0.555 | 0.442 |
| 4 | 0.742 | 0.752 | 0.747 | 0.593 | 0.488 | 0.776 | 0.724 | 0.749 | 0.603 | 0.486 |

**All four models** (F1 of the thinned line; each model at the threshold of the sweep that is best by F1 at 1 m on validation, and at 0.3 for
comparison). The thin targets need a far lower threshold: their probability maps are small and peaked, so 0.3 cuts away most of the line.

| Model | Threshold | r = 0.5 m | r = 1 m | r = 2 m | r = 3 m | r = 4 m | Predicted line pixels |
|---|---|---|---|---|---|---|---|
| Band (T2) | 0.7 (end of the sweep) | 0.235 | 0.431 | 0.624 | 0.717 | 0.760 | 336,710 |
| Band (T2) | 0.3 | 0.228 | 0.420 | 0.610 | 0.702 | 0.747 | 362,418 |
| Narrow-target, 2 m (T7) | 0.4 | 0.244 | 0.439 | 0.627 | 0.712 | 0.750 | 305,039 |
| 3-pixel (T8) | 0.02 | 0.234 | 0.423 | 0.608 | 0.695 | 0.738 | 307,492 |
| 3-pixel (T8) | 0.3 | 0.229 | 0.414 | 0.596 | 0.681 | 0.723 | 235,726 |
| 1-pixel (T9) | 0.02 | 0.207 | 0.379 | 0.551 | 0.636 | 0.683 | 273,346 |
| 1-pixel (T9) | 0.3 | 0.194 | 0.360 | 0.527 | 0.608 | 0.655 | 178,160 |

The reference has 347,906 line pixels. The sweeps are in `research/runs/centreline/val_T2_T7_T8_T9.json` (0.1 to 0.7) and `val_T8_T9_lowthr.json`
(0.01 to 0.1); T8 and T9 are flat from 0.02 to 0.1 (within 0.003 F1 at 1 m) and lower at 0.01.

**Narrower targets.** From the 6 m band to the 2 m strip to the 3-pixel strip the F1 stays within about 0.02 of each other at every tolerance
(1 m: 0.431, 0.439, 0.423 at their best thresholds), which is within what one seed can tell apart; the 2 m strip is the highest at 0.5 to 2 m. The
1-pixel line is clearly worse (0.379 at 1 m, 0.05 to 0.08 below the others at 2 to 4 m): a target that thin leaves almost every pixel as
background, the training loss is still 0.93 after 10 epochs, and the model finds less of the road (recall 0.34 against 0.40 to 0.43 at 1 m).
The median offset to the OSM line is 0.85 to 0.95 m for all four models, whatever the target; that points to the alignment of the hand-drawn
OSM lines (and the thinning, 0.35 m for a perfect band) as a floor that a narrower target does not remove, but the numbers here cannot separate it from model error.

Median distance from a predicted line pixel to the OSM line (pixels within 5 m of it) at 0.3: 0.95 m (band model), 0.85 m (narrow-target model).
Predicted line length: 362,418 pixels (band), 311,762 (narrow-target), against 347,906 reference pixels. The scores at thresholds 0.2 to 0.7
are in `research/runs/centreline/val_T2_T7.json`; F1 moves by about 0.02 across that range (0.017 to 0.022 for the band model at 1 to 3 m), and the band model's F1 was still rising
slightly at 0.7, the end of the sweep.

**Reading.**

- Thinning alone costs little at a wide tolerance (perfect band, F1 0.954 at 2 m) and a lot at a tight one (0.840 at 0.5 m): a 0.5 m tolerance
  is about 1.7 pixels, so even a perfect label thinned fails it. The 0.5 m scores of the models (0.23 to 0.24) are mostly that.
- The tolerance F1 forgives an offset up to `r` (the OSM line moved 1 m still scores F1 0.93 at r = 1 m), but the IoU of widened lines does
  not: the same moved line scores only 0.542. For two strips of width `2r` an offset `d` gives IoU = (2r - d) / (2r + d), one third at
  `d = r`; the measured value is higher because the shift direction is random against each road. So IoU on a 1 m buffer is a harsh score:
  about 0.24 for both models, against F1 0.42 to 0.44 at the same 1 m.
- Narrow target against band: higher precision and lower recall at every tolerance (it draws about 14% fewer line pixels). F1 is higher by
  0.015 to 0.019 from 0.5 to 2 m, 0.009 at 3 m and 0.002 at 4 m, so the models converge as the tolerance grows. With one seed and gaps near
  the 0.01 bar this is a small edge at tight tolerances, not a confirmed gain. A second seed would settle it.
- The best a thinned perfect band can score at 1 m is F1 0.900 and IoU 0.753; both models are far below it. The remaining gap is model error
  plus unmapped roads (they lower precision) plus the alignment of hand-drawn OSM lines; these cannot be separated from these numbers.

**Caveats.** One seed; validation only; Freetown is 94% of the tiles; length counted in pixels; no clean-up of short spurs on the thinned
lines; the labels are incomplete, so precision is a lower bound; the library draws labels with PIL, which can shift a line by up to half a
pixel (about 0.15 m), the same for every row above. Overlay sheets for 100 random chips per split, the OSM line in blue and the thinned
prediction in red (`samples_centreline_<split>.png`, with `samples_<split>.png` and `samples_pred_<split>.png` in the layout of section 12), are in
`research/runs/T7_narrow/overlays/` and `research/runs/T2_full/overlays_centreline/`.

**Reproduce** (from `hs-hot/`, shared `HOTOSM/.venv` python): `python research/train_T7.py`, then
`python research/analysis/centreline_eval.py --split val --model T2=research/runs/T2_full/T2_full.pt --model T7=research/runs/T7_narrow/T7_narrow.pt --out research/runs/centreline/val_T2_T7.json`.
T8 and T9: `python research/train_T7.py --half-width-m 0.45 --out research/runs/T8_3px --name T8_3px` and `--half-width-m 0.15 ... T9_1px`;
for them score with `--thresholds 0.01 0.02 0.03 0.05 0.075 0.1`.
The thinning and the scoring need `scikit-image` and `scipy`: the `centreline` extra of `pyproject.toml` (floors 0.26 and 1.18, the versions
tested), also in the `test` extra so the 9 tests of `tests/test_centreline.py` run in CI (148 tests pass in all). They were installed in the
shared dev venv on 2026-11-10 (scikit-image 0.26.0, scipy 1.18.1 with imageio, lazy-loader and tifffile); `import hs_hot` still loads none of them.
No `uv.lock` exists, so none was regenerated.

**Draft baseline for a challenge** (`research/zindi/baseline_centreline.py`, and a `.ipynb` built from it): chips, ONNX model, thin, buffer,
`ID,geometry` CSV of `MULTIPOLYGON` in chip pixels. It ran on 40 validation chips as a script; the notebook was not executed. The submission
format, buffer width and metric wait for the organiser's answers.

**Not done:** a second seed; longer runs for the thin targets (T8, T9 were still improving at epoch 10); the test split; spur clean-up; the band model's threshold sweep beyond 0.7; per-project scores; a check of how far
OSM lines sit from the visible road centre; centrelines as a library output (the library returns polygon bands).


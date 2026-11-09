"""Dataset helpers for hs-hot (numpy, rasterio, Pillow and pyproj only, no torch; safe to import at serving time).

- `list_chip_paths`, `spatial_split`: find the chips and split them into train and validation.
- `build_chip_labels`: one GeoJSON of road centrelines (`highway=*` lines) -> per-chip label shapes in chip pixels,
  nothing copied. `load_labels` takes either that mapping or the GeoJSON path.
- `scale_chip_and_labels`: scale a chip as at serving time and draw its labels at that resolution. The training-only
  changes (`augment_chip_and_mask`) live in `augment.py`.

A mask holds `BACKGROUND`, `ROAD` and `IGNORE` (pixels left out of the loss and of the scores; no label draws them,
only padding does). A road is a centreline buffered by `ROAD_HALF_WIDTH_M` on each side, as the Guru highways dataset
draws its masks (a band 6 m wide). A label shape is a dict `{"kind": "road", "line": (N, 2) array of (col, row)
vertices, "width_px": band width in chip pixels}`.
"""

import json
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer

from .params import PreprocessParams
from .preprocess import preprocess_rgb
from .tiles import OAM_TILE_RE, chip_scale, tile_gsd_m

CHIP_PATTERNS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
LABEL_SUFFIXES = (".geojson", ".json")
BACKGROUND, ROAD, IGNORE = 0, 1, 255
ROAD_HALF_WIDTH_M = (
    3.0  # each side of a centreline: the Guru highways masks buffer the lines by 3 m (a 6 m band)
)
MAX_BAND_M = 10.0  # chips look this far beyond their edge for centrelines whose band reaches in
METRES_PER_DEGREE = 111_320  # spherical approximation, within 1% of WGS84 latitude degrees


def list_chip_paths(root: str | Path) -> list[Path]:
    """RGB chips in a folder. Training and evaluation read any of them (the scale comes from the OAM name), but turning
    GeoJSON labels into chip pixels needs georeferenced GeoTIFFs."""
    folder = Path(root)
    return sorted(path for pattern in CHIP_PATTERNS for path in folder.glob(pattern) if path.is_file())


def spatial_split(
    chip_names: list[str], val_ratio: float, seed: int, *, block_size: int = 4
) -> tuple[list[str], list[str]]:
    """Block-spatial split on OAM tile coords; whole `(x // K, y // K)` blocks go to one side.

    Neighbouring chips share roads cut at chip borders and similar scenes, so a random
    chip split leaks. Every name must be `OAM-x-y-z.ext`; anything else raises `ValueError`.
    Returns sorted `(train, val)` name lists.
    """
    blocks_map: dict[tuple[int, int], list[str]] = {}
    for name in chip_names:
        match = OAM_TILE_RE.match(name)
        if match is None:
            raise ValueError(f"chip name {name!r} does not match OAM-<x>-<y>-<z>.<ext>")
        blocks_map.setdefault(
            (int(match.group(1)) // block_size, int(match.group(2)) // block_size), []
        ).append(name)

    rng = np.random.default_rng(seed)
    blocks = sorted(blocks_map)
    rng.shuffle(blocks)

    n_val_target = max(1, int(len(chip_names) * val_ratio)) if chip_names else 0

    val: list[str] = []
    train: list[str] = []
    for block in blocks:
        (val if len(val) < n_val_target else train).extend(blocks_map[block])
    return sorted(train), sorted(val)


def _find_labels_geojson(labels: str | Path) -> Path:
    """The labels file: the path itself if it is a `.geojson` / `.json` file, else the one such file under a folder."""
    root = Path(labels)
    if root.is_file() and root.suffix.lower() in LABEL_SUFFIXES:
        return root
    matches = sorted(root.rglob("*.geojson")) or sorted(root.rglob("*.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one .geojson (or .json) in {root}, found {len(matches)}")
    return matches[0]


def _parse_geojson_crs(feature_collection: dict[str, Any]) -> str:
    """CRS name from the legacy GeoJSON `crs` member; RFC 7946 files default to EPSG:4326."""
    crs = feature_collection.get("crs")
    if isinstance(crs, dict):
        props = crs.get("properties")
        if isinstance(props, dict) and isinstance(props.get("name"), str) and props["name"]:
            return props["name"]
    return "EPSG:4326"


def _line_shape(coordinates: list[Any]) -> dict[str, Any]:
    line = np.asarray(coordinates, dtype=float)[:, :2]
    return {"kind": "line", "line": line, "bounds": (*line.min(axis=0), *line.max(axis=0))}


def read_label_shapes(labels_geojson: str | Path) -> tuple[list[dict[str, Any]], str]:
    """Road centrelines of a GeoJSON as shapes in the file's CRS, plus that CRS.

    A `LineString` feature gives one line shape (a `MultiLineString` one per part), taken as longitude and latitude in
    degrees unless the file names another CRS. Other geometries (points, polygons) are skipped.
    """
    collection = json.loads(Path(labels_geojson).read_text(encoding="utf-8"))
    if not isinstance(collection, dict) or collection.get("type") != "FeatureCollection":
        raise ValueError("labels file must be a GeoJSON FeatureCollection")
    features = collection.get("features", [])
    if not isinstance(features, list):
        raise ValueError("GeoJSON FeatureCollection 'features' must be a list")
    shapes: list[dict[str, Any]] = []
    for feature in features:
        geometry = feature.get("geometry") if isinstance(feature, dict) else None
        if not isinstance(geometry, dict):
            continue
        kind, coordinates = geometry.get("type"), geometry.get("coordinates")
        if kind == "LineString" and coordinates and len(coordinates) >= 2:
            shapes.append(_line_shape(coordinates))
        elif kind == "MultiLineString":
            shapes.extend(_line_shape(part) for part in coordinates if part and len(part) >= 2)
    return shapes, _parse_geojson_crs(collection)


@lru_cache(maxsize=16)
def _transformer(source_crs: str, target_crs: str) -> Transformer:
    """A coordinate transformer, made once: building one costs far more than using it on a chip's few vertices."""
    return Transformer.from_crs(source_crs, target_crs, always_xy=True)


def _read_chip_georeference(chip_path: str | Path) -> tuple[int, int, Any, Any]:
    """`(width, height, affine transform, CRS)` of a georeferenced chip; a chip without a CRS cannot take GeoJSON labels."""
    import rasterio

    with rasterio.open(chip_path) as src:
        width, height, transform, crs = int(src.width), int(src.height), src.transform, src.crs
    if crs is None:
        raise ValueError(f"{chip_path} has no CRS; GeoJSON labels need a georeferenced GeoTIFF chip")
    return width, height, transform, crs


def _chip_extent_in_labels_crs(
    width: int, height: int, transform: Any, chip_crs: Any, labels_crs: str
) -> tuple[float, ...]:
    """`(min_x, min_y, max_x, max_y)` of the chip's four corners, in the CRS the labels are written in."""
    pixel_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)
    # Explicit affine maths (same reason as georef.py): works across `affine` versions.
    x = transform.a * pixel_corners[:, 0] + transform.b * pixel_corners[:, 1] + transform.c
    y = transform.d * pixel_corners[:, 0] + transform.e * pixel_corners[:, 1] + transform.f
    if str(chip_crs) != str(labels_crs):
        x, y = _transformer(str(chip_crs), str(labels_crs)).transform(x, y)
    return float(min(x)), float(min(y)), float(max(x)), float(max(y))


def _shapes_near_extent(shape_bounds: np.ndarray, extent: tuple[float, ...]) -> np.ndarray:
    """Indices of the label shapes whose bounds come within `MAX_BAND_M` of `extent` (all in the labels' CRS).

    The margin is in degrees, because the labels are: a road band can reach that far over the chip's edge, so a
    centreline just outside the chip may still draw into it.
    """
    min_x, min_y, max_x, max_y = extent
    margin = MAX_BAND_M / METRES_PER_DEGREE
    near = (
        (shape_bounds[:, 2] >= min_x - margin) & (shape_bounds[:, 0] <= max_x + margin)
        & (shape_bounds[:, 3] >= min_y - margin) & (shape_bounds[:, 1] <= max_y + margin)
    )  # fmt: skip
    return np.flatnonzero(near)


def _pixel_converter(
    transform: Any, chip_crs: Any, labels_crs: str
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """A function turning label coordinates `(x, y)` into an `(N, 2)` array of chip `(col, row)` pixels."""
    labels_to_chip = (
        None if str(chip_crs) == str(labels_crs) else _transformer(str(labels_crs), str(chip_crs))
    )
    pixel_from_map = ~transform

    def to_pixels(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if labels_to_chip is not None:
            x, y = labels_to_chip.transform(x, y)
        return np.stack(
            [
                pixel_from_map.a * x + pixel_from_map.b * y + pixel_from_map.c,
                pixel_from_map.d * x + pixel_from_map.e * y + pixel_from_map.f,
            ],
            axis=1,
        )

    return to_pixels


def _line_in_pixels(
    shape: dict[str, Any],
    to_pixels: Callable[[np.ndarray, np.ndarray], np.ndarray],
    metres_per_pixel: float,
    half_width_m: float,
) -> dict[str, Any]:
    """A centreline as a `road` shape in chip pixels, with the width of its band at the chip's true ground resolution."""
    line = to_pixels(shape["line"][:, 0], shape["line"][:, 1])
    return {"kind": "road", "line": line, "width_px": 2.0 * half_width_m / metres_per_pixel}


def _band_reaches_chip(shape: dict[str, Any], width: int, height: int) -> bool:
    """Whether the road's band (its centreline widened by half its width) reaches into the chip's pixel area."""
    cols, rows = shape["line"][:, 0], shape["line"][:, 1]
    reach = shape["width_px"] / 2.0
    return bool(
        cols.max() + reach > 0
        and cols.min() - reach < width
        and rows.max() + reach > 0
        and rows.min() - reach < height
    )


def chip_shapes(
    chip_path: str | Path,
    *,
    shapes: list[dict[str, Any]],
    bounds: np.ndarray,
    labels_crs: str,
    half_width_m: float = ROAD_HALF_WIDTH_M,
) -> list[dict[str, Any]]:
    """The road shapes whose band touches a chip, in chip pixels.

    A centreline becomes a `road` shape whose band is `2 * half_width_m` metres wide at the chip's true ground
    resolution. A road just outside the chip still draws its band into it, so lines are taken from the whole file.

    `shapes` and `bounds` are the whole file's shapes and their `(N, 4)` bounds from `read_label_shapes`; `labels_crs`
    is the CRS they are written in. The steps: read the chip's size and georeference; keep the shapes near its extent;
    convert each to chip pixels; drop those whose band does not reach into the chip.
    """
    width, height, transform, chip_crs = _read_chip_georeference(chip_path)
    metres_per_pixel = tile_gsd_m(Path(chip_path).name, width)
    extent = _chip_extent_in_labels_crs(width, height, transform, chip_crs, labels_crs)
    to_pixels = _pixel_converter(transform, chip_crs, labels_crs)
    in_pixels = [
        _line_in_pixels(shapes[index], to_pixels, metres_per_pixel, half_width_m)
        for index in _shapes_near_extent(bounds, extent)
    ]
    return [shape for shape in in_pixels if _band_reaches_chip(shape, width, height)]


def build_chip_labels(
    chips_dir: str | Path, labels: str | Path, *, half_width_m: float = ROAD_HALF_WIDTH_M
) -> dict[str, list[dict[str, Any]]]:
    """Label shapes of every chip in `chips_dir` from one GeoJSON of road centrelines. Nothing is copied.

    `labels` is the `.geojson` / `.json` file, or a folder holding exactly one. A pipeline step can call this on the
    chips and labels it was given, so no prepared folder has to be shared between isolated steps. Chips without a
    label get an empty list (all background).
    """
    shapes, crs = read_label_shapes(_find_labels_geojson(labels))
    bounds = np.array([shape["bounds"] for shape in shapes], dtype=float).reshape(-1, 4)
    return {
        path.name: chip_shapes(path, shapes=shapes, bounds=bounds, labels_crs=crs, half_width_m=half_width_m)
        for path in list_chip_paths(chips_dir)
    }


def load_labels(
    chips_dir: str | Path, labels: str | Path | Mapping[str, list[dict[str, Any]]]
) -> Mapping[str, list[dict[str, Any]]]:
    """The per-chip label shapes themselves, or built from a labels GeoJSON path (`build_chip_labels`)."""
    if isinstance(labels, Mapping):
        return labels
    return build_chip_labels(chips_dir, labels)


def rasterize_chip_labels(shapes: list[dict[str, Any]], height: int, width: int) -> np.ndarray:
    """`(H, W)` `uint8` mask of road shapes in pixels: `ROAD` where a band is drawn, background elsewhere.

    A band is the centreline drawn `width_px` wide, with a disc of that width on every vertex so bends and ends are round.
    """
    canvas = Image.new("L", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    for shape in shapes:
        points = [(float(col), float(row)) for col, row in shape["line"]]
        band = max(1, round(shape["width_px"]))
        draw.line(points, fill=ROAD, width=band)
        radius = band / 2.0
        for col, row in points:
            draw.ellipse([col - radius, row - radius, col + radius, row + radius], fill=ROAD)
    return np.asarray(canvas, dtype=np.uint8)


def scale_chip_and_labels(
    rgb: np.ndarray, shapes: list[dict[str, Any]], pre: PreprocessParams, name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Scale an RGB chip exactly as at serving time and draw its labels at that resolution.

    The label shapes (chip pixels) are scaled with the chip and rasterised at the scaled size, so the band edges are as
    sharp as the image. Returns the `(3, H', W')` float32 image and the `(H', W')` `uint8` mask.
    """
    scale = chip_scale(
        name, rgb.shape[1], pre.model_gsd_m, height_px=rgb.shape[0], max_input_px=pre.max_input_px
    )
    image, meta = preprocess_rgb(rgb, scale)
    factors = np.array([meta.scale_x, meta.scale_y])
    mean_scale = (meta.scale_x + meta.scale_y) / 2.0
    scaled = [
        {**shape, "line": shape["line"] * factors, "width_px": shape["width_px"] * mean_scale}
        for shape in shapes
    ]
    return image, rasterize_chip_labels(scaled, image.shape[1], image.shape[2])

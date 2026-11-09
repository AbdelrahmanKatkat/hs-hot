import json
import shutil

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

ORIGIN_X, ORIGIN_Y = 1_000_000.0, 2_000_000.0  # the top-left corner of every synthetic chip, in EPSG:3857


@pytest.fixture
def write_chip(tmp_path):
    """Write a synthetic georeferenced RGB GeoTIFF (EPSG:3857, 1 m pixels)."""

    def _write(name: str = "chip.tif", height: int = 256, width: int = 256, count: int = 3):
        path = tmp_path / name
        data = np.random.default_rng(0).integers(0, 256, size=(count, height, width), dtype=np.uint8)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype="uint8",
            crs="EPSG:3857",
            transform=from_origin(ORIGIN_X, ORIGIN_Y, 1.0, 1.0),
        ) as dst:
            dst.write(data)
        return path

    return _write


# A tile near the equator at zoom 20: 14.9 cm/px, so a 256 px chip is scaled by about 2.99.
CHIP_NAME = "OAM-524288-524288-20.tif"


def lon_lat(col: float, row: float) -> list[float]:
    """The longitude and latitude of chip pixel `(col, row)` of a synthetic chip (1 m pixels, origin above)."""
    return list(
        Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform(
            ORIGIN_X + col, ORIGIN_Y - row
        )
    )


def road_line(*points: tuple[float, float], **tags: str) -> dict:
    """A `highway=*` line through chip pixels `(col, row)` of a synthetic chip, in EPSG:4326."""
    return {
        "type": "Feature",
        "properties": {"highway": "residential", **tags},
        "geometry": {"type": "LineString", "coordinates": [lon_lat(col, row) for col, row in points]},
    }


@pytest.fixture
def labelled_chips(write_chip, tmp_path):
    """Four chips of one zoom-20 block, all with the same georeferencing, and one GeoJSON of labels in EPSG:4326.

    The labels: a road across the whole chip along row 64, and a shorter diagonal one from (20, 150) to (230, 230). The
    chip name says 0.149 m/px, so a 6 m band is 40.3 px wide.
    """
    chips, labels = tmp_path / "chips", tmp_path / "labels"
    chips.mkdir()
    labels.mkdir()
    names = [f"OAM-{524288 + index}-524288-20.tif" for index in range(4)]
    for name in names:
        shutil.copy(write_chip(name=name), chips / name)
    features = [road_line((-50, 64), (306, 64)), road_line((20, 150), (230, 230))]
    (labels / "labels.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return chips, labels, names

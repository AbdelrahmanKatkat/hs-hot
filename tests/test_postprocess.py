import numpy as np
from rasterio.transform import from_origin

from hs_hot.decode import decode_polygons
from hs_hot.georef import polygons_to_chip_pixels, ring_to_chip_pixels, ring_to_epsg4326
from hs_hot.params import PostprocessParams
from hs_hot.postprocess import polygons_to_feature_collection
from hs_hot.preprocess import PreprocessMeta


def _meta(crs="EPSG:3857") -> PreprocessMeta:
    """A 64 px chip with origin (1 000 000, 2 000 000) and 1 m pixels, scaled 2x for the model."""
    return PreprocessMeta(
        width=64, height=64, resized_width=128, resized_height=128,
        transform=from_origin(1_000_000.0, 2_000_000.0, 1.0, 1.0), crs=crs,
    )  # fmt: skip


def _blob(value: float = 0.8) -> np.ndarray:
    probability = np.zeros((128, 128), dtype=np.float32)
    probability[20:80, 20:80] = value
    return probability


def test_model_pixels_are_divided_by_the_scale_to_give_chip_pixels():
    ring = np.array([[0.0, 0.0], [128.0, 0.0], [128.0, 64.0], [0.0, 0.0]])
    assert ring_to_chip_pixels(ring, _meta()).tolist() == [[0, 0], [64, 0], [64, 32], [0, 0]]
    (polygon,) = decode_polygons(_blob(), PostprocessParams())
    (chip,) = polygons_to_chip_pixels([polygon], _meta())
    assert chip["confidence"] == polygon["confidence"] and chip["rings"][0].max() == 40.0  # 80 / 2


def test_a_ring_lands_on_the_map_point_of_its_chip_pixel_and_a_chip_without_crs_stays_in_pixels():
    ring = np.array([[0.0, 0.0], [3.0, 6.0]])
    assert ring_to_epsg4326(ring, _meta().transform, None) == [
        [1_000_000.0, 2_000_000.0],
        [1_000_003.0, 1_999_994.0],
    ]
    lon, lat = ring_to_epsg4326(ring, _meta().transform, "EPSG:3857")[0]
    assert abs(lon - 8.98) < 0.01 and abs(lat - 17.68) < 0.05  # Web Mercator (1 000 000, 2 000 000)
    assert ring_to_epsg4326(ring, _meta().transform, "EPSG:4326")[1] == [1_000_003.0, 1_999_994.0]


def test_features_are_epsg4326_polygons_with_confidence_class_and_source():
    meta = _meta()
    polygons = polygons_to_chip_pixels(decode_polygons(_blob(), PostprocessParams()), meta)
    collection = polygons_to_feature_collection(polygons, meta, source_name="OAM-1-2-3.tif")
    assert collection["type"] == "FeatureCollection" and len(collection["features"]) == 1
    feature = collection["features"][0]
    assert feature["properties"] == {
        "confidence": 0.8,
        "class_id": 1,
        "class_name": "highway",
        "source": "OAM-1-2-3.tif",
    }
    ring = np.array(feature["geometry"]["coordinates"][0])
    assert feature["geometry"]["type"] == "Polygon" and np.array_equal(ring[0], ring[-1])
    assert (np.abs(ring[:, 0] - 8.98) < 0.01).all() and (np.abs(ring[:, 1] - 17.68) < 0.05).all()


def test_no_polygons_give_an_empty_collection_and_holes_are_kept():
    assert polygons_to_feature_collection([], _meta()) == {"type": "FeatureCollection", "features": []}
    probability = _blob()
    probability[40:60, 40:60] = 0.0
    meta = _meta()
    collection = polygons_to_feature_collection(
        polygons_to_chip_pixels(decode_polygons(probability), meta), meta
    )
    assert len(collection["features"][0]["geometry"]["coordinates"]) == 2  # exterior and one hole

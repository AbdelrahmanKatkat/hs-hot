"""Postprocess: the model output (road probability per chip) -> polygons -> GeoJSON FeatureCollection.

`postprocess_outputs` is the ready-to-use step a pack's `postprocess` calls on what `predict_session` returned: it decodes
each probability (stage 3, `inference.*` decoding keys) and georeferences the polygons (stage 4). Inference itself
(`serve.py`) does neither.
"""

from pathlib import Path
from typing import Any

import numpy as np

from .decode import decode_polygons
from .georef import polygons_to_chip_pixels, ring_to_epsg4326
from .params import parse_postprocess_params
from .preprocess import PreprocessMeta

CLASS_NAMES = {1: "highway"}


def polygons_to_feature_collection(
    polygons: list[dict[str, Any]],
    meta: PreprocessMeta,
    *,
    source_name: str | None = None,
) -> dict[str, Any]:
    """`polygons` are in source-chip pixels; output is EPSG:4326 polygons with their holes."""
    features: list[dict[str, Any]] = []
    for polygon in polygons:
        rings = [ring_to_epsg4326(ring, meta.transform, meta.crs) for ring in polygon["rings"]]
        props: dict[str, Any] = {
            "confidence": round(float(polygon["confidence"]), 4),
            "class_id": int(polygon["class_id"]),
            "class_name": CLASS_NAMES.get(int(polygon["class_id"]), "highway"),
        }
        if source_name:
            props["source"] = source_name
        features.append(
            {"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": rings}}
        )
    return {"type": "FeatureCollection", "features": features}


def postprocess_outputs(
    outputs: list[tuple[Path, np.ndarray, PreprocessMeta]], params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """`predict_session` output -> one EPSG:4326 FeatureCollection with the features of every chip."""
    cfg = parse_postprocess_params(params)
    features: list[dict[str, Any]] = []
    for path, probability, meta in outputs:
        polygons = polygons_to_chip_pixels(decode_polygons(probability, cfg), meta)
        features.extend(
            polygons_to_feature_collection(polygons, meta, source_name=Path(path).name)["features"]
        )
    return {"type": "FeatureCollection", "features": features}

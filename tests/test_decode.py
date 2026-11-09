import numpy as np
import pytest

from hs_hot.decode import (
    _polygon_area,
    decode_polygons,
    polygon_confidence,
    simplify_ring,
    threshold_mask,
    trace_polygons,
)
from hs_hot.params import PostprocessParams


def _square(top: int, left: int, side: int, value: float = 1.0, shape=(64, 64)) -> np.ndarray:
    probability = np.zeros(shape, dtype=np.float32)
    probability[top : top + side, left : left + side] = value
    return probability


def test_threshold_is_inclusive_and_returns_uint8():
    mask = threshold_mask(np.array([[0.49, 0.5, 0.51]]), 0.5)
    assert mask.dtype == np.uint8 and mask.tolist() == [[0, 1, 1]]


def test_a_square_is_one_polygon_with_the_area_of_its_pixels():
    (rings,) = trace_polygons(threshold_mask(_square(5, 7, 10), 0.5))
    assert len(rings) == 1 and _polygon_area(rings) == 100
    assert np.array_equal(rings[0][0], rings[0][-1])  # closed
    assert rings[0].min(axis=0).tolist() == [7, 5] and rings[0].max(axis=0).tolist() == [17, 15]  # (col, row)


def test_a_region_with_a_hole_is_one_polygon_and_the_hole_is_not_counted():
    probability = _square(2, 2, 20)
    probability[8:14, 8:14] = 0.0
    (rings,) = trace_polygons(threshold_mask(probability, 0.5))
    assert len(rings) == 2 and _polygon_area(rings) == 20 * 20 - 6 * 6


def test_pixels_touching_only_at_a_corner_are_separate_polygons():
    probability = np.zeros((8, 8), dtype=np.float32)
    probability[2:4, 2:4] = 1.0
    probability[4:6, 4:6] = 1.0
    assert len(trace_polygons(threshold_mask(probability, 0.5))) == 2
    assert trace_polygons(np.zeros((8, 8), dtype=np.uint8)) == []


def _distance_to_ring(point: np.ndarray, ring: np.ndarray) -> float:
    start, end = ring[:-1], ring[1:]
    edge = end - start
    t = np.clip(((point - start) * edge).sum(axis=1) / np.maximum((edge**2).sum(axis=1), 1e-12), 0, 1)
    return float(np.hypot(*(start + t[:, None] * edge - point).T).min())


def test_simplifying_keeps_every_dropped_vertex_within_the_tolerance():
    yy, xx = np.mgrid[:80, :80]
    probability = ((xx - 40) ** 2 + (yy - 40) ** 2 <= 30**2).astype(np.float32)  # a disc: a staircase outline
    ((exact, *_),) = trace_polygons(threshold_mask(probability, 0.5))
    simplified = simplify_ring(exact, 1.0)
    assert simplified is not None and len(simplified) < len(exact) / 3
    assert np.array_equal(simplified[0], simplified[-1])
    assert max(_distance_to_ring(vertex, simplified) for vertex in exact) <= 1.0 + 1e-9
    assert simplify_ring(exact, 0) is exact


def test_a_square_keeps_its_corners_and_a_sliver_collapses():
    ((exterior, *_),) = trace_polygons(threshold_mask(_square(5, 5, 20), 0.5))
    assert len(simplify_ring(exterior, 1.0)) == 5  # four corners, closed
    sliver = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 0.4], [0.0, 0.4], [0.0, 0.0]])
    assert simplify_ring(sliver, 1.0) is None


def test_small_polygons_are_dropped_and_the_large_one_is_kept():
    probability = np.maximum(_square(5, 5, 30), _square(50, 50, 5))  # 900 px and 25 px
    polygons = decode_polygons(probability, PostprocessParams(min_area_px=400, simplify_px=1.0))
    assert len(polygons) == 1 and _polygon_area(polygons[0]["rings"]) == pytest.approx(900)
    assert len(decode_polygons(probability, PostprocessParams(min_area_px=0))) == 2


def test_empty_and_missing_parameters_give_safe_defaults():
    assert decode_polygons(np.zeros((32, 32), dtype=np.float32)) == []
    (polygon,) = decode_polygons(_square(5, 5, 30, value=0.9))
    assert set(polygon) == {"rings", "confidence", "class_id"} and polygon["class_id"] == 1


def test_confidence_is_the_mean_probability_inside_without_the_hole():
    probability = _square(2, 2, 20, value=0.9)
    probability[8:14, 8:14] = 0.1  # below the threshold: a hole, whose pixels must not count
    ((rings),) = trace_polygons(threshold_mask(probability, 0.5))
    assert polygon_confidence(probability, rings) == pytest.approx(0.9)

import json

import numpy as np
import pytest

from conftest import CHIP_NAME, lon_lat, road_line
from hs_hot.augment import augment_chip_and_mask
from hs_hot.dataset import (
    BACKGROUND,
    ROAD,
    build_chip_labels,
    load_labels,
    rasterize_chip_labels,
    read_label_shapes,
    scale_chip_and_labels,
    spatial_split,
)
from hs_hot.params import AugmentParams, PreprocessParams

GSD = 0.1493  # metres per pixel of a zoom-20 tile near the equator, from the chip name


def _names(n_x: int, n_y: int):
    return [f"OAM-{x}-{y}-19.tif" for x in range(n_x) for y in range(n_y)]


def test_split_is_deterministic_and_disjoint():
    names = _names(16, 16)
    a = spatial_split(names, 0.2, 1)
    assert a == spatial_split(names, 0.2, 1) and a != spatial_split(names, 0.2, 2)
    train, val = a
    assert set(train).isdisjoint(val) and sorted(train + val) == sorted(names)


def test_whole_blocks_stay_together():
    train, val = spatial_split(_names(16, 16), 0.25, 0, block_size=4)
    block = lambda n: tuple(int(v) // 4 for v in n.split("-")[1:3])  # noqa: E731
    assert {block(n) for n in train}.isdisjoint({block(n) for n in val})


def test_non_oam_name_raises_and_empty_input():
    with pytest.raises(ValueError, match="OAM"):
        spatial_split(["OAM-1-2-3.tif", "a.tif"], 0.4, 0)
    assert spatial_split([], 0.2, 0) == ([], [])


def _write_labels(path, features, crs=None):
    collection = {"type": "FeatureCollection", "features": features}
    if crs:
        collection["crs"] = {"type": "name", "properties": {"name": crs}}
    path.write_text(json.dumps(collection))
    return path


def test_lines_and_multilines_are_read_and_other_geometries_are_skipped(tmp_path):
    square = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
    features = [
        {"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 2]]}},
        {"type": "Feature", "properties": {}, "geometry": {"type": "MultiLineString", "coordinates": [[[0, 0], [1, 1]], [[2, 2], [3, 3]]]}},
        {"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": [[0, 0]]}},
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [square]}},
        {"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [5, 6]}},
        {"type": "Feature", "properties": {}, "geometry": None},
    ]  # fmt: skip
    shapes, crs = read_label_shapes(_write_labels(tmp_path / "labels.geojson", features))
    assert crs == "EPSG:4326" and [shape["kind"] for shape in shapes] == ["line"] * 3
    assert shapes[0]["bounds"] == (0.0, 0.0, 1.0, 2.0) and shapes[0]["line"].shape == (2, 2)


def test_the_crs_of_the_file_and_malformed_files(tmp_path):
    assert read_label_shapes(_write_labels(tmp_path / "a.geojson", [], crs="EPSG:3857"))[1] == "EPSG:3857"
    (tmp_path / "bad.geojson").write_text(json.dumps({"type": "Feature"}))
    with pytest.raises(ValueError, match="FeatureCollection"):
        read_label_shapes(tmp_path / "bad.geojson")


def test_chips_get_the_roads_that_touch_them_in_chip_pixels(labelled_chips):
    chips, labels, names = labelled_chips
    built = build_chip_labels(chips, labels)
    assert sorted(built) == sorted(names)
    across, diagonal = built[names[0]]
    assert [across["kind"], diagonal["kind"]] == ["road", "road"]
    assert across["line"][:, 1].tolist() == pytest.approx([64, 64], abs=1e-3)
    assert across["line"][:, 0].tolist() == pytest.approx([-50, 306], abs=1e-3)
    assert diagonal["line"][0].tolist() == pytest.approx([20, 150], abs=1e-3)
    assert across["width_px"] == pytest.approx(
        6 / GSD, rel=0.01
    )  # a 6 m band at the chip's ground resolution
    assert len(built[names[1]]) == 2  # every chip has the same georeferencing here


def test_a_road_just_outside_the_chip_draws_its_band_in_and_a_far_one_does_not(write_chip, tmp_path):
    chip = write_chip(name=CHIP_NAME)
    near = road_line((-5, 0), (-5, 255))  # 5 m outside the left edge; the band reaches 20 px each side
    far = road_line((-90, 0), (-90, 255))
    labels = _write_labels(tmp_path / "edge.geojson", [near, far])
    (found,) = build_chip_labels(chip.parent, labels)[CHIP_NAME]
    assert found["line"][:, 0].tolist() == pytest.approx([-5, -5], abs=1e-3)
    mask = rasterize_chip_labels([found], 256, 256)
    assert mask[100, 10] == ROAD and mask[100, 20] == BACKGROUND  # the band reaches about 15 px in


def test_load_labels_passes_a_mapping_through_and_builds_from_a_path(labelled_chips):
    chips, labels, names = labelled_chips
    mapping = {names[0]: []}
    assert load_labels(chips, mapping) is mapping
    assert sorted(load_labels(chips, labels)) == sorted(names) and sorted(
        load_labels(chips, labels / "labels.geojson")
    ) == sorted(names)


def _road(points, width_px):
    return {"kind": "road", "line": np.array(points, dtype=float), "width_px": width_px}


def test_a_road_is_a_band_of_its_width_with_round_ends_and_nothing_elsewhere():
    mask = rasterize_chip_labels([_road([[5, 10], [35, 10]], 6.0)], 20, 40)
    assert mask.dtype == np.uint8 and set(np.unique(mask)) == {BACKGROUND, ROAD}
    assert (
        mask[10, 20] == ROAD
        and mask[8, 20] == ROAD
        and mask[5, 20] == BACKGROUND
        and mask[15, 20] == BACKGROUND
    )
    assert mask[10, 3] == ROAD and mask[10, 1] == BACKGROUND  # the round end reaches 3 px past the vertex
    assert not rasterize_chip_labels([], 5, 5).any()
    assert rasterize_chip_labels([_road([[0, 0], [9, 0]], 0.2)], 3, 10)[
        0
    ].all()  # a band is at least one pixel


def test_labels_are_drawn_at_the_scaled_resolution_with_a_band_that_scales_too():
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    road = _road([[0, 128], [256, 128]], 20.0)  # 20 px of the chip: about 3x that once scaled
    image, mask = scale_chip_and_labels(
        rgb, [road], PreprocessParams(model_gsd_m=0.05), "OAM-485915-499567-20.tif"
    )
    assert image.shape[1:] == mask.shape and image.shape[1] > 700  # zoom 20 is upscaled about 3x
    rows = np.flatnonzero(mask[:, mask.shape[1] // 2] == ROAD)
    assert set(np.unique(mask)) == {BACKGROUND, ROAD}
    assert rows.max() - rows.min() + 1 == pytest.approx(20.0 * image.shape[1] / 256, abs=2)


def test_augmentation_moves_image_and_mask_together_and_keeps_all_pixels():
    rng = np.random.default_rng(3)
    mask = np.random.default_rng(5).integers(0, 3, size=(48, 48)).astype(np.uint8)  # no symmetry of its own
    image = np.stack([mask, 2 * mask, 3 * mask]).astype(np.float32)
    seen = set()
    for _ in range(40):
        out_image, out_mask = augment_chip_and_mask(image, mask, rng, AugmentParams(orientation="dihedral"))
        assert out_image.flags["C_CONTIGUOUS"] and out_mask.flags["C_CONTIGUOUS"]
        assert np.array_equal(out_image[0], out_mask.astype(np.float32))  # still aligned
        assert np.array_equal(np.sort(out_mask.ravel()), np.sort(mask.ravel()))  # nothing lost or invented
        seen.add(out_mask.tobytes())
    assert len(seen) == 8  # the 4 quarter turns x mirror or not


def test_lon_lat_helper_is_consistent_with_the_chip_origin():
    assert lon_lat(0, 0) == pytest.approx([8.983, 17.676], abs=0.01)

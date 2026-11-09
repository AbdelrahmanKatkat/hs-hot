import math

import pytest

from hs_hot.tiles import chip_scale, parse_oam_tile, tile_gsd_m


def test_parse_oam_tile_reads_x_y_z_for_every_extension():
    for ext in ("tif", "tiff", "png", "jpg", "JPEG"):
        assert parse_oam_tile(f"OAM-10-20-19.{ext}") == (10, 20, 19)


@pytest.mark.parametrize("bad", ["chip.tif", "OAM-1-2.tif", "OAM-1-2-3-4.tif", "x/OAM-1-2-3.tif"])
def test_non_oam_names_raise(bad):
    with pytest.raises(ValueError, match="OAM"):
        parse_oam_tile(bad)


def test_tile_outside_the_zoom_grid_raises():
    with pytest.raises(ValueError, match="outside"):
        parse_oam_tile("OAM-0-8-3.tif")  # zoom 3 has rows 0-7


def test_gsd_at_the_equator_matches_the_web_mercator_constant():
    # zoom 20, 256 px tile: 156543.03392 / 2^20 = 0.14929 m/px at the equator
    assert tile_gsd_m("OAM-524288-524287-20.tif", 256) == pytest.approx(0.149291, rel=1e-4)


def test_gsd_shrinks_with_latitude_and_is_the_same_north_and_south():
    z = 19
    n = 2**z

    def row(lat_deg):  # XYZ row of a latitude
        return int((1 - math.asinh(math.tan(math.radians(lat_deg))) / math.pi) / 2 * n)

    equator = tile_gsd_m(f"OAM-1-{row(0.01)}-{z}.tif", 256)
    sixty_n = tile_gsd_m(f"OAM-1-{row(60.0)}-{z}.tif", 256)
    sixty_s = tile_gsd_m(f"OAM-1-{row(-60.0)}-{z}.tif", 256)
    assert sixty_n / equator == pytest.approx(0.5, rel=0.01)
    assert sixty_s == pytest.approx(sixty_n, rel=1e-3)


def test_gsd_scales_with_chip_width():
    assert tile_gsd_m("OAM-524288-524288-20.tif", 512) == pytest.approx(
        tile_gsd_m("OAM-524288-524288-20.tif", 256) / 2
    )


def test_chip_scale_is_gsd_over_model_gsd():
    scale = chip_scale("OAM-524288-524288-20.tif", 256, 0.05)
    assert scale == pytest.approx(0.149291 / 0.05, rel=1e-3)
    assert chip_scale("OAM-524288-524288-20.tif", 256, 0.10) == pytest.approx(scale / 2)


def test_max_input_px_caps_the_longest_side_of_the_scaled_chip():
    zoom18 = "OAM-193343-110117-18.tif"  # latitude 27.6: 0.53 m/px, uncapped scale 10.6
    assert chip_scale(zoom18, 256, 0.05) == pytest.approx(10.58, abs=0.02)
    assert chip_scale(zoom18, 256, 0.05, max_input_px=1024) == pytest.approx(4.0)
    assert chip_scale(zoom18, 256, 0.05, height_px=512, max_input_px=1024) == pytest.approx(
        2.0
    )  # longer side
    zoom20 = "OAM-524288-524288-20.tif"  # about 3x: 765 px, below the cap, so unchanged
    assert chip_scale(zoom20, 256, 0.05, max_input_px=1024) == chip_scale(zoom20, 256, 0.05)


def test_chip_scale_rejects_zooms_that_are_too_coarse():
    with pytest.raises(ValueError, match="too coarse"):
        chip_scale(
            "OAM-262144-262144-19.tif", 256, 0.01, max_input_px=1024
        )  # x30 is refused, not just capped

"""Skeleton masks to road graphs and the small-fragment filter (tiny generated masks, offline)."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from hs_hot.road_graph import drop_small_fragments, points_along, remove_small_fragments, skeleton_to_graph

H, W, ROW = 160, 320, 20


def _road(length: int) -> np.ndarray:
    mask = np.zeros((H, W), bool)
    mask[ROW, 0 : length + 1] = True
    return mask


def test_a_road_is_two_nodes_and_one_edge_with_its_length():
    g = skeleton_to_graph(_road(100), 1.0)
    assert len(g) == 2 and len(g.edges) == 1
    assert g.edges[0][3] == pytest.approx(100.0)
    assert g.xy[:, 0].min() == 0 and g.xy[:, 0].max() == 100


def test_metres_per_pixel_scales_coordinates_and_lengths():
    g = skeleton_to_graph(_road(100), 0.3)
    assert g.edges[0][3] == pytest.approx(30.0)


def test_a_t_junction_is_four_nodes_and_three_unmerged_edges():
    t = _road(300)
    t[ROW + 1 : ROW + 101, 150] = True  # the branch touches the road diagonally next to the junction
    g = skeleton_to_graph(t, 1.0)
    assert len(g) == 4 and len(g.edges) == 3
    assert sorted(round(e[3]) for e in g.edges) == [100, 150, 150]
    assert g.untraced == 0 and g.dead_ends == 0


def test_a_staircase_and_a_square_corner_are_single_edges():
    stairs = np.zeros((H, W), bool)
    for i in range(15):  # 2:1 line drawn as horizontal pairs joined diagonally
        stairs[ROW + i, 10 + 2 * i] = stairs[ROW + i, 11 + 2 * i] = True
    corner = np.zeros((H, W), bool)
    corner[ROW, 10:31] = True
    corner[ROW : ROW + 21, 30] = True  # the corner pixel is redundant
    for mask in (stairs, corner):
        g = skeleton_to_graph(mask, 1.0)
        assert (len(g), len(g.edges), g.untraced, g.dead_ends) == (2, 1, 0, 0)


def test_a_ring_is_one_node_with_a_loop_edge():
    ring = np.zeros((H, W), bool)
    ring[30, 30:50] = ring[50, 30:50] = ring[30:51, 30] = ring[30:51, 49] = True
    g = skeleton_to_graph(ring, 1.0)
    assert len(g) == 1 and len(g.edges) == 1 and g.edges[0][0] == g.edges[0][1]


def test_an_isolated_pixel_two_neighbouring_pixels_and_an_empty_mask():
    dot = np.zeros((H, W), bool)
    dot[10, 10] = True
    two = np.zeros((H, W), bool)
    two[10, 10:12] = True
    g_dot, g_two = skeleton_to_graph(dot, 1.0), skeleton_to_graph(two, 1.0)
    assert (len(g_dot), len(g_dot.edges)) == (1, 0)
    assert (len(g_two), len(g_two.edges)) == (2, 1)
    assert len(skeleton_to_graph(np.zeros((H, W), bool), 1.0)) == 0


def test_points_along_a_polyline_clamps_at_the_end():
    line = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    out = points_along(line, np.array([5.0, 15.0, 99.0]))
    assert out.tolist() == [[5.0, 0.0], [10.0, 5.0], [10.0, 10.0]]


def test_fragment_filter_drops_short_pieces_and_isolated_pixels_only():
    m = np.zeros((H, W), bool)
    m[ROW, 0:5] = True  # 4 m
    m[ROW + 40, 0:31] = True  # 30 m
    m[100, 100] = True
    kept = remove_small_fragments(m, 1.0, 5.0)
    assert not kept[ROW, 0:5].any() and kept[ROW + 40, 0:31].all() and not kept[100, 100]
    assert remove_small_fragments(m, 1.0, 0.0).sum() == m.sum()  # 0 switches it off


def _ring(side_rows: int, side_cols: int, tail: int = 0) -> np.ndarray:
    mask = np.zeros((H, W), bool)
    mask[30, 30 : 30 + side_cols + 1] = mask[30 + side_rows, 30 : 30 + side_cols + 1] = True
    mask[30 : 30 + side_rows + 1, 30] = mask[30 : 30 + side_rows + 1, 30 + side_cols] = True
    if tail:
        mask[30 + side_rows : 30 + side_rows + tail + 1, 30 + side_cols // 2] = True
    return mask


def test_a_closed_ring_is_kept_when_it_is_long_enough_whatever_its_nodes():
    """The official filter measures the longest path between two nodes; a ring has one node and no pair, so it deletes every ring, a roundabout included."""
    big, small = _ring(40, 60), _ring(1, 1)
    assert remove_small_fragments(big, 1.0, 5.0).sum() == big.sum()  # 200 m round
    assert remove_small_fragments(small, 1.0, 5.0).sum() == 0  # 4 m round: really too short
    lollipop = _ring(40, 60, tail=2)  # a big ring on a 2 m tail: the tail is the only pair of nodes
    assert remove_small_fragments(lollipop, 1.0, 5.0).sum() == lollipop.sum()


def test_parallel_roads_between_the_same_nodes_count_by_their_longest_road():
    lens = np.zeros((H, W), bool)
    lens[20, 40:61] = lens[26, 40:61] = True
    lens[20:27, 40] = lens[20:27, 60] = True  # a closed loop of 52 m, two nodes would be 20 m apart
    lens[23, 30:41] = lens[23, 60:71] = True  # with a stub each side
    assert remove_small_fragments(lens, 1.0, 5.0).sum() == lens.sum()


def test_drop_small_fragments_counts_what_it_removed():
    m = np.zeros((H, W), bool)
    m[ROW, 0:4] = True
    m[ROW + 40, 0:4] = True
    m[ROW + 80, 0:60] = True
    kept, removed = drop_small_fragments(m, 1.0, 5.0)
    assert removed == 2 and kept.sum() == 60
    assert drop_small_fragments(m, 1.0, 0.0)[1] == 0

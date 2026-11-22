"""Image-space measures on hand-built drawings whose answer is known."""
import math

import networkx as nx
import numpy as np
import pytest
import torch

from data.raster import raster
from metrics.image_quality import (_pairs_sharing_pixels, band_pixels, edge_element, image_quality,
                                   node_element)

W = H = 80


def P(*xy):
    return torch.tensor(xy, dtype=torch.float64)


def dense(element, width=W, height=H):
    img = np.zeros(width * height)
    img[element.pixels] = element.coverage
    return img


# --- coverage of the elements ------------------------------------------------------------------------------
@pytest.mark.parametrize("radius", [0.7, 2.0, 4.0, 9.5])
def test_node_coverage_sums_to_the_disk_area(radius):
    e = node_element((40.3, 39.8), radius, W, H)
    assert e.coverage.sum() == pytest.approx(math.pi * radius ** 2, rel=1e-9)
    assert len(np.unique(e.pixels)) == len(e.pixels)


@pytest.mark.parametrize("seed", range(6))
def test_sparse_edge_coverage_equals_the_full_grid(seed):
    """band_pixels must not miss any pixel the edge touches: compare with the coverage of EVERY pixel."""
    rng = np.random.default_rng(seed)
    a, b = rng.uniform(-5, W + 5, 2), rng.uniform(-5, W + 5, 2)  # may leave the frame
    w = float(rng.choice([0.6, 1.0, 2.5, 4.0]))
    Y, X = np.mgrid[0:H, 0:W].astype(float)
    full = np.clip(raster.edge_coverage(X, Y, a + 0.5, b + 0.5, w), 0, 1).ravel()
    np.testing.assert_allclose(dense(edge_element(a, b, w, W, H)), full, atol=1e-9)


def test_band_pixels_stay_in_frame_and_are_unique():
    X, Y = band_pixels(np.array([[-10.0, 5.0], [90.0, 70.0]]), 3.0, W, H)
    assert X.min() >= 0 and X.max() < W and Y.min() >= 0 and Y.max() < H
    assert len(set(zip(X.tolist(), Y.tolist()))) == len(X)


def test_pairs_sharing_pixels_brute_force():
    rng = np.random.default_rng(0)
    pa, oa = rng.integers(0, 30, 200), rng.integers(0, 7, 200)
    pb, ob = rng.integers(0, 30, 150), rng.integers(0, 5, 150)
    expected = {(int(x), int(y)) for p, x in zip(pa, oa) for q, y in zip(pb, ob) if p == q}
    got = {tuple(map(int, r)) for r in _pairs_sharing_pixels(pa, oa, pb, ob)}
    assert got == expected


# --- visibility ---------------------------------------------------------------------------------------------
def test_single_node_fully_visible(make_graph):
    g = make_graph([], n=1, node_size=5.0)
    q = image_quality(P([40, 40]), g, W, H)
    assert q["node_visibility"] == pytest.approx(1.0, abs=1e-9)
    assert q["hidden_nodes"] == 0 and q["merged_nodes"] == 0


def test_node_half_outside_the_frame(make_graph):
    g = make_graph([], n=1, node_size=5.0)
    q = image_quality(P([-0.5, 40]), g, W, H)  # center on the left border of the frame
    assert q["node_visibility"] == pytest.approx(0.5, abs=1e-9)


def test_overlap_follows_the_lens_area(make_graph):
    r, d = 10.0, 10.0
    g = make_graph([], n=2, node_size=r)
    q = image_quality(P([30, 40], [30 + d, 40]), g, W, H)
    lens = 2 * r * r * math.acos(d / (2 * r)) - d / 2 * math.sqrt(4 * r * r - d * d)
    assert q["node_visibility"] == pytest.approx(1 - lens / (math.pi * r * r), abs=0.02)
    assert q["hidden_nodes"] == 0  # 61 % visible each


def test_hidden_node(make_graph):
    g = make_graph([], n=2, node_size=[3.0, 8.0])
    q = image_quality(P([40, 40], [40.5, 40]), g, W, H)  # the small one inside the big one
    # the small one is hidden; the big one keeps 1 - 9/64 = 86 % of its disk
    assert q["hidden_nodes"] == pytest.approx(0.5)
    assert q["node_visibility"] == pytest.approx((0 + (1 - 9 / 64)) / 2, abs=0.02)


# --- merged nodes: the effect a geometric measure cannot see -------------------------------------------------
def test_nodes_merge_without_overlapping(make_graph):
    r = 4.0
    g = make_graph([], n=2, node_size=r)
    close = image_quality(P([30, 40], [30 + 2 * r + 0.3, 40]), g, W, H)  # 0.3 px gap: no geometric overlap
    far = image_quality(P([30, 40], [30 + 2 * r + 3.0, 40]), g, W, H)
    assert close["merged_nodes"] == pytest.approx(0.5)  # 2 nodes, 1 blob
    assert far["merged_nodes"] == 0


def test_merged_counts_blobs(make_graph):
    g = make_graph([], n=4, node_size=3.0)
    # three in a touching row, one apart
    q = image_quality(P([20, 20], [26, 20], [32, 20], [60, 60]), g, W, H)
    assert q["merged_nodes"] == pytest.approx(2 / 4)


# --- ambiguous edges ----------------------------------------------------------------------------------------
def test_edge_through_a_node_is_ambiguous(make_graph):
    g = make_graph([(0, 1)], n=3, node_size=3.0)
    through = image_quality(P([10, 40], [70, 40], [40, 40]), g, W, H)
    # stroke over y in [40, 41], disk from y = 41.1: no contact, but both in the pixel row [40.5, 41.5]
    near = image_quality(P([10, 40.5], [70, 40.5], [40, 44.1]), g, W, H)
    away = image_quality(P([10, 40], [70, 40], [40, 50]), g, W, H)
    assert through["ambiguous_edges"] == 1
    assert near["ambiguous_edges"] == 1  # does not touch, but shares a pixel: looks connected
    assert away["ambiguous_edges"] == 0


def test_edge_ends_are_not_ambiguous(make_graph):
    g = make_graph([(0, 1), (1, 2)], node_size=4.0)
    q = image_quality(P([10, 10], [40, 40], [70, 10]), g, W, H)
    assert q["ambiguous_edges"] == 0


# --- edge overlap -------------------------------------------------------------------------------------------
def test_edge_overlap(make_graph):
    g = make_graph([(0, 1), (2, 3)], node_size=2.0)
    on_top = image_quality(P([10, 30], [70, 30], [15, 30], [65, 30]), g, W, H)
    # 0.5 px apart: rows [29.5, 30.5] and [30, 31] share the pixel row 30, row 31 is only the second one's
    shifted = image_quality(P([10, 30], [70, 30], [10, 30.5], [70, 30.5]), g, W, H)
    apart = image_quality(P([10, 30], [70, 30], [10, 50], [70, 50]), g, W, H)
    crossing = image_quality(P([10, 10], [70, 70], [10, 70], [70, 10]), g, W, H)
    assert on_top["edge_overlap"] > 0.95
    assert shifted["edge_overlap"] == pytest.approx(0.5, abs=0.05)
    assert apart["edge_overlap"] == 0
    assert 0 < crossing["edge_overlap"] < 0.1  # a few pixels around the crossing


# --- clutter ------------------------------------------------------------------------------------------------
def test_ink_density_of_one_node(make_graph):
    g = make_graph([], n=1, node_size=5.0)
    q = image_quality(P([40, 40]), g, W, H)
    assert q["ink_density"] == pytest.approx(math.pi * 25 / (W * H), rel=1e-6)
    assert q["ink_peak"] == pytest.approx(min(1.0, math.pi * 25 / 256), rel=0.05)


# --- invariances --------------------------------------------------------------------------------------------
def test_integer_translation_changes_nothing(make_graph):
    g = make_graph([(0, 1), (1, 2), (2, 0), (2, 3)], node_size=3.0)
    pos = P([20.2, 20.7], [27.1, 21.3], [24.0, 27.9], [35.5, 30.0])
    a = image_quality(pos, g, W, H)
    b = image_quality(pos + torch.tensor([13.0, 7.0], dtype=torch.float64), g, W, H)
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=1e-9), k


# --- real graphs: in the easy setting of the benchmark, Kamada-Kawai has no defect of the nodes --------------
def test_kamada_kawai_on_rome_at_500px(rome_dir):
    from data.input import load_graphml
    for path in sorted(rome_dir.glob("*.graphml"))[:3]:
        g = load_graphml(path)  # node radius 4 px, edges 1 px: the benchmark setting
        G = nx.Graph(g.edge_index.t().tolist())
        layout = nx.kamada_kawai_layout(G)
        X = torch.tensor(np.array([layout[i] for i in range(g.num_nodes)]))
        X = X - X.mean(0)
        X = X * (250 - 4) / X.abs().max() + 250  # as benchmark/compare.to_frame
        q = image_quality(X, g, 500, 500)
        assert q["hidden_nodes"] == 0
        assert q["node_visibility"] > 0.99


# --- defect mask --------------------------------------------------------------------------------------------
def test_defect_mask_marks_exactly_the_counted_defects(make_graph):
    from metrics.image_quality import defect_mask
    g = make_graph([(0, 1)], n=5, node_size=3.0)
    clean = P([10, 10], [30, 10], [60, 60], [10, 60], [40, 40])
    assert defect_mask(clean, g, W, H).sum() == 0
    # node 2 on edge (0, 1); nodes 3 and 4 merged
    bad = P([10, 10], [30, 10], [20, 10], [10, 60], [16.2, 60])  # 0.2 px between nodes 3 and 4
    mask = defect_mask(bad, g, W, H)
    assert mask[10, 20] == 1  # where the edge crosses node 2
    assert image_quality(bad, g, W, H)["merged_nodes"] == pytest.approx(1 / 5)
    assert mask[60, 10] == 1 and mask[60, 16] == 1  # the merged blob
    assert mask[40, 40] == 0 and mask[10, 30] == 0  # elsewhere: clean (node 1 is an end of the edge)

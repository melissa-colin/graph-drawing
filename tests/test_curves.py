"""Curved edges: geometry, SDF, rendering, losses, optimization."""
import math

import numpy as np
import pytest
import torch

from curves.bezier import bend, capsule_sdf, curve_sdf, sample, segment_distance
from curves.losses import (count_curve_crossings, curve_bend, curve_crossings, curve_curve_overlaps,
                           curve_node_overlaps)
from curves.optimize import optimize_curves
from curves.render import curve_element, curve_elements, render_curved
from data.output import Layout
from metrics.edge_crossings import count_crossings
from metrics.image_quality import image_quality
from metrics.overlaps_vectorised import edge_node_overlaps, edge_overlaps

W = H = 80


def T(*xy):
    return torch.tensor(xy, dtype=torch.float64)


# --- geometry -----------------------------------------------------------------------------------------------
def test_sample_ends_and_straight_case():
    pos = T([0, 0], [8, 4])
    ei = torch.tensor([[0], [1]])
    pts = sample(pos, ei, torch.zeros(1, 2, dtype=torch.float64), 4)[0]
    assert torch.allclose(pts[0], pos[0]) and torch.allclose(pts[-1], pos[1])
    assert torch.allclose(pts, torch.stack([pos[0] + t * (pos[1] - pos[0]) for t in (0, .25, .5, .75, 1)]))
    curved = sample(pos, ei, T([0, 6]), 2)[0]
    assert torch.allclose(curved[1], (pos[0] + pos[1]) / 2 + T([0, 3]))  # B(1/2) = mid + o / 2


def test_polyline_converges_to_the_curve():
    """Distance to the K-segment polyline vs to a 4000-segment one: the error falls like 1 / K^2."""
    pos, ei, off = T([0, 0], [60, 0]), torch.tensor([[0], [1]]), T([0, 50])
    rng = np.random.default_rng(0)
    px, py = (torch.tensor(rng.uniform(lo, hi, 400)) for lo, hi in ((-10, 70), (-10, 40)))
    w = torch.tensor(0.0, dtype=torch.float64)
    exact = curve_sdf(px, py, sample(pos, ei, off, 4000)[0], w)
    errors = [float((curve_sdf(px, py, sample(pos, ei, off, k)[0], w) - exact).abs().max()) for k in (4, 8, 16)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[1] / errors[2] > 3  # about 4 for a quadratic convergence
    assert errors[1] < 0.6  # K = 8 on a 60 px edge bent by 25 px: under one pixel


def test_straight_curve_is_a_capsule():
    pos, ei = T([10, 10], [50, 30]), torch.tensor([[0], [1]])
    pts = sample(pos, ei, torch.zeros(1, 2, dtype=torch.float64))[0]
    px, py = torch.meshgrid(torch.linspace(0, 60, 31, dtype=torch.float64),
                            torch.linspace(0, 40, 21, dtype=torch.float64), indexing="ij")
    w = torch.tensor(3.0, dtype=torch.float64)
    expected = capsule_sdf(px, py, (pos[0, 0], pos[0, 1]), (pos[1, 0], pos[1, 1]), w)
    assert torch.allclose(curve_sdf(px, py, pts, w), expected, atol=1e-6)


def test_bend_is_scale_free():
    pos, ei = T([0, 0], [10, 0]), torch.tensor([[0], [1]])
    assert float(bend(pos, ei, T([0, 5]))) == pytest.approx(float(bend(3 * pos, ei, T([0, 15]))), rel=1e-6)


# --- rendering ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("w", [1.0, 2.0, 4.0])
def test_curve_coverage_is_the_stroke_area(w):
    pos, ei = T([15, 40], [65, 40]), torch.tensor([[0], [1]])
    pts = sample(pos, ei, T([0, 30]), 32)[0]
    length = float((pts[1:] - pts[:-1]).norm(dim=1).sum())
    area = length * w + math.pi * (w / 2) ** 2  # stroke with round caps
    assert curve_element(pts.numpy(), w, W, H, supersample=8).coverage.sum() == pytest.approx(area, rel=0.02)


def test_straight_curves_measure_like_straight_edges(make_graph):
    g = make_graph([(0, 1)], n=3, node_size=3.0)
    for third in ([40, 40], [40, 44.1], [40, 52]):
        pos = T([10, 40.5], [70, 40.5], third)
        straight = image_quality(pos, g, W, H)
        curved = image_quality(pos, g, W, H, edges=curve_elements(pos, torch.zeros(1, 2), g, W, H))
        assert curved["ambiguous_edges"] == straight["ambiguous_edges"]


def test_render_curved_draws_nodes_on_top(make_graph):
    g = make_graph([(0, 1)], node_size=4.0, edge_width=2.0)
    img = render_curved(T([20, 40], [60, 40]), T([0, 30]), g, W, H)
    assert img.shape == (H, W, 3)
    assert img[40, 20].max() < 0.7  # node center inked
    assert img[40, 40].min() > 0.99  # the straight path is empty: the edge bends away
    assert img[40 + 15, 40].max() < 0.8  # the curve passes 15 px lower (B(1/2) = mid + o / 2)


# --- losses -------------------------------------------------------------------------------------------------
def test_straight_curve_over_node_matches_the_straight_loss(make_graph):
    g = make_graph([(0, 1)], n=3, node_size=4.0, edge_width=2.0)
    layout = Layout(T([10, 40], [70, 40], [40, 42]), W, H)
    straight = float(edge_node_overlaps(layout, g))
    curved = float(curve_node_overlaps(layout, torch.zeros(1, 2, dtype=torch.float64), g))
    assert straight > 1 and curved == pytest.approx(straight, rel=0.02)


def test_curve_node_overlaps_vanish_when_the_curve_avoids_the_node(make_graph):
    g = make_graph([(0, 1)], n=3, node_size=4.0, edge_width=2.0)
    layout = Layout(T([10, 40], [70, 40], [40, 40]), W, H)
    assert float(curve_node_overlaps(layout, T([0, 40]), g)) == 0.0  # the curve passes 20 px below


def test_curve_curve_overlaps_match_the_straight_value(make_graph):
    g = make_graph([(0, 1), (2, 3)], node_size=2.0, edge_width=2.0)
    layout = Layout(T([10, 10], [70, 70], [10, 70], [70, 10]), W, H)
    straight = float(edge_overlaps(layout, g))
    curved = float(curve_curve_overlaps(layout, torch.zeros(2, 2, dtype=torch.float64), g))
    # the crossing is at t = 1/2 of both curves, a joint of both polylines: still counted once
    assert curved == pytest.approx(straight, rel=0.01)
    assert float(curve_curve_overlaps(Layout(T([10, 10], [70, 10], [10, 60], [70, 60]), W, H),
                                      torch.zeros(2, 2, dtype=torch.float64), g)) == 0.0


def test_crossings_of_curves(make_graph):
    g = make_graph([(0, 1), (2, 3)], node_size=2.0)
    layout = Layout(T([10, 40], [70, 40], [40, 20], [40, 70]), W, H)
    zero = torch.zeros(2, 2, dtype=torch.float64)
    assert count_curve_crossings(layout, zero, g) == float(count_crossings(layout, g)) == 1
    # smallest distance from an end to the other curve: (40, 20) is 20 px from the horizontal edge
    assert float(curve_crossings(layout, zero, g)) == pytest.approx(20.0)
    # the crossing point (40, 40) is a joint of the horizontal polyline (t = 4/8): counted once, as above
    above = Layout(T([10, 40], [70, 40], [40, 10], [40, 35]), W, H)
    assert count_curve_crossings(above, zero, g) == 0
    assert float(curve_crossings(above, zero, g)) == 0.0


@pytest.mark.parametrize("term", [curve_node_overlaps, curve_curve_overlaps, curve_crossings, curve_bend])
def test_gradcheck_offsets(make_graph, term):
    # edge (2, 3) crosses edge (0, 1); node 4 lies 1.5 px from edge (0, 1), which does not end at it
    g = make_graph([(0, 1), (2, 3), (1, 4)], node_size=4.0, edge_width=2.0)
    layout = Layout(T([10, 40], [70, 40], [40, 25], [42, 70], [25, 41.5]), W, H)
    off = T([1.0, 3.0], [-2.0, 1.0], [0.5, -0.5]).requires_grad_(True)
    assert float(term(layout, off, g)) > 0
    assert torch.autograd.gradcheck(lambda o: term(layout, o, g), (off,), eps=1e-6, atol=1e-4)


# --- optimization -------------------------------------------------------------------------------------------
def test_bending_removes_an_edge_over_a_node(make_graph):
    """An edge passes over a node it does not connect: bending it makes the image unambiguous."""
    g = make_graph([(0, 1)], n=3, node_size=4.0, edge_width=1.0)
    layout = Layout(T([10, 40], [70, 40], [40, 41]).float(), W, H)
    assert image_quality(layout.positions, g, W, H)["ambiguous_edges"] == 1
    weights = {"curve_node_overlaps": 1.0, "curve_bend": 1.0}
    new_layout, off = optimize_curves(layout, g, weights, steps=150, lr=1.0)
    assert torch.equal(new_layout.positions, layout.positions)  # nodes fixed
    q = image_quality(layout.positions, g, W, H, edges=curve_elements(layout.positions, off, g, W, H))
    assert q["ambiguous_edges"] == 0
    assert 0 < float(bend(layout.positions, g.edge_index, off)) < 0.5  # bent, not wildly


def test_control_points_stay_in_frame(make_graph):
    g = make_graph([(0, 1)], n=3, node_size=4.0)
    layout = Layout(T([10, 6], [70, 6], [40, 7]).float(), W, H)  # pushing the edge up would leave the frame
    _, off = optimize_curves(layout, g, {"curve_node_overlaps": 1.0}, steps=100, lr=2.0, offsets=T([0, -30]).float())
    c = (layout.positions[0] + layout.positions[1]) / 2 + off[0]
    assert 4 - 1e-4 <= float(c[1]) <= H - 4 + 1e-4


def test_joint_optimization_moves_nodes(make_graph):
    from resolution import make_loss
    g = make_graph([(0, 1)], n=3, node_size=4.0)
    layout = Layout(T([10, 40], [70, 40], [40, 41]).float(), W, H)
    new_layout, _ = optimize_curves(layout, g, {"curve_node_overlaps": 1.0}, steps=20, joint=True,
                                    position_loss=make_loss("screen", {"node_overlaps": 1.0}))
    assert not torch.equal(new_layout.positions, layout.positions)
    with pytest.raises(ValueError):
        optimize_curves(layout, g, {"curve_bend": 1.0}, joint=True)


def test_bending_uncrosses_two_edges(make_graph):
    """Nodes fixed: the horizontal edge bends past the end of the short vertical one."""
    g = make_graph([(0, 1), (2, 3)], node_size=2.0)
    layout = Layout(T([10, 40], [70, 40], [40, 30], [40, 48]).float(), W, H)
    assert count_curve_crossings(layout, torch.zeros(2, 2), g) == 1
    _, off = optimize_curves(layout, g, {"curve_crossings": 1.0, "curve_bend": 1.0}, steps=200, lr=1.0)
    assert count_curve_crossings(layout, off, g) == 0


def test_curves_setting_end_to_end(make_graph):
    """Tiny experiment: straight drawings where edges cross nodes; curving must lower the ambiguity."""
    from curves.experiment import curves_setting
    from resolution import rescale_to_frame, with_style
    torch.manual_seed(0)
    edges = [(0, 1), (2, 3), (4, 5)]
    def drawing(i):
        g = with_style(make_graph(edges, name=f"c{i}"), 4.0, 1.0)
        # each edge passes over the start of the next one
        P = torch.tensor([[10., 40.], [70., 40.], [40., 41.], [40., 70.], [20., 69.], [65., 69.]]) + i
        return rescale_to_frame(P, 80, 4.0), g
    val, test = [drawing(0), drawing(1)], [drawing(2), drawing(3)]
    rows, selection, offsets = curves_setting(80, val, test, fractions=(1.0,), learning_rates=(1.0,), steps=60,
                                              batch_size=2)
    straight = [r for r in rows if r["edges"] == "straight"]
    curved = [r for r in rows if r["edges"] == "curved"]
    assert len(straight) == len(curved) == 2 and len(offsets) == 2
    assert sum(r["ambiguous_edges"] for r in curved) < sum(r["ambiguous_edges"] for r in straight)
    assert all(r["bend"] == 0 for r in straight) and all(r["bend"] > 0 for r in curved)
    assert selection["weights"]["curve_bend"] == 1.0

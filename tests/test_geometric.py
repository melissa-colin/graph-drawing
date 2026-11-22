"""Geometric control terms: zero without contact, the right value with, gradients that separate."""
import math

import pytest
import torch
from torch_geometric.data import Batch

from data.output import Layout
from metrics.geometric import edge_node_overlaps, edge_overlaps, node_overlaps


def L(*xy, size=100):
    return Layout(torch.tensor(xy, dtype=torch.float64, requires_grad=True), size, size)


def test_node_overlaps_value_and_zero(make_graph):
    g = make_graph([], n=2, node_size=[3.0, 4.0])
    assert float(node_overlaps(L([10, 10], [20, 10]), g)) == 0.0
    assert float(node_overlaps(L([10, 10], [15, 10]), g)) == pytest.approx((7 - 5) ** 2)


def test_node_overlaps_gradient_separates(make_graph):
    g = make_graph([], n=2, node_size=4.0)
    layout = L([10, 10], [14, 10])
    node_overlaps(layout, g).backward()
    grad = layout.positions.grad
    assert grad[0, 0] > 0 and grad[1, 0] < 0  # descent moves 0 left, 1 right
    assert grad[:, 1].abs().max() < 1e-9


def test_edge_node_overlaps(make_graph):
    g = make_graph([(0, 1)], n=3, node_size=3.0, edge_width=2.0)
    assert float(edge_node_overlaps(L([0, 0], [40, 0], [20, 10]), g)) == 0.0
    # node 2 at distance 2 of the edge: reach r + w/2 = 4, penetration 2
    assert float(edge_node_overlaps(L([0, 0], [40, 0], [20, 2]), g)) == pytest.approx(4.0)
    # beyond the end of the segment: distance to the end point
    assert float(edge_node_overlaps(L([0, 0], [40, 0], [43, 0]), g)) == pytest.approx(1.0)


def test_edge_node_ignores_the_ends(make_graph):
    g = make_graph([(0, 1)], node_size=5.0)
    assert float(edge_node_overlaps(L([0, 0], [3, 0]), g)) == 0.0


def test_edge_overlaps_crossing_is_the_parallelogram(make_graph):
    g = make_graph([(0, 1), (2, 3)], node_size=1.0, edge_width=2.0)
    square = float(edge_overlaps(L([0, 0], [40, 40], [0, 40], [40, 0]), g))
    assert square == pytest.approx(2 * 2 / 1.0)  # perpendicular: w^2
    angle = math.radians(30)
    slanted = L([0, 20], [40, 20], [20 - 20 * math.cos(angle), 20 - 20 * math.sin(angle)],
                [20 + 20 * math.cos(angle), 20 + 20 * math.sin(angle)])
    assert float(edge_overlaps(slanted, g)) == pytest.approx(4 / math.sin(angle), rel=1e-6)


def test_edge_overlaps_crossing_gradient_opens_the_angle(make_graph):
    g = make_graph([(0, 1), (2, 3)], node_size=1.0, edge_width=2.0)
    angle = math.radians(20)
    layout = L([0, 20], [40, 20], [20 - 20 * math.cos(angle), 20 - 20 * math.sin(angle)],
               [20 + 20 * math.cos(angle), 20 + 20 * math.sin(angle)])
    edge_overlaps(layout, g).backward()
    # moving node 3 against the gradient must increase the angle, i.e. lower the cost
    with torch.no_grad():
        moved = layout.positions - 0.5 * layout.positions.grad / layout.positions.grad.norm()
    assert float(edge_overlaps(Layout(moved, 100, 100), g)) < float(edge_overlaps(layout, g))


def test_edge_overlaps_near_parallel(make_graph):
    g = make_graph([(0, 1), (2, 3)], node_size=1.0, edge_width=2.0)
    assert float(edge_overlaps(L([0, 0], [40, 0], [0, 5], [40, 5]), g)) == 0.0
    assert float(edge_overlaps(L([0, 0], [40, 0], [0, 1.5], [40, 1.5]), g)) == pytest.approx(0.5 ** 2)


@pytest.mark.parametrize("term", [node_overlaps, edge_node_overlaps, edge_overlaps])
def test_gradcheck(make_graph, term):
    torch.manual_seed(0)
    g = make_graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4)], node_size=4.0, edge_width=2.0)
    P = (torch.rand(5, 2, dtype=torch.float64) * 12).requires_grad_(True)  # crowded: every term active
    assert float(term(Layout(P, 100, 100), g)) > 0
    assert torch.autograd.gradcheck(lambda p: term(Layout(p, 100, 100), g), (P,), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("term", [node_overlaps, edge_node_overlaps, edge_overlaps])
def test_batch_is_the_mean_of_the_graphs(make_graph, term):
    torch.manual_seed(1)
    g1 = make_graph([(0, 1), (1, 2), (2, 3), (3, 0)], node_size=4.0, name="a")
    g2 = make_graph([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4)], node_size=3.0, name="b")
    P1, P2 = torch.rand(4, 2) * 15, torch.rand(5, 2) * 15
    single = (float(term(Layout(P1, 100, 100), g1)) + float(term(Layout(P2, 100, 100), g2))) / 2
    batch = Batch.from_data_list([g1, g2])
    assert float(term(Layout(torch.cat((P1, P2)), 100, 100), batch)) == pytest.approx(single, rel=1e-5)

"""Smoke tests of the team's code that this work builds on (loss, refinement): they must keep working on any
image size and node radius, since the resolution sweep refines drawings far from the 500 px / 4 px setting."""
import pytest
import torch

from data.output import Layout
from loss import Loss
from refine import refine
from train import REFINE_WEIGHTS


@pytest.mark.parametrize("size,radius", [(100, 2.0), (200, 6.0), (500, 4.0)])
def test_refine_lowers_the_loss_and_stays_in_frame(make_graph, size, radius):
    torch.manual_seed(0)
    # a 4 x 4 grid, drawn at random in a corner: overlapping nodes and crossings to remove
    edges = [(r * 4 + c, r * 4 + c + 1) for r in range(4) for c in range(3)] + \
            [(r * 4 + c, (r + 1) * 4 + c) for r in range(3) for c in range(4)]
    g = make_graph(edges, node_size=radius)
    start = Layout(radius + torch.rand(16, 2) * size / 4, size, size)
    loss_fn = Loss(REFINE_WEIGHTS)
    before = float(loss_fn(start, g)[0])
    after_layout = refine(start, g, loss_fn, steps=60, lr=size / 500, margin=radius)
    after = float(loss_fn(after_layout, g)[0])
    assert torch.isfinite(after_layout.positions).all()
    assert after < before
    p = after_layout.positions
    assert (p >= radius - 1e-4).all() and (p <= size - radius + 1e-4).all()
